// stochastic gradient descent tricks, http://leno.bottou.org

#include <iostream>
#include <cmath>
#include <memory>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include "data.h"
using namespace std;
class LR {
public:
    double** scale(double**x, int m, int n) {
        double** scale_x = dmatrix(m, n);
        for(int i = 0; i < n ; ++i) { //feature
            double mean = 0.0;
            double var = 0.0;
            for(int j = 0; j < m; j++) {
                mean += x[j][i];
                var += x[j][i] * x[j][i];
            }
            mean = mean / m;
            var = var / m - mean * mean;
            for(int j = 0; j < m; j++) {
                scale_x[j][i] = (x[j][i] - mean) / var;
            }
        }
        return scale_x;
    }
    static double inner_prod(const double* v1, const double* v2, int n) {
        double r = 0.0;
        for(int i = 0; i < n; ++i) {
            r += v1[i] * v2[i];
        }
        return r;
    }
    double loss(double p, double y) {
        return -y * log(p) - (1 - y) * log(1 - p);
    }
    double distance(const double* v1, const double* v2, int n) {
        double sum = 0;
        for(int i = 0; i < n; ++i) {
            double minus = v1[i] - v2[i];
            double r = minus * minus;
            sum += r;
        }
        return sqrt(sum);
    }
    double sigmoid(double x) {
        double e = 2.718281828;
        if(x >= 10) {
            return 1.0 / (1.0 + pow(e, -10));
        } else if(x <= -10) {
            return 1.0 / (1.0 + pow(e, 10));
        }
        return 1.0 / (1.0 + pow(e, -x));
    }
    int binary(double* x) {
        return inner_prod(x, _weight, _dim) + _bias > 0;
    }
    double h(double* x) {
        return h(x, _weight, _dim, _bias);
    }
    double h(double* x, double* weight, int n, double bias) {
        double y =  inner_prod(x, weight, n) + bias;
        return sigmoid(y);
    }
    double learning_rate(double alpha, int round) {
        double lambda = 0.1;
        return alpha / (1 + lambda * alpha * round);
    }
    double update(double*x, int n, double y, double alpha, double l2, double l1, int round) {
        double predict = h(x);
        double lrate = learning_rate(alpha, round);
        for(int i = 0;  i < n; ++i) {
            double gradient = (predict - y) * x[i];
            _weight[i] = _weight[i] - lrate * gradient - l2 * _weight[i];
        }
        //update bias
        _bias = _bias - lrate * (predict - y) - l2 * _bias;
        return 0.0;
    }
    int random_pick(int m) {
        return rand() % m;
    }
    double perf(double**x, int m, int n, double* y) {
        double mse = 0;
        for(int i = 0; i < m; ++i) {
            mse += (y[i] - h(x[i])) * (y[i] - h(x[i]));
        }
        return sqrt(mse / m);
    }
    double loss(double**x, int m, int n, double* y) {
        double ls = 0;
        for(int i = 0; i < m; ++i) {
            ls += loss(h(x[i]), y[i]);
        }
        return ls;
    }
    double backtracking(double**x, int m, int n, double* y) {
        const double factor = 2.0;
        double loEta = 1;
        double loCost = fit(x, m, n, y, loEta);
        double hiEta = loEta * factor;
        double hiCost = fit(x, m, n, y, hiEta);
        if(loCost < hiCost){
            while(loCost < hiCost) {
                hiEta = loEta;
                hiCost = loCost;
                loEta = hiEta / factor;
                loCost =  fit(x, m, n, y, loEta);
            }
        }
        else if(hiCost < loCost){
            while(hiCost < loCost) {
                loEta = hiEta;
                loCost = hiCost;
                hiEta = loEta * factor;
                hiCost = fit(x, m, n, y, hiEta);
            }
        }
        cout << "# Using eta0=" << loEta << endl;
        return loEta;
    }
    //y: 0,1
    double fit(double**nx, int m, int n, double* y, double alpha = 0.01, double l2 = 0.0, double l1 = 0.0) {
        int max_iters = 10000;
        memset(_weight, 0, sizeof(_weight[0])*_dim);
        //double** x = scale(nx, m, n);
        double* predict = new double[m];
        unique_ptr<double[]> ptr(predict);
        for(int i = 0; i < max_iters; ++i) {
            int id = random_pick(m);
            update(nx[id], n, y[id], alpha, l2, l1, i);
        }
        return loss(nx, m, n, y);
    }
    void save(std::ostream& os) {
        os << "b:" << _bias << " ";
        for(int i = 0; i < _dim; ++i)
            os <<  i << ":" << _weight[i] << " " ;
        os << endl;
    }
    LR(int dim): _dim(dim) {
        _weight = new double[dim];
        _bias = 0.0;
    }
    ~LR() {
        delete[] _weight;
    }
private:
    double* _weight;
    int _dim;
    double _bias;
};
int main(int argc, char* argv[]) {
    if(argc < 5) {
        cerr << "Usage: " << argv[0] << " <train_feature> <train_target> <rows> <cols> [test]" << endl
             << "\t data_file: the training date\n";
        return -1;
    }
    srand(time(0));
    const char* feature = argv[1];
    const char* target = argv[2];
    int row = atoi(argv[3]);
    int col = atoi(argv[4]); //add bias
    double**x = dmatrix(row, col);
    double* y = dvector(row);
    //load_data(train_instance, x,y);  //if train_target\ttrain_feature are merged in one file
    csv_load_feature(feature, x);
    load_target(target, y);
    LR model(col);
    //sample x and y
    double alpha = model.backtracking(x, row, col, y);
    model.fit(x, row, col, y, alpha);
    model.save(std::cout);
    double** confuse = dmatrix(2, 2);
    for(int i = 0; i < row; ++i) {
        int pred = model.binary(x[i]);
        int label = (int)y[i];
        confuse[label][pred]++;
    }
    cout << "L\t0\t1\tprecision\tsupprt<-prdict\n";
    double label0 = confuse[0][0] + confuse[0][1];
    double label1 = confuse[1][0] + confuse[1][1];
    cout << "0\t" << confuse[0][0] << "\t" << confuse[0][1] << "\t" << confuse[0][0] / label0 << "\t" << label0 << endl;
    cout << "1\t" << confuse[1][0] << "\t" << confuse[1][1] << "\t" << confuse[1][1] / label1 << "\t" << label1 << endl;
    if(argc == 6) {
        ifstream test(argv[5]);
        string line;
        while(getline(test, line)) {
            csv_read(line.c_str(), x[0]);
            cerr << model.binary(x[0]) << endl;
        }
    }
    free_matrix(x, row);
    free_vector(y);
    return 0;
}
