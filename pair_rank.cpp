// stochastic gradient descent tricks, http://leno.bottou.org

#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include "data.h"
using namespace std;
class LRank {
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
        //_bias = _bias - lrate * (predict - y) - l2 * _bias;
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
        int max_iters = 40000;
        memset(_weight, 0, sizeof(_weight[0])*_dim);
        //double** x = scale(nx, m, n);
        double* predict = new double[m];
        double* feature = new double[n];
        vector<int> pos;
        vector<int> neg;
        for(int i = 0; i < m ; ++i){
            if(y[i] < 0.5) 
               neg.push_back(i);
            else
               pos.push_back(i);
        }
        unique_ptr<double[]> ptr1(predict);
        unique_ptr<double[]> ptr2(feature);
        for(int i = 0; i < max_iters; ++i) {
            int pi = random_pick(pos.size());
            int ni = random_pick(neg.size());
            for(int j = 0; j < n; j++){
                feature[j] = nx[pi][j] - nx[ni][j];
            }
            update(feature, n, 1, alpha, l2, l1, i);
            /*
            for(int j = 0; j < n; j++){
                feature[j] = -feature[j];
            }
            update(feature, n, 0, alpha, l2, l1, i);
            */
        }
        return loss(nx, m, n, y);
    }
    void save(std::ostream& os) {
        os << "b:" << _bias << " ";
        for(int i = 0; i < _dim; ++i)
            os <<  i << ":" << _weight[i] << " " ;
        os << endl;
    }
    LRank(int dim): _dim(dim) {
        _weight = new double[dim];
        _bias = 0.0;
    }
    ~LRank() {
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
    LRank model(col);
    //sample x and y
    //double alpha = model.backtracking(x, row, col, y);
    double alpha = 0.1;
    model.fit(x, row, col, y, alpha);
    model.save(std::cout);
    //cacl auc 
    free_matrix(x, row);
    free_vector(y);
    return 0;
}
