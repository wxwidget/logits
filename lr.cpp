//http://people.csail.mit.edu/jrennie/writing/lr.pdf  y=+1,-1
#include <iostream>
#include <cmath>
#include <memory>
#include <fstream>
#include "data.h"
using namespace std;
class LR {
public:
    double** scale(double**x,int m, int n){
        double** scale_x = dmatrix(m,n);
        for(int i = 0; i < n ;++i){//feature
            double mean = 0.0;
            double var = 0.0;
            for(int j = 0; j < m; j++){
                mean += x[j][i];
                var += x[j][i] * x[j][i];
            }
            mean = mean/m;
            var = var/m - mean * mean;
            for(int j = 0; j < m; j++){
                scale_x[j][i] = (x[j][i] - mean)/var;
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
        if (x >= 10){
            return 1.0 / (1.0 + pow(e, -10));
        }else if (x <= -10){
            return 1.0 / (1.0 + pow(e, 10));
        }
        return 1.0 / (1.0 + pow(e, -x));
    }
    int binary(double* x){
        return inner_prod(x, _weight_new, _dim) + _bias > 0;
    }
    double h(double* x) {
        return h(x, _weight_new, _dim, _bias);
    }
    double h(double* x, double* weight, int n, double bias) {
        double y =  inner_prod(x, weight, n) + bias;
        return sigmoid(y);
    }
    //y: 0,1
    double fit(double**nx, int m, int n, double* y, double alpha = 0.01, double l2 = 0.0, double l1=0.0) {
        int max_iters = 4000;
        memset(_weight_old, 0, sizeof(_weight_old[0])*_dim);
        memset(_weight_new, 0, sizeof(_weight_new[0])*_dim);
        //double** x = scale(nx, m, n);
        double**x = nx;
        double* predict = new double[m];
        unique_ptr<double[]> ptr(predict);
        double last_mrse = 1e10;
        for(int iter = 0; iter < max_iters; ++iter) {
            //predict
            double mrse = 0;
            for(int i = 0; i < m; ++i) {
                predict[i] = h(x[i], _weight_old, _dim, _bias_old);
                mrse += (y[i] - predict[i]) * (y[i] - predict[i]);
            }
            //cout << "mrse:" << sqrt(mrse/m) << endl;
            if (last_mrse - mrse < 0.0001){
                return mrse;
            }
            last_mrse = mrse;
            std::swap(_weight_old, _weight_new);
            _bias = _bias_old;
            //update each weight
            for(int k = 0; k < _dim; ++k) {
                double gradient = 0.0;
                for(int i = 0; i < m; ++i) {
                    gradient += (predict[i] - y[i]) * x[i][k];
                }
                _weight_new[k] = _weight_old[k] - alpha * gradient/m - l2 * _weight_old[k];
                //if (_weight_new[k] < 11){ _weight_new[k] = 0; }
            }
            //update bias
            double g = 0.0;
            for(int i = 0; i < m; ++i) {
                g += (predict[i] - y[i]);
            }
            _bias_old = _bias - alpha * g/m - l2 * _bias;
        }
        return distance(_weight_new, _weight_old, _dim);
    }
    void save(std::ostream& os) {
        os << "b:" << _bias << " ";
        for(int i = 0; i < _dim; ++i)
            os <<  i << ":" << _weight_new[i] << " " ;
        os << endl;
    }
    LR(int dim): _dim(dim) {
        _weight_new = new double[dim];
        _weight_old = new double[dim];
        _bias = 0.0;
        _bias_old = 0.0;
    }
    ~LR() {
        delete[] _weight_old;
        delete[] _weight_new;
    }
private:
    double* _weight_old;
    double* _weight_new;
    int _dim;
    double _bias;
    double _bias_old;
};
int main(int argc, char* argv[]) {
    if(argc < 5) {
        cerr << "Usage: " << argv[0] << " <train_feature> <train_target> <rows> <cols> [test]" << endl
             << "\t data_file: the training date\n";
        return -1;
    }
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
    model.fit(x, row, col, y, 0.1);
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
    cout << "0\t" << confuse[0][0] << "\t" << confuse[0][1] << "\t" << confuse[0][0]/label0 << "\t" << label0 << endl;
    cout << "1\t" << confuse[1][0] << "\t" << confuse[1][1] << "\t" << confuse[1][1]/label1 << "\t" << label1 << endl;
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
