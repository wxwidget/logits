#include "common/parse.h"
#include "data.h"
#include <fstream>
using namespace std;
using namespace logitlab::legacy;
double** dmatrix(int row, int col) {
    double** x = new double*[row];
    for(int i = 0; i < row; ++i) {
        x[i] = new double[col];
        memset(x[i], 0, sizeof(double)*col);
    }
    return x;
}
void free_matrix(double**x, int row) {
    for(int i = 0; i < row; ++i) {
        delete[] x[i];
    }
    delete[] x;
}
double* dvector(int col) {
    double* x = new double[col];
    memset(x, 0, sizeof(double)*col);
    return x;
}
void free_vector(double* x){
    delete []x;
}
bool ends_with(const string& str, const string& suffix) {
    return str.size()>=suffix.size() && str.compare(str.size()-suffix.size(),suffix.size(),suffix)==0;
}
void csv_read(const char* input, double* x) {
    std::vector<std::string> fields=absl::StrSplit(input, ',');
    for(std::size_t i=0;i<fields.size();++i)x[i]=logitlab::legacy::lexical_cast<double>(fields[i]);
}
void svm_read_line(const string& line, double* x, double& y) {
    //libsvm format
    //<class/target> [<attribute number>:<attribute value>]*
    //csv format
    //target[,features_values]
    vector<string> strs;
    split(strs, line, is_any_of("\t ,"));
    double target = 0;
    if(strs[0] == "+1") target = 1;
    else if(strs[0] == "1") target = 1;
    for(unsigned int i = 1; i < strs.size(); ++i) {
        vector<string> kvs;
        split(kvs, strs[i], is_any_of(":"));
        if(kvs.size() == 2) { //id:value for libsvm
            int index = lexical_cast<int>(kvs[0]);
            x[index] = lexical_cast<double>(kvs[1]);
        } else if(kvs.size() == 1) { //value for csv
            x[i] =  lexical_cast<double>(kvs[0]);
        }
    }
    y = target;
}
void svm_load(const string& filename, double** x, double* y) {
    int cur_row = 0;
    ifstream ifs(filename.c_str());
    string line;
    while(getline(ifs, line)) {
        svm_read_line(line, x[cur_row], y[cur_row]);
        cur_row++;
    }
}
void csv_load_feature(const char* train_feature, double**x) {
    int cur_row = 0;
    ifstream ifs(train_feature);
    string line;
    while(getline(ifs, line)) {
        csv_read(line.c_str(), x[cur_row]);
        cur_row++;
    }
}
void load_target(const char* train_target, double* y) {
    int cur_row = 0;
    ifstream ifs(train_target);//target 0 or 1
    double temp = 0;
    while(ifs >> temp) {
        y[cur_row++] = temp;
    }
}
