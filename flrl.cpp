#include "common/parse.h"
/*
 * ref paper
 *    http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/pubs/archive/41159.pdf
 *    http://jmlr.org/proceedings/papers/v15/mcmahan11b/mcmahan11b.pdf
 *
 * how to Parallelized
 *    Parallelized Stochastic Gradient Descent  http://martin.zinkevich.org/publications/nips2010.pdf
 */
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <list>
#include "data.h"
using namespace std;

class LR {
public:
    LR(double a, double b = 1): alpha(a), beta(b) {
        largerConstrain.insert("month_ipv");
    }
    double sigmoid(double x) {
        double e = 2.718281828;
        return 1.0 / (1.0 + pow(e, -x));
    }
    double classify(const absl::flat_hash_map<string, double>& x) {
        absl::flat_hash_map<string, double>::const_iterator it = x.begin();
        double y =  0;
        for(; it != x.end(); ++it) {
            y += weight[it->first] * it->second;
        }
        return sigmoid(y);
    }
    int binary(const absl::flat_hash_map<string, double>& x) {
        absl::flat_hash_map<string, double>::const_iterator it = x.begin();
        double y =  0;
        for(; it != x.end(); ++it) {
            y += weight[it->first] * it->second;
        }
        return y > 0;
    }
    static int sgn(double x) {
        if(x > 0) return(1);
        if(x < 0) return(-1);
        return(0);
    }
    // y: +1,0
    double update(const absl::flat_hash_map<string, double>& x, double target, double L1, double L2,  double iter) {
        static int round = 1;
        absl::flat_hash_map<string, double>::const_iterator it = x.begin();
        for(; it != x.end(); ++it) {
            const string& name = it->first;
            if(fabs(zi[name]) <= L1) {
                weight[name] = 0;
            } else {
                weight[name] = -(zi[name] - sgn(zi[name]) * L1) / ((beta + sqrt(ni[name])) / alpha + L2);
            }
        }
        double pt = classify(x);
        it = x.begin();
        for(; it != x.end(); ++it) {
            const string& name = it->first;
            double value = it->second;
            double gi = (pt - target) * value;
            double thi = (sqrt(ni[name] + gi * gi) - sqrt(ni[name])) / alpha;
            zi[name] += gi - thi * weight[name];
            ni[name] += gi * gi;
        }
        if(round++ % 10000 == 0) {
            save(std::cerr);
        }
        return 0;
    }
    void save(ostream& osr) {
        absl::flat_hash_map<string, double>::iterator it = weight.begin();
        for(; it != weight.end(); ++it) {
            if(it->second != 0)
                osr << it->first << ":" << it->second << " ";
        }
        osr << endl;
    }
private:
    absl::flat_hash_set<string> largerConstrain;
    absl::flat_hash_map<string, double> weight;
    absl::flat_hash_map<string, double> zi;
    absl::flat_hash_map<string, double> ni;
    double alpha;
    double beta;
};
int main(int argc, char* argv[]) {
    if(argc != 2) {
        cout << "Usage: " << argv[0] << " data_file" << endl;
        return -1;
    }
    LR model(0.5);
    typedef absl::flat_hash_map<string, double> Instance;
    vector<Instance> dataset;
    vector<double> labels; 
    string line;
    bool csv = ends_with(argv[1], ".csv");
    bool svm = ends_with(argv[1], ".svm");
    int iterator = 1;
    while(iterator < 10) {
        ifstream ifs(argv[1]);
        float pos = 0;
        float neg = 0;
        while(getline(ifs, line)) {
            //libsvm format
            //<class/target> [<attribute number>:<attribute value>]*
            Instance features;
            vector<string> strs;
            logitlab::legacy::split(strs, line, logitlab::legacy::is_any_of("\t ,"));
            double target = 1;//logitlab::legacy::lexical_cast<double>(strs[0]);
            if(strs[0] == "-1") target = 0;
            else if(strs[0] == "0") target = 0;
            pos += target;
            neg += 1 - target;
            //double target = logitlab::legacy::lexical_cast<double>(strs[0]);
            for(unsigned int i = 1; i < strs.size(); ++i) {
                vector<string> kvs;
                logitlab::legacy::split(kvs, strs[i], logitlab::legacy::is_any_of(":"));
                if(kvs.size() == 2) {
                    features[kvs[0]] = logitlab::legacy::lexical_cast<double>(kvs[1]);
                } else if(kvs.size() == 1) {
                    if (csv){
                        string name = logitlab::legacy::feature_name(i-1);
                        features[name] = logitlab::legacy::lexical_cast<double>(kvs[0]);
                    }
                    else
                        features[kvs[0]] = 1;
                }
            }
            features["b"] = 1;
            model.update(features, target, 0.5, 0.5, iterator++);
            dataset.push_back(features);
            labels.push_back(target);
        }
        model.save(std::cout);
    }
    double** confuse = dmatrix(2, 2);
    for(int i = 0; i < (int)dataset.size();++i) {
        int pred = model.binary(dataset[i]);
        int label = (int)labels[i];
        confuse[label][pred]++;
    }
    cerr << "L\t0\t1\tprecision\tsupprt<-prdict\n";
    double label0 = confuse[0][0] + confuse[0][1];
    double label1 = confuse[1][0] + confuse[1][1];
    cerr << "0\t" << confuse[0][0] << "\t" << confuse[0][1] << "\t" << confuse[0][0]/label0 << "\t" << label0 << endl;
    cerr << "1\t" << confuse[1][0] << "\t" << confuse[1][1] << "\t" << confuse[1][1]/label1 << "\t" << label1 << endl;
    return 0;
}
