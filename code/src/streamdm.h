#ifndef STREAMDM_H
#define STREAMDM_H

#include "learners/Learner.h"
#include "learners/Classifiers/Trees/HoeffdingTree.h"
#include "learners/Classifiers/Trees/HoeffdingAdaptiveTree.h"
#include "learners/Classifiers/Bayes/Naivebayes.h"
#include "learners/Classifiers/Functions/Logisticregression.h"
#include "learners/Classifiers/Functions/Majorityclass.h"
#include "learners/Classifiers/Functions/Perceptron.h"
#include "learners/Classifiers/Meta/Bagging.h"

using namespace std;


template <class T>
class LearnerWrapper : public T {
public:
    LearnerWrapper();
    void fit(double* samples, int nSampels, int nFeatures, int* targets, int nTargets);
    void predict(double* samples, int nSampels, int nFeatures, int* predictions, int nPredictions);
    //void setAttributes(const vector<string>& featureDefs, const vector<string>& classDefs);
    //void setAttributes(const int nFeatures, const int nClasses);
    //void fit(const vector<double>& features, const int target);
    //void fit(const vector<vector<double>>& samples, const vector<int>& targets);
    //void fitBagging(const vector<double>& features, const int target);
    //void fitBagging(const vector<vector<double>>& samples, const vector<int> targets);
    //void process(const vector<double>& features, const int target);
    //void process(const vector<vector<double>>& samples, const vector<int> targets);
    //int predict(const vector<double>& features);
    //vector<int> predict(const vector<vector<double>>& samples);
    bool export_json(const string& file_name);
    //bool import_json(const string& file_name);
};


template <typename T>
vector<T> toVector(const T* arr, int size) {
    vector<T> vec(arr, arr + size);

    return vec;
}

template <typename T>
vector<vector<T>> toVector2d(const T* arr, int dim1, int dim2) {
    vector<vector<T>> vec2d;
    vec2d.reserve(dim1);

    for (int i = 0; i < dim1; i++) {
        vector<T> vec(arr + i * dim2, arr + (i + 1) * dim2);
        vec2d.push_back(vec);
    }

    return vec2d;
}


template <class T>
LearnerWrapper<T>::LearnerWrapper() : T() {}

template <>
LearnerWrapper<HT::HoeffdingTree>::LearnerWrapper() {
    const string htParams = "{"
        "\"MaxByteSize\":33554432,"
        "\"MemoryEstimatePeriod\":1000000,"
        "\"GracePeriod\":200,"
        "\"SplitConfidence\":0.0000001,"
        "\"TieThreshold\":0.05,"
        "\"BinarySplits\":false,"
        "\"StopMemManagement\":false,"
        "\"RemovePoorAtts\":false,"
        "\"LeafLearner\":\"NB\","
        "\"BbThreshold\":0,"
        "\"ShowTreePath\":false,"
        "\"TreePropertyIndexList\":\"\","
        "\"NoPrePrune\":false"
        "}";

    HT::HoeffdingTree::setParams(htParams);
}

template <>
LearnerWrapper<HT::HoeffdingAdaptiveTree>::LearnerWrapper() {
    const string htParams = "{"
        "\"MaxByteSize\":33554432,"
        "\"MemoryEstimatePeriod\":1000000,"
        "\"GracePeriod\":200,"
        "\"SplitConfidence\":0.0000001,"
        "\"TieThreshold\":0.05,"
        "\"BinarySplits\":false,"
        "\"StopMemManagement\":false,"
        "\"RemovePoorAtts\":false,"
        "\"LeafLearner\":\"NB\","
        "\"BbThreshold\":0,"
        "\"ShowTreePath\":false,"
        "\"TreePropertyIndexList\":\"\","
        "\"NoPrePrune\":false"
        "}";

    HT::HoeffdingAdaptiveTree::setParams(htParams);
}

template <class T>
void LearnerWrapper<T>::fit(double* samples, int nSampels, int nFeatures, int* targets, int nTargets) {
    vector<vector<double>> s = toVector2d(samples, nSampels, nFeatures);
    vector<int> t = toVector(targets, nTargets);
    T::fit(s, t);
}

template <class T>
void LearnerWrapper<T>::predict(double* samples, int nSampels, int nFeatures, int* predictions, int nPredictions) {
    vector<vector<double>> s = toVector2d(samples, nSampels, nFeatures);
    vector<int> p = T::predict(s);
    for (int i = 0; i < p.size(); i++) {
        predictions[i] = p[i];
    }
}

template <class T>
bool LearnerWrapper<T>::export_json(const string& file_name) {
    return T::exportToFile(file_name);
}

//template <class T>
//bool LearnerWrapper<T>::import_json(const string& file_name) {
//    return T::importFromFile(file_name);
//}


#endif //STREAMDM_H
