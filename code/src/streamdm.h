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
#include "utils/json.h"

using namespace std;

Json::Value mergeParams(const Json::Value params, const Json::Value defaultParams) {
    Json::Value::Members members = params.getMemberNames();
    Json::Value mergedParams(defaultParams);

    for (size_t i = 0; i < members.size(); i++) {
        const string key = members[i];
        const Json::Value value = params[key];

        if (defaultParams.isMember(key)) {
            mergedParams[key] = value;
        }
        else {
            // TODO: LOG_ERROR
            cout << "Unknown argument: " + key << endl;
        }
    }

    return mergedParams;
}

string mergeParams(const string& params, const string& defaultParams) {
    stringstream ssDefaultParams(defaultParams != "" ? defaultParams : "{}");
    Json::Value jsonDefaultParams;
    ssDefaultParams >> jsonDefaultParams;

    stringstream ssParams(params != "" ? params : "{}");
    Json::Value jsonParams;
    ssParams >> jsonParams;

    Json::Value mergedParams = mergeParams(jsonParams, jsonDefaultParams);
    return mergedParams.toStyledString();
}


template <class T>
class LearnerWrapper : public T {
public:
    //LearnerWrapper();
    LearnerWrapper(const string& params = "");
    void set_params(const string& params);
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
    bool export_json(const string& file_name, const string& json = "");
    string import_json(const string& file_name);
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
LearnerWrapper<T>::LearnerWrapper(const string& params) : T() {}

template <>
void LearnerWrapper<HT::HoeffdingTree>::set_params(const string& params) {
    const string defaultParams = "{"
        "\"MaxByteSize\":33554432,"
        "\"MemoryEstimatePeriod\":1000000,"
        "\"GracePeriod\":200,"
        "\"SplitConfidence\":0.0000001,"
        "\"TieThreshold\":0.05,"
        "\"BinarySplits\":false,"
        "\"StopMemManagement\":false,"
        "\"RemovePoorAtts\":false,"
        "\"LeafLearner\":\"NB\","
        "\"NbThreshold\":0,"
        //"\"ShowTreePath\":false,"
        "\"TreePropertyIndexList\":\"\","
        "\"NoPrePrune\":false"
        "}";

    HT::HoeffdingTree::setParams(mergeParams(params, defaultParams));
}

template <>
LearnerWrapper<HT::HoeffdingTree>::LearnerWrapper(const string& params) {
    LearnerWrapper<HT::HoeffdingTree>::set_params(params);
}

template <>
void LearnerWrapper<HT::HoeffdingAdaptiveTree>::set_params(const string& params) {
    const string defaultParams = "{"
        "\"MaxByteSize\":33554432,"
        "\"MemoryEstimatePeriod\":1000000,"
        "\"GracePeriod\":200,"
        "\"SplitConfidence\":0.0000001,"
        "\"TieThreshold\":0.05,"
        "\"BinarySplits\":false,"
        "\"StopMemManagement\":false,"
        "\"RemovePoorAtts\":false,"
        "\"LeafLearner\":\"NB\","
        "\"NbThreshold\":0,"
        //"\"ShowTreePath\":false,"
        "\"TreePropertyIndexList\":\"\","
        "\"NoPrePrune\":false"
        "}";

    HT::HoeffdingAdaptiveTree::setParams(mergeParams(params, defaultParams));
}

template <>
LearnerWrapper<HT::HoeffdingAdaptiveTree>::LearnerWrapper(const string& params) {
    LearnerWrapper<HT::HoeffdingAdaptiveTree>::set_params(params);
}

template <>
void LearnerWrapper<NaiveBayes>::set_params(const string& params) {
    const string defaultParams = "{"
        "}";

    NaiveBayes::setParams(mergeParams(params, defaultParams));
}

template <>
LearnerWrapper<NaiveBayes>::LearnerWrapper(const string& params) {
    LearnerWrapper<NaiveBayes>::set_params(params);
}

template <>
void LearnerWrapper<LogisticRegression>::set_params(const string& params) {
    const string defaultParams = "{"
        "\"LearningRatio\":0.01,"
        "\"Lambda\":0.0001"
        "}";

    LogisticRegression::setParams(mergeParams(params, defaultParams));
}

template <>
LearnerWrapper<LogisticRegression>::LearnerWrapper(const string& params) {
    LearnerWrapper<LogisticRegression>::set_params(params);
}

template <>
void LearnerWrapper<Perceptron>::set_params(const string& params) {
    const string defaultParams = "{"
        "\"LearningRatio\":1.0"
        "}";

    Perceptron::setParams(mergeParams(params, defaultParams));
}

template <>
LearnerWrapper<Perceptron>::LearnerWrapper(const string& params) {
    LearnerWrapper<Perceptron>::set_params(params);
}

template <>
void LearnerWrapper<MajorityClass>::set_params(const string& params) {
    const string defaultParams = "{"
        "}";

    MajorityClass::setParams(mergeParams(params, defaultParams));
}

template <>
LearnerWrapper<MajorityClass>::LearnerWrapper(const string& params) {
    LearnerWrapper<MajorityClass>::set_params(params);
}

template <>
void LearnerWrapper<Bagging>::set_params(const string& params) {
    const string defaultParams = "{"
        "\"EnsembleSize\":10,"
        "\"Learner\":{"
        "\"Name\":\"HoeffdingTree\","
        "\"MaxByteSize\":33554432,"
        "\"MemoryEstimatePeriod\":1000000,"
        "\"GracePeriod\":200,"
        "\"SplitConfidence\":0.0000001,"
        "\"TieThreshold\":0.05,"
        "\"BinarySplits\":false,"
        "\"StopMemManagement\":false,"
        "\"RemovePoorAtts\":false,"
        "\"LeafLearner\":\"NB\","
        "\"NbThreshold\":0,"
        "\"TreePropertyIndexList\":\"\","
        "\"NoPrePrune\":false"
        "}"
        "}";

    Bagging::setParams(mergeParams(params, defaultParams));
}

template <>
LearnerWrapper<Bagging>::LearnerWrapper(const string& params) {
    LearnerWrapper<Bagging>::set_params(params);
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
bool LearnerWrapper<T>::export_json(const string& file_name, const string& json) {
    return T::exportToFile(file_name, json);
}

template <class T>
string LearnerWrapper<T>::import_json(const string& file_name) {
    string json;
    T::importFromFile(file_name, json);

    return json;
}


#endif //STREAMDM_H
