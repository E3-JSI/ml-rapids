%module streamdm

%{
    #define SWIG_FILE_WITH_INIT
    #include "streamdm.h"
%}

%include "std_string.i"
%include "numpy.i"

%inline %{
    using namespace std;
%}

%init %{
    import_array();
%}

// Apply typemaps
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* samples, int nSampels, int nFeatures)};
%apply (int* IN_ARRAY1, int DIM1) {(int* targets, int nTargets)};
%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* predictions, int nPredictions)};

%pythoncode %{
import numpy as np
%}

// Rewrite methods
%feature("shadow") fit(double*, int, int, int*, int) %{
def fit(self, samples, targets):
    indexed_targets = []
    for target in targets:
        if (target not in self.label_map):
            map_len = len(self.label_map)
            self.label_map[target] = map_len
            self.label_map_inv[map_len] = target
        indexed_targets.append(self.label_map[target])
    return $action(self, samples, indexed_targets)
%}

%feature("shadow") predict(double*, int, int, int*, int) %{
def predict(self, samples):
    predictions_len = len(samples)
    predictions = $action(self, samples, predictions_len)
    return np.array([self.label_map_inv[p] for p in predictions])
%}

%feature("pythonprepend") LearnerWrapper() %{
    self.label_map = {}
    self.label_map_inv = {}
%}

%include "streamdm.h"

// Instantiate templates
%template(HoeffdingTree) LearnerWrapper<HT::HoeffdingTree>;
%template(HoeffdingAdaptiveTree) LearnerWrapper<HT::HoeffdingAdaptiveTree>;
%template(NaiveBayes) LearnerWrapper<NaiveBayes>;
%template(LogisticRegression) LearnerWrapper<LogisticRegression>;
%template(MajorityClass) LearnerWrapper<MajorityClass>;
%template(Perceptron) LearnerWrapper<Perceptron>;
%template(Bagging) LearnerWrapper<Bagging>;
