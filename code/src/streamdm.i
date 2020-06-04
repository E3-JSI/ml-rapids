%module streamdm
%rename (fn) file_name;

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
import json
import numpy as np

def convertType(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def capitalizeKeys(kwargs):
    args = {}
    for k, v in kwargs.items():
        key = ''.join([s.capitalize() for s in k.split('_')])
        args[key] = capitalizeKeys(v) if isinstance(v, dict) else v
    return args
%}

// Rewrite methods
%extend LearnerWrapper {
    %pythoncode %{
        SWIG__init__ = __init__
        def __init__(self, *args, **kwargs):
            self.label_map = {}
            self.label_map_inv = {}
            # Pass all keyword arguments as JSON encoded positional argument.
            # Argument validation should be implemented on the C++ side.
            args = (json.dumps(capitalizeKeys(kwargs)),)
            self.SWIG__init__(*args)
    %}
};

%feature("shadow") set_params %{
def set_params(self, **kwargs):
    params = json.dumps(capitalizeKeys(kwargs))
    return $action(self, params)
%}

%feature("shadow") fit(double*, int, int, int*, int) %{
def fit(self, samples, targets):
    indexed_targets = []
    for target in targets:
        if (target not in self.label_map):
            map_len = len(self.label_map)
            self.label_map[convertType(target)] = map_len
            self.label_map_inv[map_len] = convertType(target)
        indexed_targets.append(self.label_map[target])
    return $action(self, samples, indexed_targets)
%}

%feature("shadow") predict(double*, int, int, int*, int) %{
def predict(self, samples):
    predictions_len = len(samples)
    predictions = $action(self, samples, predictions_len)
    return np.array([self.label_map_inv[p] for p in predictions])
%}

%feature("shadow") export_json %{
    def export_json(self, file_name):
        props = json.dumps({
            'label_map': [i for i in self.label_map.items()],
            'label_map_inv': [i for i in self.label_map_inv.items()],
        }) 
        return $action(self, file_name, props)
%}

%feature("shadow") import_json %{
    def import_json(self, file_name):
        props_json = $action(self, file_name)
        if (props_json):
            props = json.loads(props_json)
            self.label_map = {i[0]: i[1] for i in props['label_map']}
            self.label_map_inv = {i[0]: i[1] for i in props['label_map_inv']}
        return props_json
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
