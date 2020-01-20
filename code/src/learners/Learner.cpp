/*
 * Copyright (C) 2015 Holmes Team at HUAWEI Noah's Ark Lab.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "Learner.h"
#include "../Common.h"
#include <cmath>
#include <time.h>

Learner::Learner() {
	instancesSeen = 0;
	init = false;
	this->evaluator = nullptr;
}

Learner::~Learner() {
    delete mInstanceInformation;
}

vector<string> Learner::splitCSV(const string& csvString) const {
    const string delim = ",";
    size_t start = 0;
    size_t end = csvString.find(delim);
    vector<string> labels;

    while ((end = csvString.find(delim, start)) != string::npos) {
        labels.push_back(csvString.substr(start, end - start));
        start = end + delim.size();
    }
    labels.push_back(csvString.substr(start, end));

    return labels;
}

void Learner::setAttributes(const int nFeatures, const int nClasses) {
    if (mInstanceInformation != nullptr) {
        return;
    }

    mInstanceInformation = new InstanceInformation();

    for (int i = 0; i < nFeatures; i++) {
        mInstanceInformation->addAttribute(new Attribute(), i);
    }

    vector<string> classes;
    for (int i = 0; i < nClasses; i++) {
        classes.push_back(to_string(i));
    }
    mInstanceInformation->addClass(new Attribute(classes), 0);
}

void Learner::setAttributes(const vector<string>& featureDefs, const vector<string>& classDefs) {
    if (mInstanceInformation != nullptr) {
        return;
    }

    mInstanceInformation = new InstanceInformation();

    for (int i = 0; i < featureDefs.size(); i++) {
        vector<string> labels = splitCSV(featureDefs[i]);

        if (labels.size() < 2) {
            mInstanceInformation->addAttribute(new Attribute(), i);
        }
        else {
            mInstanceInformation->addAttribute(new Attribute(labels), i);
        }
    }

    for (int i = 0; i < classDefs.size(); i++) {
        vector<string> labels = splitCSV(classDefs[i]);

        if (labels.size() < 2) {
            mInstanceInformation->addClass(new Attribute(), i);
        }
        else {
            mInstanceInformation->addClass(new Attribute(labels), i);
        }
    }
}

void Learner::setAttributes(const vector<double>& features, const int target) {
    if (mInstanceInformation == nullptr) {
        mInstanceInformation = new InstanceInformation();

        for (int i = 0; i < features.size(); i++) {
            mInstanceInformation->addAttribute(new Attribute(), i);
        }

        string labelStr = to_string(target);
        mClasses.insert(labelStr);

        vector<string> labels;
        labels.push_back(labelStr);
        mInstanceInformation->addClass(new Attribute(labels), 0);
    }
    else if (mClasses.size() > 0) {
        string labelStr = to_string(target);

        if (mClasses.find(labelStr) == mClasses.end()) {
            mClasses.insert(labelStr);

            // Remove old output attribute (classDefs).
            Attribute *attr = mInstanceInformation->getOutputAttribute(0);
            delete attr;

            // Add new output featureDefs (classDefs).
            vector<string> labels;
            labels.insert(labels.end(), mClasses.begin(), mClasses.end());
            mInstanceInformation->addClass(new Attribute(labels), 0);
        }
    }
}

void Learner::setAttributes(const vector<vector<double>>& samples, const vector<int>& targets) {
    if (mInstanceInformation != nullptr) {
        return;
    }

    // Get number of classDefs
    int nClasses = 0;
    unordered_set<int> set;

    for (int i = 0; i < targets.size(); i++) {
        if (set.find(targets[i]) == set.end()) {
            set.insert(targets[i]);
            nClasses++;
        }
    }

    mInstanceInformation = new InstanceInformation();

    for (int i = 0; i < samples[0].size(); i++) {
        mInstanceInformation->addAttribute(new Attribute(), i);
    }

    vector<string> classes;
    unordered_set<int>::iterator itr;
    for (itr = set.begin(); itr != set.end(); itr++) {
        classes.push_back(to_string(*itr));
    }

    mInstanceInformation->addClass(new Attribute(classes), 0);
}

Instance* Learner::generateInstance(const vector<double>& features, const int target) const {
    DenseInstance* instance = new DenseInstance();
    instance->setInstanceInformation(mInstanceInformation);
    instance->addValues(features);

    vector<double> labels(1);
    labels[0] = mInstanceInformation->getOutputAttributeIndex(0, to_string(target));
    instance->addLabels(labels);

    return instance;
}

void Learner::fit(const vector<double>& features, const int target) {
    setAttributes(features, target);

    Instance* instance = generateInstance(features, target);

    // Train
    train(*instance);

    delete instance;
}

void Learner::fit(const vector<vector<double>>& samples, const vector<int>& targets) {
    setAttributes(samples, targets);

    // Train
    for (int i = 0; i < samples.size(); i++) {
        fit(samples[i], targets[i]);
    }
}

void Learner::process(const Instance &inst) {
	// Test
	if (instancesSeen > 0) {
		evaluator->addResult(inst, this->getPrediction(inst));
	}
	instancesSeen++;
	// Train
	train(inst);
}

void Learner::process(const vector<double>& features, const int target) {
    setAttributes(features, target);

    Instance* instance = generateInstance(features, target);

    process(*instance);

    delete instance;
}

void Learner::process(const vector<vector<double>>& samples, const vector<int> targets) {
    setAttributes(samples, targets);

    // Process.
    for (int i = 0; i < samples.size(); i++) {
        process(samples[i], targets[i]);
    }
}

void Learner::trainBagging(const Instance &inst) {
	int weight = Utils::poisson(1.0);
	while (weight > 0) {
		weight--;
		train(inst);
	}
}

void Learner::fitBagging(const vector<double>& features, const int target) {
    setAttributes(features, target);

    Instance* instance = generateInstance(features, target);

    trainBagging(*instance);

    delete instance;
}

void Learner::fitBagging(const vector<vector<double>>& samples, const vector<int> targets) {
    setAttributes(samples, targets);

    // Fit bagging.
    for (int i = 0; i < samples.size(); i++) {
        fitBagging(samples[i], targets[i]);
    }
}

int Learner::getInstanceSeen() const {
	return instancesSeen;
}

int Learner::predict(const Instance& instance) {
	double numberClasses = instance.getNumberClasses();
	double* classPredictions = getPrediction(instance);
	int result = 0;
	double max = classPredictions[0];
	//Find class target with higher probability
	for (int i = 1; i < numberClasses; i++) {
		if (max < classPredictions[i]) {
			max = classPredictions[i];
			result = i;
		}
	}
	return result;
}

int Learner::predict(const vector<double>& features) {
    DenseInstance instance;
    instance.setInstanceInformation(mInstanceInformation);
    instance.addValues(features);

    int index = predict(instance);

    return stoi(instance.getOutputAttribute(0)->getValue(index));
}

vector<int> Learner::predict(const vector<vector<double>>& samples) {
    vector<int> predictions;
    for (int i = 0; i < samples.size(); i++) {
        predictions.push_back(predict(samples[i]));
    }

    return predictions;
}

void Learner::setEvaluator(Evaluator* ev) {
	this->evaluator = ev;
}

Evaluator* Learner::getEvaluator() const {
	return evaluator;
}

string Learner::getEnsemblePrediction(const Instance&) {
	return "";
}
