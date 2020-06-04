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

#include "Naivebayes.h"
#include "SimpleNaiveBayesStatistics.h"
#include "../../../Common.h"

REGISTER_CLASS(NaiveBayes);
REGISTER_COMMAND_LINE_PARAMETER(
		NaiveBayes,
		"{\"type\":\"Learner\","
		"\"name\":\"NaiveBayes\","
		"\"parameter\":{"
		"}}"
		"");


//Naive Bayes Classifier
NaiveBayes::NaiveBayes():
	nbStatistics(nullptr),
    numberClasses(0),
    classPrediction(nullptr)
{
}

NaiveBayes::~NaiveBayes() {
    if (nbStatistics != nullptr) {
        delete nbStatistics;
    }

    if (classPrediction != nullptr) {
        delete[] classPrediction;
    }
}

void NaiveBayes::doSetParams() {
	//
}

void NaiveBayes::train(const Instance& instance) {
	if (!init) {
        init = true;
		nbStatistics = new SimpleNaiveBayesStatistics();
        if (classPrediction == nullptr) {
            numberClasses = instance.getNumberClasses();
            classPrediction = new double[numberClasses];
        }
	}

	int label = (int) instance.getLabel();
	double weight = instance.getWeight();

	nbStatistics->addObservation(label, weight);
	for (int j = 0; j < instance.getNumberInputAttributes(); j++) {
		double value = instance.getInputAttributeValue(j);
        bool isAttributeNumeric = instance.getInputAttribute(j)->isNumeric();
		nbStatistics->addObservation(label, j, isAttributeNumeric, value, weight);
	}
}

double* NaiveBayes::getPrediction(const Instance& instance) {
    if (!init) {
        if (classPrediction == nullptr) {
            numberClasses = instance.getNumberClasses();
            classPrediction = new double[numberClasses];

            for (int i = 0; i < numberClasses; i++) {
                classPrediction[i] = 0.0;
            }
        }

        return classPrediction;
    }

	for (int i = 0; i < numberClasses; i++) {
		classPrediction[i] = probability(instance, i);
	}

	return classPrediction;
}

double NaiveBayes::probability(const Instance& instance, int label) {
	if (!init) {
		return 0.0;
	}

	double prob = nbStatistics->probabilityOfClass(label);
	for (int j = 0; j < instance.getNumberInputAttributes(); j++) {
		double value = instance.getInputAttributeValue(j);
		prob *= nbStatistics->probabilityOfClassAttrValue(label, j, value);
	}

	return prob;
}

bool NaiveBayes::exportToJson(Json::Value& jv) {
    if (!init) {
        return false;
    }

	nbStatistics->exportToJson(jv);
    jv["instanceInformation"] = mInstanceInformation->toJson();

	return true;
}

bool NaiveBayes::importFromJson(const Json::Value& jv) {
    const int nClasses = jv["classCounts"].size();

    if (!nClasses) {
        return false;
    }

    numberClasses = nClasses;
	nbStatistics = new SimpleNaiveBayesStatistics();
	nbStatistics->importFromJson(jv);
    setAttributes(jv["instanceInformation"]);

    if (classPrediction == nullptr) {
        classPrediction = new double[numberClasses];
    }

    init = true;

	return true;
}
