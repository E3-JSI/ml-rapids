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

#ifndef LEARNER_H
#define LEARNER_H

#include <unordered_set>
#include "../core/Instance.h"
#include "../evaluation/Evaluator.h"
#include "../utils/Configurable.h"
#include "../utils/LearnerModel.h"
#include "../API.h"

using namespace std;

class STREAMDM_API Learner : public Configurable, public LearnerModel {
public:
	Learner();
	virtual ~Learner();
	virtual void train(const Instance&) = 0;
	virtual double* getPrediction(const Instance&) = 0;
	int predict(const Instance&);
	void trainBagging(const Instance&);
	void process(const Instance&);
	int getInstanceSeen() const;
	void setEvaluator(Evaluator*);
	virtual void initPara(int argc, char* argv[]) {};
	Evaluator* getEvaluator() const;
	virtual string getEnsemblePrediction(const Instance&);

    void setAttributes(const vector<string>& featureDefs, const vector<string>& classDefs);
    void setAttributes(const int nFeatures, const int nClasses);
    void fit(const vector<double>& features, const int target);
    void fit(const vector<vector<double>>& samples, const vector<int>& targets);
    void fitBagging(const vector<double>& features, const int target);
    void fitBagging(const vector<vector<double>>& samples, const vector<int> targets);
    void process(const vector<double>& features, const int target);
    void process(const vector<vector<double>>& samples, const vector<int> targets);
    int predict(const vector<double>& features);
    vector<int> predict(const vector<vector<double>>& samples);

private:
    vector<string> splitCSV(const string& csvString) const;
    Instance* generateInstance(const vector<double>& features, const int target) const;
    void setAttributes(const vector<double>& features, const int target);
    void setAttributes(const vector<vector<double>>& samples, const vector<int>& targets);

protected:
	int instancesSeen;
	bool init;
	Evaluator* evaluator;
    InstanceInformation* mInstanceInformation = nullptr;
    unordered_set<string> mClasses;
};

#endif //LEARNER_H