#ifndef MAJORITYCLASS_H
#define MAJORITYCLASS_H

#include "../../Learner.h"
#include "../../../API.h"

class STREAMDM_API MajorityClass: public Learner {
public:
	MajorityClass();
	~MajorityClass();
	void train(const Instance&);
	double* getPrediction(const Instance&);
    virtual bool exportToJson(Json::Value& jv);
    virtual bool importFromJson(const Json::Value& jv);

private:
	int* classCounts;
	double* predArray;
	int predClassCount;
	void doSetParams();
};

#endif //MAJORITYCLASS_H
