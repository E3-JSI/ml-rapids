#include "Majorityclass.h"
#include "../../../Common.h"

REGISTER_CLASS(MajorityClass);
REGISTER_COMMAND_LINE_PARAMETER(MajorityClass, "{\"type\":\"Learner\","
		"\"name\":\"MajorityClass\","
		"\"parameter\":{"
		"}}"
		"");

MajorityClass::MajorityClass() {
	predArray = nullptr;
	classCounts = nullptr;
	predClassCount = 0;
}

MajorityClass::~MajorityClass() {
	if (classCounts != nullptr) {
		delete[] classCounts;
	}
	if (predArray != nullptr) {
		delete[] predArray;
	}
}

void MajorityClass::doSetParams() {
	//
}

void MajorityClass::train(const Instance& instance) {
	if (!init) {
		init = true;
        predClassCount = instance.getNumberClasses();
		classCounts = new int[predClassCount];
        predArray = new double[predClassCount];
		for (int i = 0; i < predClassCount; i++) {
			classCounts[i] = 0;
		}
	}
	
	int label = instance.getLabel();
	classCounts[label]++;
}

double* MajorityClass::getPrediction(const Instance& instance) {
    if (!init) {
        if (predArray == nullptr) {
            predClassCount = instance.getNumberClasses();
            predArray = new double[predClassCount];

            for (int i = 1; i < predClassCount; i++) {
                predArray[i] = 0.0;
            }
        }

        return predArray;
    }
    
	for (int i = 1; i < predClassCount; i++) {
		predArray[i] = 0.0;
	}
    
    int pred = 0;
	double max = classCounts[0];
	for (int i = 1; i < predClassCount; i++) {
		if (max < classCounts[i]) {
			max = classCounts[i];
			pred = i;
		}
	}

	predArray[pred] = 1.0;
	return predArray;
}

bool MajorityClass::exportToJson(Json::Value& jv) {
    if (!init) {
        return false;
    }

    jv["nClasses"] = predClassCount;

    for (size_t i = 0; i < predClassCount; i++) {
        jv["classCounts"].append(classCounts[i]);
    }

    jv["instanceInformation"] = mInstanceInformation->toJson();

    return true;
}

bool MajorityClass::importFromJson(const Json::Value& jv) {
    const int nClasses = jv["nClasses"].asInt();

    if (!nClasses || jv["classCounts"].size() != nClasses) {
        return false;
    }

    predClassCount = nClasses;

    if (classCounts != nullptr) {
        delete[] classCounts;
    }

    classCounts = new int[predClassCount];
    
    for (unsigned int i = 0; i < predClassCount; i++) {
        classCounts[i] = jv["classCounts"][i].asInt();
    }

    setAttributes(jv["instanceInformation"]);

    if (predArray == nullptr) {
        predArray = new double[predClassCount];
    }

    init = true;

    return true;
}
