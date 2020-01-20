#include <chrono>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include "Common.h"
#include "streams/ArffReader.h"
#include "streams/CSVReader.h"
#include "learners/Classifiers/Trees/HoeffdingTree.h"
#include "evaluation/BasicClassificationEvaluator.h"
#include "tasks/Task.h"
#include "utils/json.h"

using namespace std;
using namespace HT;

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

struct Dataset {
    vector<vector<double>> X;
    vector<int> y;
};

struct TTDataset {
    Dataset train;
    Dataset test;
};

unsigned int stopwatch() {
    static auto startTime = chrono::steady_clock::now();

    auto endTime = chrono::steady_clock::now();
    auto delta = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

    startTime = endTime;

    return (unsigned int)delta.count();
}

void printTime(const unsigned int time) {
    cout << (double)time / 1000000 << " seconds" << endl;
}

Dataset readData(const string& fileName, const int maxSamples = 0) {
    // Initialize reader.
    ArffReader reader;
    reader.setFile(fileName);

    Dataset ds;
    Instance* inst;
    int n = 0;
    while (reader.hasNextInstance()) {
        n++;
        if (maxSamples > 0 && n > maxSamples) {
            break;
        }

    	inst = reader.nextInstance();

    	vector<double> x;
    	for (int i = 0; i < inst->getNumberInputAttributes(); i++) {
    		x.push_back(inst->getInputAttributeValue(i));
    	}

    	ds.X.push_back(x);
    	//ds.y.push_back((int)inst->getOutputAttributeValue(0));
    	ds.y.push_back(stoi(inst->getOutputAttribute(0)->getValue(inst->getOutputAttributeValue(0))));

        delete inst;
    }

    return ds;
}

TTDataset splitData(const Dataset& dataset, const double testSetFraction = 0.25) {
    TTDataset ttDataset;
    const size_t dsSize = dataset.X.size();
    const int trainSetSize = (int)round(max(0.0, min(1.0 - testSetFraction, 1.0)) * dsSize);

    for (int i = 0; i < dsSize; i++) {
        if (i < trainSetSize) {
            ttDataset.train.X.push_back(dataset.X[i]);
            ttDataset.train.y.push_back(dataset.y[i]);
        }
        else {
            ttDataset.test.X.push_back(dataset.X[i]);
            ttDataset.test.y.push_back(dataset.y[i]);
        }
    }

    return ttDataset;
}

void printData(const Dataset& dataset, const bool verbose = false) {
    // Print feature matrix.
    const size_t xRows = dataset.X.size();
    const size_t xCols = xRows > 0 ? dataset.X[0].size() : 0;

    cout << "X: shape(" << xRows << ", " << xCols << ")" << endl;

    if (verbose) {
        for (int i = 0; i < xRows; i++) {
            for (int j = 0; j < xCols; j++) {
                cout << dataset.X[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    // Print target vector.
    const size_t ySize = dataset.y.size();

    cout << "y: shape(" << ySize << ")" << endl;

    if (verbose) {
        for (int i = 0; i < ySize; i++) {
            cout << dataset.y[i] << " ";
        }
        cout << endl;
    }
}

void testTask() {
	const string taskName = "EvaluatePrequential";
	const string taskParams = "{"
        "\"DataSource\": \"covtypeNorm.arff\","
        "\"Evaluator\": {"
            "\"Frequency\": \"10000\","
            "\"Name\": \"BasicClassificationEvaluator\""
        "},"
        "\"Learner\": {"
            "\"LeafLearner\": \"NBAdaptive\","
            "\"Name\": \"HoeffdingTree\""
        "},"
        "\"Name\": \"EvaluatePrequential\","
        "\"Reader\": \"ArffReader\""
    "}";

	Task* task = (Task*)CREATE_CLASS(taskName);
	task->setParams(taskParams);
	task->doTask();

    delete task;
}

void testBatch(const string& fileName, const int datasetSize, const double testSetFraction = 0.25) {
    cout << "*******************" << endl;
    cout << "* BATCH LEARNING: *" << endl;
    cout << "*******************" << endl;
    cout << endl;

    cout << "Initializing data... ";
    
    stopwatch();
    const Dataset ds = readData(fileName, datasetSize);
    printTime(stopwatch());

    cout << endl;
    cout << "Dataset: " << endl;
    printData(ds);

    const TTDataset ttds = splitData(ds, testSetFraction);
    cout << "Train Set: " << endl;
    printData(ttds.train);
    cout << "Test Set: " << endl;
    printData(ttds.test);

    cout << endl;
    cout << "Training... ";

    HoeffdingTree learner;
    vector<string> features = {
        "numeric",
        "numeric",
        "numeric",
        "numeric",
        "numeric",
        "numeric",
        "numeric",
        "numeric",
        "numeric",
        "numeric",
        "numeric",
        "numeric",
        "numeric"
    };
    vector<string> classes = {
        "1100,1211,1221,1222,1240,1300,1410,1500,1600,1800,2000,3000,7000"
    };
    learner.setAttributes(features, classes);
    learner.setParams(htParams);

    BasicClassificationEvaluator evaluator;
    learner.setEvaluator(&evaluator);

    stopwatch();
    learner.process(ttds.train.X, ttds.train.y);
    printTime(stopwatch());

    cout << evaluator.toString() << endl;
    cout << "Testing... ";

    const vector<int> predictions = learner.predict(ttds.test.X);
    const size_t testSetSize = ttds.test.y.size();
    int hits = 0;

    stopwatch();
    for (int i = 0; i < testSetSize; i++) {
        if (ttds.test.y[i] == predictions[i]) {
            hits++;
        }
    }
    printTime(stopwatch());

    cout << endl;
    cout << "Accuracy: " << hits << " / " << testSetSize << " = " << (double)hits / testSetSize << endl;

    const string modelFileName = "htBatch.json";
    cout << "Saving to file: " << modelFileName << endl;
    learner.exportToFile(modelFileName);
}

void testIncremental(const string& fileName, const int datasetSize, const double testSetFraction = 0.25) {
    cout << "*************************" << endl;
    cout << "* INCREMENTAL LEARNING: *" << endl;
    cout << "*************************" << endl;
    cout << endl;

    const int trainSetSize = (int)round(max(0.0, min(1.0 - testSetFraction, 1.0)) * datasetSize);
    const int testSetSize = datasetSize - trainSetSize;

    cout << "Dataset: " << endl;
    cout << "size: " << datasetSize << endl;
    cout << "Train Set: " << endl;
    cout << "size: " << trainSetSize << endl;
    cout << "Test Set: " << endl;
    cout << "size: " << testSetSize << endl;

    ArffReader reader;
    reader.setFile(fileName);

    HoeffdingTree learner;
    learner.setParams(htParams);

    BasicClassificationEvaluator evaluator;
    learner.setEvaluator(&evaluator);

    Instance* inst;
    short state = 0;
    int hits = 0;
    int n = 0;
    while (reader.hasNextInstance()) {
        n++;
        if (datasetSize > 0 && n > datasetSize) {
            break;
        }

        inst = reader.nextInstance();

        if (n <= trainSetSize) {
            if (state == 0) {
                state++;

                cout << "Training... ";
                stopwatch();
            }
            learner.process(*inst);
        }
        else {
            if (state == 1) {
                state++;

                printTime(stopwatch());
                cout << evaluator.toString() << endl;
                cout << "Testing... ";
                stopwatch();
            }

            int prediction = learner.predict(*inst);
            if ((int)inst->getLabel() == prediction) {
                hits++;
            }
        }

        delete inst;
    }
    printTime(stopwatch());

    cout << endl;
    cout << "Accuracy: " << hits << " / " << testSetSize << " = " << (double)hits / testSetSize << endl;

    const string modelFileName = "htIncremental.json";
    cout << "Saving to file: " << modelFileName << endl;
    learner.exportToFile(modelFileName);
}

int main(int argc, char* argv[])
{
    const string fileName = "data.arff";
    const int datasetSize = 50000;
    const double testSetFraction = 0.25;

    testIncremental(fileName, datasetSize, testSetFraction);
    cout << endl;
    testBatch(fileName, datasetSize, testSetFraction);

	return 0;
}
