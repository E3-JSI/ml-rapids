# ml-rapids: Incremental learning written in C++ exposed in Python and NodeJS

`ml-rapids` implements incremental learning methods in C++ and exposes them via SWIG in Python and NodeJS.

## Incremental learning methods:

* Classification
    * Majority Class
    * Naive Bayes
    * Logistic Regression
    * Perceptron
    * VFDT (Very Fast Decision Trees) aka Hoeffding Trees
    * HAT (Hoeffding Adaptive Trees)
    * Bagging
* Regression
    * /

All the methods implement [`sklearn`](https://scikit-learn.org/) incremantal learner interface (includes `fit`, `partial_fit` and `predict` methods).


## Future plans
Streaming random forest on top of Hoeffding trees will be implemented.

The library will be exposed via `pypi` and `npm` packages.

Python:

* `pip install ml-rapids`

NodeJS:

* `npm install ml-rapids`


## Development
Development notes can be read [here](./docs/DEV.md).

## Acknowledgements
`ml-rapids` is developed by AILab at Jozef Stefan Institute.

This repository is based strongly on [streamDM-cpp](https://github.com/huawei-noah/streamDM-Cpp).

Project has received funding from European Union's Horizon 2020 Research and Innovation Programme under the Grant Agreement [776115](http://www.perceptivesentinel.eu/) (PerceptiveSentinel).
