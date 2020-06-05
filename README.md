# ml-rapids: Incremental learning written in C++ exposed in Python

`ml-rapids` implements incremental learning methods in C++ and exposes them via SWIG in Python. Installation can be achieved simply with `pip install ml_rapids`. You can test your installation with running Python:
```python
# testing ml-rapids
import ml_rapids
ml_rapids.test()
```

Further documentation is available here:
* [Users' Manual](https://github.com/JozefStefanInstitute/ml-rapids/blob/master/docs/MANUAL.md)

## Implemented incremental learning methods

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
The library will be exposed via also via `npm` packages.


## Development
Development notes can be read [here](https://github.com/JozefStefanInstitute/ml-rapdis/docs/DEV.md).

Python deployment notes can be read [here](https://github.com/JozefStefanInstitute/ml-rapdis/docs/PYPI-DEPLOY.md).   

## Acknowledgements
`ml-rapids` is developed by AILab at Jozef Stefan Institute.

This repository is based strongly on [streamDM-cpp](https://github.com/huawei-noah/streamDM-Cpp).

Project has received funding from European Union's Horizon 2020 Research and Innovation Programme under the Grant Agreement [776115](http://www.perceptivesentinel.eu/) (PerceptiveSentinel).
