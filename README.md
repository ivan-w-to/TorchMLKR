# PyTorch-Compatible Metric Learning for Kernel Regression (TorchMLKR)

This repository seeks to streamline the production of metric learning neural networks for few-shot regression problems, problems where
the dataset is relatively small and the variability among labeled examples is relatively large. 
Deep metric learning has been popularly used for few-shot classification problems (e.g. face recognition), but the literature on metric learning for regression has been
quite scant. [Metric Learning for Kernel Regression (MLKR)](https://proceedings.mlr.press/v2/weinberger07a/weinberger07a.pdf) by Weinberger and Tesauro is a seminal work in this
area, but it focuses entirely on learning linear transformations for use in metric learning. This repository extends Weinberger and Tesauro's ideas to learning neural networks
in PyTorch. Further credit must be given to the developers of [metric-learn](https://github.com/scikit-learn-contrib/metric-learn/tree/master), a fantastic library for classical
metric learning. They offer one of the few implementations of MLKR that I know of, but unfortunately, their implementation cannot minimize a custom loss function or handle time dependent data (where future data points ought to be hidden during the training process). This repository solves both of those problems.

## Dependencies
- PyTorch
- Matplotlib (if one wishes to run the tests)
