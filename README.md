# nested-cross-validation-comparison
Analysis of various implementations and methods of nested cross-validation in R and Python

Nested cross-validation has become a recommended technique for situations in which the size of our dataset is insufficient to handle both hyperparameter tuning and algorithm comparison. Using standard k-fold cross-validation in such situations results in  significant optimization bias. Nested cross-validation has been shown to provide an unbiased estimation of out-of-sample error using datasets with only a few hundred rows.

The primary issue with this technique is that it is computationally very expensive with potentially 1000s of models being trained in the process. This analysis seeks to answer two questions:
1. Which implementation is fastest?
2. How many *repeats*, given the size of the training set, should we expect to need to obtain an accurate out-of-sample error estimate?

While researching this technique, I found two *methods* of performing nested cross-validation — one authored by [Sabastian Raschka](https://github.com/rasbt/stat479-machine-learning-fs19/blob/master/11_eval4-algo/code/11-eval4-algo__nested-cv_verbose1.ipynb) and the other by [Max Kuhn and Kjell Johnson](https://tidymodels.github.io/rsample/articles/Applications/Nested_Resampling.html).

With regards to the question of speed, this analysis will be examining implementations of both methods from various packages which include {tune}, {mlr3}, {h2o}, and {sklearn}.

The fastest implementation of each method will then be used to answer the second question for each method.

(experiment ongoing)
