
# Nested Cross-Validation: Comparing Methods and Implementations

Nested cross-validation has become a recommended technique for
situations in which the size of our dataset is insufficient to handle
both hyperparameter tuning and algorithm comparison. Using standard
k-fold cross-validation in such situations results in significant
optimization bias. Nested cross-validation has been shown to provide an
unbiased estimation of out-of-sample error using datasets with only a
few hundred rows.

The primary issue with this technique is that it is computationally very
expensive with potentially tens of 1000s of models being trained in the
process. This experiment seeks to answer two questions:  
1\. Which implementation is fastest?  
2\. How many *repeats*, given the size of the training set, should we
expect to need to obtain a reasonably accurate out-of-sample error
estimate?

While researching this technique, I found two *methods* of performing
nested cross-validation — one authored by [Sabastian
Raschka](https://github.com/rasbt/stat479-machine-learning-fs19/blob/master/11_eval4-algo/code/11-eval4-algo__nested-cv_verbose1.ipynb)
and the other by [Max Kuhn and Kjell
Johnson](https://tidymodels.github.io/rsample/articles/Applications/Nested_Resampling.html).

With regards to the question of speed, I’ll will be testing
implementations of both methods from various packages which include
{tune}, {mlr3}, {h2o}, and {sklearn}.

Duration experiment details:

  - Random Forest and Elastic Net Regression algorithms  
  - Both with 100x2 hyperparameter grids  
  - Kuhn-Johnson
      - 100 observations 10 features, numeric target variable  
      - outer loop: 2 repeats, 10 folds  
      - inner loop: 25 bootstrap resamples  
  - Raschka
      - 5000 observations: 10 features, numeric target variable  
      - outer loop: 5 folds  
      - inner loop: 2 folds

(Size of the data sets are the same as those in the original scripts by
the authors)

Various elements of the technique can be altered to improve performance.
These include:  
1\. Hyperparameter value grids  
2\. Outer-Loop CV strategy  
3\. Inner-Loop CV strategy  
4\. Grid search strategy

For the performance experiemnt (question 2), I’ll be varying the repeats
of the outer-loop cv strategy for each method. The fastest
implementation of each method will be tuned with different sizes of data
ranging from 100 to 5000 observations. The mean absolute error will be
calculated for each combination of repeat, data size, and method.

I’m using a 4 core, 16 GB RAM machine.

Progress (duration in seconds)

![](duration-experiment/kuhn-johnson/outputs/0224-results.png)

References

Boulesteix, AL, and C Strobl. 2009. “Optimal Classifier Selection and
Negative Bias in Error Rate Estimation: An Empirical Study on
High-Dimensional Prediction.” BMC Medical Research Methodology 9 (1):
85.
[link](https://www.researchgate.net/publication/40756303_Optimal_classifier_selection_and_negative_bias_in_error_rate_estimation_An_empirical_study_on_high-dimensional_prediction)

Sabastian Raschka, “STAT 479 Statistical Tests and Algorithm
Comparison,” (Lecture Notes, University of Wisconsin-Madison, Fall
2019).
[link](https://github.com/rasbt/stat479-machine-learning-fs19/blob/master/11_eval4-algo/11-eval4-algo__notes.pdf)

Sudhir Varma and Richard Simon. “Bias in error estimation when using
cross-validation for model selection”. In: BMC bioinformatics 7.1
(2006). p. 91.
[link](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-91)
