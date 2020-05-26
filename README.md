
# Nested Cross-Validation: Comparing Methods and Implementations

### (In-progress)

Nested cross-validation has become a recommended technique for
situations in which the size of our dataset is insufficient to
simultaneously handle hyperparameter tuning and algorithm comparison.
Examples of such situations include: proof of concept, start-ups,
medical studies, time series, etc. Using standard methods such as k-fold
cross-validation in these cases may result in significant increases in
optimization bias. Nested cross-validation has been shown to produce low
bias, out-of-sample error estimates even using datasets with only
hundreds of rows and therefore gives a better judgement of
generalization performance.

The primary issue with this technique is that it is computationally very
expensive with potentially tens of 1000s of models being trained during
the process. While researching this technique, I found two slightly
different methods of performing nested cross-validation — one authored
by [Sabastian
Raschka](https://github.com/rasbt/stat479-machine-learning-fs19/blob/master/11_eval4-algo/code/11-eval4-algo__nested-cv_verbose1.ipynb)
and the other by [Max Kuhn and Kjell
Johnson](https://tidymodels.github.io/rsample/articles/Applications/Nested_Resampling.html).  
I’ll be examining two aspects of nested cross-validation:

1.  Duration: Which packages and functions give us the fastest
    implementation of each method?  
2.  Performance: First, develop a testing framework. Then, using a
    generated dataset, find how many repeats, given the number of
    samples, should we expect to need in order to obtain a reasonably
    accurate out-of-sample error estimate.

With regards to the question of speed, I’ll will be testing
implementations of both methods from various packages which include
{tune}, {mlr3}, {h2o}, and {sklearn}.

## Duration Experiment

Experiment details:

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

1.  Hyperparameter value grids  
2.  Outer-Loop CV strategy  
3.  Inner-Loop CV strategy  
4.  Grid search strategy

These elements also affect the run times. Both methods will be using the
same size grids, but Kuhn-Johnson uses repeats and more folds in the
outer and inner loops while Raschka’s trains an extra model over the
entire training set at the end at the end. Using Kuhn-Johnson, 50,000
models will be trained for each algorithm — using Raschka’s, 1,001
models.

MLFlow was used to keep track of the duration (seconds) of each run
along with the implementation and method used. I’ve used implementation
to describe the various changes in coding structures that accompanies
using each package’s functions. A couple examples are the python
for-loop being replaced with a while-loop and `iter_next` function when
using {reticulate} and {mlr3} entirely using R’s R6 Object Oriented
Programming system.

![](duration-experiment/outputs/0225-results.png)

![](duration-experiment/outputs/duration-pkg-tbl.png)

![](README_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

## Performance Experiment

Experiment details:

  - The fastest implementation of each method will be used in running a
    nested cross-validation with different sizes of data ranging from
    100 to 5000 observations and different numbers of repeats of the
    outer-loop cv strategy.
      - The {mlr3} implementation was the fastest for Raschka’s method,
        but the Ranger-Kuhn-Johnson implementation is close. So I’ll be
        using Ranger-Kuhn-Johnson for both methods.  
  - The chosen algorithm and hyperparameters will used to predict on a
    100K row simulated dataset and the mean absolute error will be
    calculated for each combination of repeat, data size, and method.  
  - Runtimes began to explode after n = 800 for my 8 vcpu, 16 GB RAM
    desktop, so I ran this experiment using AWS instances: a r5.2xlarge
    for the Elastic Net and a r5.24xlarge for Random Forest.  
  - I’ll be iterating through different numbers of repeats and sample
    sizes, so I’ll be transitioning from imperative scripts to a
    functional approach. Given the long runtimes and impermanent nature
    of my internet connection, it would be nice to cache each iteration
    as it finishes. The [{drake}](https://github.com/ropensci/drake)
    package is superb on both counts, so I’m using it to orchestrate.

![](README_files/figure-gfm/perf_bt_charts-1.png)<!-- -->

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
