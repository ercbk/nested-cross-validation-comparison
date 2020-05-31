
# Nested Cross-Validation: Comparing Methods and Implementations

### (In-progress)

![](images/ncv.png)

Nested cross-validation has become a recommended technique for
situations in which the size of our dataset is insufficient to
simultaneously handle hyperparameter tuning and algorithm comparison.
Examples of such situations include: proof of concept, start-ups,
medical studies, time series, etc. Using standard methods such as k-fold
cross-validation in these cases may result in substantial increases in
optimization bias. Nested cross-validation has been shown to produce
less biased, out-of-sample error estimates even using datasets with only
hundreds of rows and therefore gives a better judgement of
generalization performance.

The primary issue with this technique is that it can be computationally
expensive with potentially tens of 1000s of models being trained during
the process. While researching this technique, I found two slightly
different variations of performing nested cross-validation — one
authored by [Sabastian
Raschka](https://github.com/rasbt/stat479-machine-learning-fs19/blob/master/11_eval4-algo/code/11-eval4-algo__nested-cv_verbose1.ipynb)
and the other by [Max Kuhn and Kjell
Johnson](https://tidymodels.github.io/rsample/articles/Applications/Nested_Resampling.html).

Various elements of the technique affect the run times and performance.
These include:

1.  Hyperparameter value grids  
2.  Grid search strategy  
3.  Inner-Loop CV strategy  
4.  Outer-Loop CV strategy

I’ll be examining two aspects of nested cross-validation:

1.  Duration: Find out which packages and combinations of model
    functions give us the fastest implementation of each method.  
2.  Performance: First, develop a testing framework. Then, for a given
    data generating process, how large of sample size is needed to
    obtain reasonably accurate out-of-sample error estimate? And how
    many repeats in the outer-loop cv strategy should be used to
    calculate this error estimate?

## Duration

#### Experiment details:

  - Random Forest and Elastic Net Regression algorithms  
  - Both algorithms are tuned with 100x2 hyperparameter grids using a
    latin hypercube design.  
  - From {mlbench}, I’m using the generated data set, friedman1, from
    Friedman’s Multivariate Adaptive Regression Splines (MARS) paper.
  - Kuhn-Johnson
      - 100 observations: 10 features, numeric target variable  
      - outer loop: 2 repeats, 10 folds  
      - inner loop: 25 bootstrap resamples  
  - Raschka
      - 5000 observations: 10 features, numeric target variable  
      - outer loop: 5 folds  
      - inner loop: 2 folds

The sizes of the data sets are the same as those in the original scripts
by the authors. Using Kuhn-Johnson, 50,000 models (grid size \* number
of repeats \* number of folds in the outer-loop \* number of
folds/resamples in the inner-loop) are trained for each algorithm —
using Raschka’s, 1,001 models for each algorithm. The one extra model in
the Raschka variation is due to his method of choosing the
hyperparameter values for the final model. He performs an extra k-fold
cross-validation using the inner-loop cv strategy on the entire training
set. Kuhn-Johnson uses majority vote. Whichever set of hyperparameter
values has been chosen during the inner-loop tuning procedure the most
often is the set used to fit the final model.

[MLFlow](https://mlflow.org/docs/latest/index.html) is used to keep
track of the duration (seconds) of each run along with the
implementation and method used.

![](duration-experiment/outputs/0225-results.png)

![](duration-experiment/outputs/duration-pkg-tbl.png)

![](README_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

## Performance

#### Experiment details:

  - The same data, algorithms, and hyperparameter grids are used.
  - The fastest implementation of each method is used in running a
    nested cross-validation with different sizes of data ranging from
    100 to 5000 observations and different numbers of repeats of the
    outer-loop cv strategy.
      - The {mlr3} implementation is the fastest for Raschka’s method,
        but the Ranger-Kuhn-Johnson implementation is close. To
        simplify, I am using
        [Ranger-Kuhn-Johnson](https://github.com/ercbk/nested-cross-validation-comparison/blob/master/duration-experiment/kuhn-johnson/nested-cv-ranger-kj.R)
        for both methods.  
  - The chosen algorithm with hyperparameters is fit on the entire
    training set, and the resulting final model predicts on a 100K row
    Friedman dataset.  
  - The percent error between the the average mean absolute error (MAE)
    across the outer-loop folds and the MAE of the predictions on this
    100K dataset is calculated for each combination of repeat, data
    size, and method.  
  - To make this experiment manageable in terms of runtimes, I am using
    AWS instances: a r5.2xlarge for the Elastic Net and a r5.24xlarge
    for Random Forest.
      - Also see the Other Notes section  
  - Iterating through different numbers of repeats, sample sizes, and
    methods makes a functional approach more appropriate than running
    imperative scripts. Also, given the long runtimes and impermanent
    nature of my internet connection, it would also be nice to cache
    each iteration as it finishes. The
    [{drake}](https://github.com/ropensci/drake) package is superb on
    both counts, so I’m using it to orchestrate.

![](README_files/figure-gfm/kj_patch_kj-1.png)<!-- -->

#### Results:

  - Runtimes for n = 100 and n = 800 are close, and there’s a large jump
    in runtime going from n = 2000 to n = 5000.  
  - The number of repeats has little effect on the amount of percent
    error.
  - For n = 100, there is substantially more variation in percent error
    than in the other sample sizes.  
  - While there is a large runtime cost that comes with increasing the
    sample size from 2000 to 5000 observations, it doesn’t seem to
    provide any benefit in gaining a more accurate estimate of the
    out-of-sample error.

![](README_files/figure-gfm/kj-patch-1.png)<!-- -->

#### Results:

  - The longest runtime is under 30 minutes, so runtime isn’t a large
    consideration if we are making a choice about sample size.  
  - There isn’t much difference in runtime between n = 100 and n =
    2000.  
  - For n = 100, there’s a relatively large change in percent error when
    going from 1 repeat to 2 repeats. The error estimate then stabilizes
    for repeats 3 through 5.  
  - n = 5000 gives poorer out-of-sample error estimates than n = 800 and
    n = 2000 for all values of repeats.  
  - n = 800 remains under 2.5% percent error for all repeat values, but
    also shows considerable volatility.

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
