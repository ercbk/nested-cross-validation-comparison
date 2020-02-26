# Nested Cross-validation using Scikit-Learn


# Raschka's method
# reticulate


# Notes
# 1. Uses the reticulate pkg to implement Raschka's nested cv
# 2. *** make sure target variable is the last variable in the data.frame ***
# 3. Bootstrap CV strategy isn't offered by Scikit Learn and I couldn't find any other Python packages offering it.
# 4. With n_iter set to the number of rows in the grids, RandomizedGridSearch just reshuffles the grids. Shouldn't be done this way in real life with Latin Hypercubes because I'd think shuffling defeats the purpose of grid algorithm. Sklearn doesn't execute ParameterGrid in parallel though, and I'm just worried about fairly testing the speed of implentations.


# Sections:
# 1. Set-Up
# 2. Data
# 3. Estimators
# 4. Hyperparameter Grids
# 5. Inner and Outer CV Strategies
# 6. Compare Algorithms
# 7. Tune and Score Chosen Algorithm





######################################
# Set-up
######################################


# text me if any errors occur
options(error = function() { 
   library(RPushbullet)
   pbPost("note", "Error", geterrmessage())
   if(!interactive()) stop(geterrmessage())
})


# start MLflow server
sys::exec_background("mlflow server")
Sys.sleep(10)


library(tictoc)
tic()

pacman::p_load(RPushbullet, dplyr, glue, reticulate, mlflow)

# make explicit the name of the exeriement to record to
mlflow_set_experiment("ncv_duration")


#------- Required for Multiprocessing with reticulate (in just Windows?)

# update executable path in sys module
sys <- import("sys")
exe <- file.path(sys$exec_prefix, "pythonw.exe")
sys$executable <- exe
sys$`_base_executable` <- exe

# update executable path in multiprocessing module
multiprocessing <- import("multiprocessing")
multiprocessing$set_executable(exe)

#-------


sk_lm <- import("sklearn.linear_model")
sk_e <- import("sklearn.ensemble")
sk_ms <- import("sklearn.model_selection")
sk_m <- import("sklearn.metrics")

set.seed(2019)
py_set_seed(2019)


######################################
# Data
######################################


sim_data <- function(n) {
   tmp <- mlbench::mlbench.friedman1(n, sd=1)
   tmp <- cbind(tmp$x, tmp$y)
   tmp <- as.data.frame(tmp)
   names(tmp)[ncol(tmp)] <- "y"
   tmp
}

dat <- sim_data(5000)

pdat = r_to_py(dat)

y_idx <- py_len(pdat$columns) - 1
X_idx <- y_idx - 1
y = pdat$iloc(axis = 1L)[y_idx]$values
X = pdat$iloc(axis = 1L)[0:X_idx]

dat_splits<- sk_ms$train_test_split(X, y,
                                test_size=0.2,
                                random_state=1L)

X_train <- dat_splits[[1]]
X_test <- dat_splits[[2]]
y_train <- as.numeric(dat_splits[[3]])
y_test <- as.numeric(dat_splits[[4]])




######################################
# Estimators
######################################

# Elastic Net Regression
elast_est <- sk_lm$ElasticNet(normalize = TRUE,
                         fit_intercept = TRUE)

# Random Forest
rf_est <- sk_e$RandomForestRegressor(criterion = "mae",
                                     random_state = 1L)


alg_list <- list(elastic_net = elast_est, rf = rf_est)




######################################
# Hyperparameter grids
######################################


# Elastic Net Regression
elast_params <- r_to_py(dials::grid_latin_hypercube(
   dials::mixture(),
   dials::penalty(),
   size = 100
))
alpha <- elast_params$pop('penalty')$values
l1_ratio <- elast_params$pop('mixture')$values
elast_grid <- py_dict(list('alpha', 'l1_ratio'),
                      list(alpha, l1_ratio))



# Random Forest
rf_params <- r_to_py(dials::grid_latin_hypercube(
   dials::mtry(range = c(3, 4)),
   dials::trees(range = c(200, 300)),
   size = 100
))
max_features <- rf_params$pop('mtry')$values
n_estimators <- rf_params$pop('trees')$values
rf_grid <- py_dict(list('max_features', 'n_estimators'), list(max_features, n_estimators))


grid_list <- list(elastic_net = elast_grid, rf = rf_grid)




######################################
# Inner and Outer CV strategies
######################################

# Setting the inner-loop tuning strategy
inner_cv <- sk_ms$KFold(n_splits = 2L,
                        shuffle = TRUE,
                        random_state = 1L)

# Setting the outer-loop cv strategy
outer_cv <- sk_ms$KFold(n_splits = 5L,
                        shuffle = TRUE,
                        random_state = 1L)

# Setting n_iter to the total rows for each grid just reshuffles the grids
n_iter_list <- list(elastic_net = py_len(elast_params), rf = py_len(rf_params))

lol <- list(alg_list, grid_list, n_iter_list)

# Setting up multiple RandomSearchCV objects, 1 for each algorithm
# Collecting them in the inner-loop list
inner_loop <- purrr::pmap(lol, function(alg, grid, n_iter) {
   sk_ms$RandomizedSearchCV(estimator = alg,
                      param_distributions = grid,
                      n_iter = n_iter,
                      scoring = 'neg_mean_absolute_error',
                      cv = inner_cv,
                      n_jobs = -1L,
                      pre_dispatch = '2*n_jobs',
                      refit = TRUE)
})




######################################
# Compare Algorithms
######################################


algorithm_comparison <- purrr::map(inner_loop, function(grid_search) {
   
   outer_scores <- list()
   counter <- 0
   
   outer_split <- outer_cv$split(X_train, y_train)
   
   # while loop + iter_next = a python for-loop
   while (TRUE) {
      
      # python methods create "iterable" objs
      fold <- iter_next(outer_split)
      # loop ends once we run out of folds
      if (is.null(fold))
         break
      
      # python 0-indexed, so need to add 1 in order to correctly subset in R
      train_idx <- as.integer(fold[[1]]) + 1
      valid_idx <- as.integer(fold[[2]]) + 1
      
      pX_train <- r_to_py(X_train[train_idx,])
      pX_valid <- r_to_py(X_train[valid_idx,])
      
      py_train <- r_to_py(y_train[train_idx])
      py_valid <- r_to_py(y_train[valid_idx])
      
      # training set is fed into inner-loop for tuning
      tuning_results <- grid_search$fit(pX_train, py_train)
      
      counter <- sum(counter, 1)
      
      # best model chosen to be scored on the outer-loop valid set
      outer_scores[[counter]] <- data.frame(error = tuning_results$best_estimator_$score(pX_valid, py_valid))
   }
   
   # stats calc'd on outer fold scores
   outer_stats <- outer_scores %>% 
      bind_rows() %>% 
      summarize(mean_error = mean(error, na.rm = TRUE),
                median_error = median(error, na.rm = TRUE),
                sd_error = sd(error, na.rm = TRUE))
})




######################################
# Tune and Score the Chosen Algorithm
######################################


# Choose the best algorithm based on the lowest mean error on the outer loop folds
chosen_alg <- purrr::map_dfr(algorithm_comparison, ~select(., mean_error), .id = "model") %>% 
   filter(mean_error == min(mean_error)) %>% 
   pull(1)

alg <- alg_list[[chosen_alg]]
grid <- grid_list[[chosen_alg]]
n_iter <- n_iter_list[[chosen_alg]]

# Use inner-loop tuning strategy to tune the chosen model
chosen_tuned <- sk_ms$RandomizedSearchCV(estimator = alg,
                                   param_distributions = grid,
                                   n_iter = n_iter,
                                   scoring = 'neg_mean_absolute_error',
                                   cv = inner_cv,
                                   n_jobs = -1L,
                                   refit = TRUE)

# Tune the chosen model on the entire training set
chosen_train <- chosen_tuned$fit(X_train, y_train)

toc(log = TRUE)



# log duration metric to MLflow
ncv_times <- tic.log(format = FALSE)
duration <- as.numeric(ncv_times[[1]]$toc - ncv_times[[1]]$tic)
mlflow_log_metric("duration", duration)
mlflow_set_tag("implementation", "reticulate")
mlflow_set_tag("method", "raschka")
mlflow_end_run()



chosen_model <- chosen_train$best_estimator_

# Score the best model on the entire training set and the held out test set
train_error <- round(sk_m$mean_absolute_error(y_true = y_train, y_pred = chosen_model$predict(X_train)), 4)
test_error <- round(sk_m$mean_absolute_error(y_true = y_test, y_pred = chosen_model$predict(X_test)), 4)

# Average error across tuning folds and the parameter values chosen during tuning
kfold_error <- round(-chosen_tuned$best_score_, 4)
best_params <- as.data.frame(chosen_tuned$best_params_) %>%
   tidyr::pivot_longer(cols = names(.), names_to = "param", values_to = "value" ) %>% 
   glue_data("{param} = {value}") %>% 
   glue_collapse(sep = ",", last = " and ")


msg <- glue("Average of K-fold CV test folds: {kfold_error}
     Training Error: {train_error}
     Test Error: {test_error}
     Best Parameters for {chosen_alg}:
     {best_params}")

log.txt <- tic.log(format = TRUE)
text_msg <- glue("{log.txt[[1]]} for script to complete
                 Results:
                 {msg}")

# text me the results
pbPost("note", title="reticulate-raschka script finished", body=text_msg)
tic.clearlog()


# MLflow uses waitress for Windows. Killing it also kills mlflow.exe, python.exe, console window host processes
installr::kill_process(process = c("waitress-serve.exe", "pythonw.exe"))
