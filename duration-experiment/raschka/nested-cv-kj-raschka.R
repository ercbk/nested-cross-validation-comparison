# Nested cross-validation for tuning and algorithm comparison


# Raschka method
# ranger-kj



# Notes
# 1. *** Make sure the target column is last in dataframe ***


# Sections
# 1. Set-Up
# 2. Error function 
# 3. Model Functions  
# 4. Hyperparameter Grids
# 5. Functions used in the loops
# 6. Compare algorithms; tune chosen algorithm
# 7. Score chosen algorithm





######################################################
# Set-Up
######################################################


# texts me if an error occurs
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

set.seed(2019)

# simulated data; generates 10 multi-patterned, numeric predictors plus outcome variable
sim_data <- function(n) {
      tmp <- mlbench::mlbench.friedman1(n, sd=1)
      tmp <- cbind(tmp$x, tmp$y)
      tmp <- as.data.frame(tmp)
      names(tmp)[ncol(tmp)] <- "y"
      tmp
}

dat <- sim_data(5000)


pacman::p_load(RPushbullet, glue, ranger, tidymodels, data.table, dtplyr, dplyr, furrr, mlflow)

# make explicit the name of the exeriement to record to
mlflow_set_experiment("ncv_duration")

plan(multiprocess)


train_idx <- caret::createDataPartition(y = dat$y, p = 0.80, list = FALSE)
train_dat <- dat[train_idx, ]
test_dat <- dat[-train_idx, ]

ncv_dat <- nested_cv(train_dat,
                       outside = vfold_cv(v = 5),
                       inside = vfold_cv(v = 2))




######################################################
# Error Functions
######################################################


error_FUN <- function(y_obs, y_hat){
      y_obs <- unlist(y_obs)
      y_hat <- unlist(y_hat)
      Metrics::mae(y_obs, y_hat)
}



######################################################
# Model Functions
######################################################


# Random Forest

ranger_FUN <- function(params, analysis_set) {
      mtry <- params$mtry[[1]]
      trees <- params$trees[[1]]
      model <- ranger::ranger(y ~ ., data = analysis_set, mtry = mtry, num.trees = trees)
      model
}


# Regularized Regression

glm_FUN <- function(params, analysis_set) {
      alpha <- params$mixture[[1]]
      lambda <- params$penalty[[1]]
      model <- parsnip::linear_reg(mixture = alpha, penalty = lambda) %>%
            parsnip::set_engine("glmnet") %>%
            generics::fit(y ~ ., data = analysis_set)
      model
}


mod_FUN_list <- list(glmnet = glm_FUN, ranger = ranger_FUN)



######################################################
# Hyperparameter Grids
######################################################


# size = number of rows
glm_params <- grid_latin_hypercube(
      mixture(),
      penalty(),
      size = 100
)

rf_params <- grid_latin_hypercube(
      mtry(range = c(3, 4)),
      trees(range = c(200, 300)),
      size = 100
)

params_list <- list(glmnet = glm_params, ranger = rf_params)




######################################################
# Functions used in the loops
######################################################


# inputs params, model, and fold, calls model and error functions, outputs error
mod_error <- function(params, mod_FUN, dat) {
      y_col <- ncol(dat$data)
      y_obs <- assessment(dat)[y_col]
      mod <- mod_FUN(params, analysis(dat))
      pred <- predict(mod, assessment(dat))
      if (!is.data.frame(pred)) {
            pred <- pred$predictions
      }
      error <- error_FUN(y_obs, pred)
      error
}

# inputs fold, loops hyperparam grid values to model/error function, collects error value for hyperparam combo
tune_over_params <- function(dat, mod_FUN, params) {
      params$error <- map_dbl(1:nrow(params), function(row) {
            params <- params[row,]
            mod_error(params, mod_FUN, dat)
      })
      params
}

# inputs outer fold and sends outer fold's inner folds to tuning function, collects and averages fold's error for each hyperparameter combo
summarize_tune_results <- function(dat, mod_FUN, params) {
      # Return row-bound tibble that has the 25 bootstrap results
      param_names <- names(params)
      future_map_dfr(dat$splits, tune_over_params, mod_FUN, params, .progress = TRUE) %>%
            lazy_dt(., key_by = param_names) %>% 
            # For each value of the tuning parameter, compute the
            # average <error> which is the inner bootstrap estimate.
            group_by_at(vars(param_names)) %>%
            summarize(mean_error = mean(error, na.rm = TRUE),
                      n = length(error)) %>% 
            as_tibble()
}


# primary function for the nested-cv
compare_algs <- function(mod_FUN, params, ncv_dat){
      # tune models by grid searching on resamples in the inner loop (e.g. 5 repeats 10 folds = list of 50 tibbles with param and mean_error cols)
      tuning_results <- map(ncv_dat$inner_resamples, summarize_tune_results, mod_FUN, params)
      
      # Choose best hyperparameter combos across all the resamples for each fold (e.g. 5 repeats 10 folds = 50 best hyperparam combos)
      best_hyper_vals <- tuning_results %>%
            map_df(function(dat) {
                  dat[which.min(dat$mean_error),]
            }) %>%
            select(names(params))
      
      # fit models on the outer-loop folds using best hyperparams
      outer_fold_error <- future_map2_dbl(ncv_dat$splits, 1:nrow(best_hyper_vals), function(dat, row) {
            params <- best_hyper_vals[row,]
            mod_error(params, mod_FUN, dat)
      }, .progress = TRUE)
      
      
      # output error stats
      tibble(
            mean_error = mean(outer_fold_error),
            median_error = median(outer_fold_error),
            sd_error = sd(outer_fold_error)
      )
}




######################################################
# Compare algorithms, Tune chosen algorithm
######################################################


# outputs df with outer fold stats for each algorithm
algorithm_comparison <- map2_dfr(mod_FUN_list, params_list, compare_algs, ncv_dat) %>%
      mutate(model = names(mod_FUN_list)) %>%
      select(model, everything())

# Choose alg with lowest avg error
chosen_alg <- algorithm_comparison %>% 
      filter(mean_error == min(mean_error)) %>% 
      pull(1)

# Set inputs to chosen alg
mod_FUN <- mod_FUN_list[[chosen_alg]]
params <- params_list[[chosen_alg]]
total_train <- vfold_cv(train_dat, v = 2)

# tune chosen alg on the inner-loop cv strategy
# code is an amalgam of funs: summarize_tune_results, tune_over_params
tuning_results <- map(total_train$splits, function(dat, mod_FUN, params) {
      params$error <- future_map_dbl(1:nrow(params), function(row) {
            params <- params[row,]
            mod_error(params, mod_FUN, dat)
      })
      return(params)
}, mod_FUN, params) %>% 
      bind_rows() %>% 
      lazy_dt(., key_by = names(params)) %>% 
      # For each value of the tuning parameter, compute the
      # average <error> which is the inner bootstrap estimate.
      group_by_at(vars(names(params))) %>%
      summarize(mean_error = mean(error, na.rm = TRUE)) %>% 
      as_tibble()




######################################################
# Score chosen algorithm
######################################################


# Get best params from the tuning
best_hyper_vals <- tuning_results %>%
      filter(mean_error == min(mean_error)) %>%
      slice(1) %>% 
      select(names(params))

# Get avg error across validation folds from alg using best params
avg_kfold_error <- tuning_results %>% 
      filter(mean_error == min(mean_error)) %>%
      mutate(mean_error = round(mean_error, 5)) %>% 
      select(mean_error)

# train tuned mod on entire training set; score on test set
chosen_mod <- mod_FUN(best_hyper_vals, train_dat)

toc(log = TRUE)



# log duration metric to MLflow
ncv_times <- tic.log(format = FALSE)
duration <- as.numeric(ncv_times[[1]]$toc - ncv_times[[1]]$tic)
mlflow_log_metric("duration", duration)
mlflow_set_tag("implementation", "ranger-kj")
mlflow_set_tag("method", "raschka")
mlflow_end_run()


# Score on the held out test set
chosen_preds <- predict(chosen_mod, test_dat)
if (!is.data.frame(chosen_preds)) {
   chosen_preds <- chosen_preds$predictions
}
test_error <- round(error_FUN(test_dat$y, chosen_preds), 5)


# Create output message and text me the results
best_hyper_vals <- best_hyper_vals %>% 
      tidyr::pivot_longer(cols = names(.), names_to = "param", values_to = "value" ) %>% 
      glue_data("{param} = {value}") %>% 
      glue_collapse(sep = ",", last = " and ")

msg <- glue("Avg K-Fold CV error: {avg_kfold_error[[1]]}
     Test Error: {test_error}
     Best Parameters for {chosen_alg}:
     {best_hyper_vals}")

log.txt <- tic.log(format = TRUE)
text_msg <- glue("{log.txt[[1]]} for kj-raschka script to complete
                 Results:
                 {msg}")

pbPost("note", title="kj-raschka script finished", body=text_msg)
tic.clearlog()


# MLflow uses waitress for Windows. Killing it also kills mlflow.exe, python.exe, console window host processes
installr::kill_process(process = c("waitress-serve.exe"))
