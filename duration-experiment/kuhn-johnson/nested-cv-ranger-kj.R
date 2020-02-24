# Nested cross-validation for tuning and algorithm comparison

# Kuhn-Johnson method
# ranger



# Notes
# 1. *** Make sure the target column is last in dataframe ***
# 2. Using the Ranger Random Forest model function from the ranger package


# Sections
# 1. Set-Up
# 2. Error
# 3. Model Functions
# 4. Hyperparameter Grids
# 5. Functions used in the loops
# 6. Compare algorithms





####################################################
# Set-Up
####################################################


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

# Use small data to tune and compare models
small_dat <- sim_data(100)



pacman::p_load(RPushbullet, glue, ranger, tidymodels, data.table, dtplyr, dplyr, furrr, mlflow)

# make explicit the name of the exeriement to record to
mlflow_set_experiment("ncv_duration")

plan(multiprocess)


ncv_dat_10 <- nested_cv(small_dat,
                     outside = vfold_cv(v = 10, repeats = 2),
                     inside = bootstraps(times = 25))




##################################
# Error Functions
##################################


error_FUN <- function(y_obs, y_hat){
   y_obs <- unlist(y_obs)
   y_hat <- unlist(y_hat)
   Metrics::mae(y_obs, y_hat)
}




#####################################
# Model Functions
#####################################


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


mod_FUN_list_ranger <- list(glmnet = glm_FUN, ranger = ranger_FUN)




###################################
# Hyperparameter Grids
###################################


# size = number of rows
# Default ranges look good for mixture and penalty
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




#####################################################
# Functions used in the loops
#####################################################


# inputs params, model, and resample, calls model and error functions, outputs error
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

# inputs resample, loops hyperparam grid values to model/error function, collects error value for hyperparam combo
tune_over_params <- function(dat, mod_FUN, params) {
   params$error <- map_dbl(1:nrow(params), function(row) {
      params <- params[row,]
      mod_error(params, mod_FUN, dat)
   })
   params
}

# inputs and sends fold's resamples to tuning function, collects and averages fold's error for each hyperparameter combo
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



######################################################
# Compare algorithms
######################################################


compare_algs <- function(mod_FUN, params, ncv_dat){
   # tune models by grid searching on resamples in the inner loop (e.g. 5 repeats 10 folds = list of 50 tibbles with param and mean_error cols)
   tuning_results <- map(ncv_dat$inner_resamples, summarize_tune_results, mod_FUN, params)
   
   # Choose best hyperparameter combos across all the resamples for each fold (e.g. 5 repeats 10 folds = 50 best hyperparam combos)
   best_hyper_vals <- tuning_results %>%
      map_df(function(dat) {
         dat[which.min(dat$mean_error),]
      }) %>%
      select(names(params))
   
   # fit models on the outer-loop folds using best hyperparams (e.g. 5 repeats, 10 folds = 50 models)
   outer_fold_error <- future_map2_dbl(ncv_dat$splits, 1:nrow(best_hyper_vals), function(dat, row) {
      params <- best_hyper_vals[row,]
      mod_error(params, mod_FUN, dat)
   }, .progress = TRUE)
   
   # hyperparam values for final model will be the ones most selected to use on the outer-loop folds
   chosen_params <- best_hyper_vals %>% 
      group_by_all() %>% 
      tally() %>% 
      ungroup() %>% 
      filter(n == max(n))
   
   # output error stats and chosen hyperparams
   tibble(
      chosen_params = list(chosen_params),
      mean_error = mean(outer_fold_error),
      median_error = median(outer_fold_error),
      sd_error = sd(outer_fold_error)
   )
}



# Start the nested-cv
algorithm_comparison_ten_rang <- map2_dfr(mod_FUN_list_ranger, params_list, compare_algs, ncv_dat_10) %>%
   mutate(model = names(mod_FUN_list_ranger)) %>%
   select(model, everything())
toc(log = TRUE)


# log duration metric to MLflow
ncv_times <- tic.log(format = FALSE)
duration <- as.numeric(ncv_times[[1]]$toc - ncv_times[[1]]$tic)
mlflow_log_metric("duration", duration)
mlflow_set_tag("implementation", "ranger")
mlflow_set_tag("method", "kj")
mlflow_end_run()


# text me results
log.txt <- tic.log(format = TRUE)
msg <- glue("Using ranger-kj script: \n After running 10 fold rang,  {log.txt[[1]]}")
pbPost("note", title="ranger-kj script finished", body=msg)
tic.clearlog()


# MLflow uses waitress for Windows. Killing it also kills mlflow.exe, python.exe, console window host processes
installr::kill_process(process = c("waitress-serve.exe"))

