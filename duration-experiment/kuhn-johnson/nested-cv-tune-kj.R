# Nested cross-validation using tune package

# Kuhn-Johnson method
# tune



# Notes
# 1.  *** Make sure the target column is last in dataframe ***
# 2. Starts the MLflow server in the background with a OS command and kills the server process at the end of the script


# Sections
# 1. Set-up
# 2. Error functions
# 3. Model functions
# 4, Hyperparameter Grids
# 5. Functions used in the loops
# 6. Compare Algorithms




################################
# Set-up
################################


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

pacman::p_load(RPushbullet, glue, ranger, doFuture, dplyr, purrr, tidymodels, tune, dials, mlflow)


set.seed(2019)

# make explicit the name of the exeriement to record to
mlflow_set_experiment("ncv_duration")

registerDoFuture()
plan(multiprocess)



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



ncv_dat_10 <- nested_cv(small_dat,
                     outside = vfold_cv(v = 10, repeats = 2),
                     inside = bootstraps(times = 25))




################################
# Error functions
################################


error_funs <- metric_set(mae)
error_FUN <- function(y_obs, y_hat){
   y_obs <- unlist(y_obs)
   y_hat <- unlist(y_hat)
   Metrics::mae(y_obs, y_hat)
}



################################
# Mode functions
################################


# Random Forest

# inner-loop tuning
rf_inner <- rand_forest(mtry = tune(), trees = tune()) %>%
      set_engine("ranger", importance = 'impurity') %>%
      set_mode("regression")

# outer loop scoring and model selection
rf_FUN <- function(params, analysis_set) {
      mtry <- params$mtry[[1]]
      trees <- params$trees[[1]]
      rand_forest(mode = "regression", mtry = mtry, trees = trees) %>%
            set_engine("ranger", importance = 'impurity') %>%
            fit(y ~ ., data = analysis_set)
}


# Regularized Regression

glm_inner <- linear_reg(mixture = tune(), penalty = tune()) %>% 
   set_engine("glmnet")

glm_FUN <- function(params, analysis_set) {
   alpha <- params$mixture[[1]]
   lambda <- params$penalty[[1]]
   model <- parsnip::linear_reg(mixture = alpha, penalty = lambda) %>%
      parsnip::set_engine("glmnet") %>%
      generics::fit(y ~ ., data = analysis_set)
   model
}


mod_inner_list <- list(glm = glm_inner, rf = rf_inner)
mod_FUN_list <- list(glm = glm_FUN, rf = rf_FUN)



################################
# Hyperparameter Grids
################################


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


params_list <- list(glm = glm_params, rf = rf_params)



################################
# Functions used in the loops
################################


mod_error <- function(params, mod_FUN, dat) {
   y_col <- ncol(dat$data)
   y_obs <- assessment(dat)[y_col]
   mod <- mod_FUN(params, analysis(dat))
   pred <- predict(mod, assessment(dat))      
   error <- error_FUN(y_obs, pred)
   error
}


compare_algs <- function(mod_inner, params, mod_FUN, ncv_dat){
   
   # tune models by grid searching on resamples in the inner loop (e.g. 5 repeats 10 folds = list of 50 tibbles with param and mean_error cols)
   tuning_results <- map(ncv_dat$inner_resamples, function(dat, mod_inner, params) {
      tune_grid(y ~ .,
                model = mod_inner,
                resamples = dat,
                grid = params,
                metrics = error_funs)
   },
   mod_inner, params)
   
   num_params <- ncol(params)
   
   # Choose best hyperparameter combos across all the resamples for each fold (e.g. 5 repeats 10 folds = 50 best hyperparam combos)
   best_hyper_vals <- tuning_results %>%
      map_dfr(function(dat){
         dat %>% 
            collect_metrics() %>%
            filter(mean == min(mean)) %>%
            slice(1)
      }) %>% 
      select(1:num_params)
   
   # fit models on the outer-loop folds using best hyperparams (e.g. 5 repeats, 10 folds = 50 models)
   outer_fold_error <- furrr::future_map2_dbl(ncv_dat$splits, 1:nrow(best_hyper_vals), function(dat, row) {
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



################################
# Compare Algorithms
################################


lol <- list(mod_inner_list, params_list, mod_FUN_list)


# start the nested-cv
algorithm_comparison_ten <- pmap_dfr(lol, compare_algs, ncv_dat_10) %>%
   mutate(model = names(mod_FUN_list)) %>%
   select(model, everything())
toc(log = TRUE)


# log duration metric to MLflow
ncv_times <- tic.log(format = FALSE)
duration <- as.numeric(ncv_times[[1]]$toc - ncv_times[[1]]$tic)
mlflow_log_metric("duration", duration)
mlflow_set_tag("implementation", "tune")
mlflow_set_tag("method", "kj")
mlflow_end_run()


# text me results
log.txt <- tic.log(format = TRUE)
msg <- glue("Using tune-kj script: \n After running 10 fold,  {log.txt[[1]]}")
pbPost("note", title="tune-kj script finished", body=msg)
tic.clearlog()


# MLflow uses waitress for Windows. Killing it also kills mlflow.exe, python.exe, console window host processes
installr::kill_process(process = c("waitress-serve.exe"))

