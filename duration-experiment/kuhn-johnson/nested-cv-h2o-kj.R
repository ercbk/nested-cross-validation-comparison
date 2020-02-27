# Nested cross-validation for tuning and algorithm comparison

# Kuhn-Johnson method
# H2O


# Notes
# 1. *** Make sure target variable is the last column in dataframe ***
# 2. h2o_grid arg parallelism sets the number of models to be computed at a time in parallel. parallelism = 0 lets h2o decide and it will use all resources.
# 3. *** For glm models, setting parallelism = 0 causes the grid search to hang. Manually set the number of models to compute in parallel ***
# 4. H2O doesn't have a cv strategy that just fits the grid row by row. Your options are RandomDiscrete or Cartesian Grid. I set the parameter ranges so that using the Cartesian grid would create a 100 row grid to match the other scripts.
# 5. Uses few cpu resources when tuning glmnet, but does maximize cpu usage for tuning the rf


# Sections
# 1. Set-up
# 2. Error and Model Functions; 
# 3. Hyperparameter Grids
# 4. Functions used in the loops
# 5. Compare Algorithms




####################################################
# Set-Up
####################################################


# texts me when there's an error
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

pacman::p_load(glue, RPushbullet, dials, h2o, rsample, purrr, dplyr, mlflow)

# make explicit the name of the exeriement to record to
mlflow_set_experiment("ncv_duration")

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



ncv_dat_10 <- rsample::nested_cv(small_dat,
                              outside = vfold_cv(v = 10, repeats = 2),
                              inside = bootstraps(times = 25))


# Start h2o cluster
h2o.init()




###################################################
# # Error and Model Functions
###################################################


error_FUN <- function(model){
   h2o.mae(model, valid = T)
}


# Distributed Random Forest

rf_FUN <- function(x, y, anal_h2o, ass_h2o, params) {
   
   mtries <- params$mtries[[1]]
   ntrees <- params$ntrees[[1]]
   
   # h20 ususally needs unique ids or loops will return exact same values over and over
   gridId <- as.character(dqrng::dqrnorm(1))
   
   h2o.show_progress()
   
   h2o.randomForest(x = x,
                    y = y,
                    training_frame = anal_h2o,
                    model_id = modelId,
                    validation_frame = ass_h2o,
                    mtries = mtries,
                    ntrees = ntrees)
}


# Elastic Net Regression

glm_FUN <- function(x, y, anal_h2o, ass_h2o, params) {
   
   alpha <- params$alpha[[1]]
   lambda <- params$lambda[[1]]
   
   # h20 needs unique ids or loops will return exact same values over and over
   modelId <- as.character(dqrng::dqrnorm(1))
   
   h2o.show_progress()
   
   h2o.glm(x = x,
           y = y,
           training_frame = anal_h2o,
           model_id = modelId,
           validation_frame = ass_h2o,
           alpha = alpha,
           lambda = lambda)
}


mod_FUN_list <- list(glm = glm_FUN, rf = rf_FUN)

alg_list <- list(glm = "glm", rf = "drf")




#####################################################
# Hyperparameter Grids
#####################################################


# 5*20 = 100 rows for glm grid, 2*50 = 100 rows for the rf grid
params_list <- list(glm = list(alpha = c(0, 0.25, 0.5, 0.75, 1),
                               lambda = -1/log10(sample(seq(0, 1, by = 0.00001), 20))),
                    rf = list(mtries = c(3,4),
                              ntrees = seq(200, 300, by = 2))
                    )




#####################################################
# Functions used in the loops
#####################################################


# inputs params, model, and resample, calls model and error functions, outputs error
mod_error <- function(params, mod_FUN, dat) {
   anal_df <- rsample::analysis(dat)
   ass_df <- rsample::assessment(dat)
   
   h2o.no_progress()
   
   # send data to the h2o cluster
   anal_h2o <- as.h2o(anal_df)
   ass_h2o <- as.h2o(ass_df)
   
   y <- names(anal_h2o)[[ncol(anal_h2o)]]
   x <- setdiff(names(anal_h2o), y)
   
   mod <- mod_FUN(x, y, anal_h2o, ass_h2o, params)     
   error <- error_FUN(mod)
   error
}


compare_algs <- function(alg, params, mod_FUN, dat){
   
   # tune models by grid searching on resamples in the inner loop, returns df with fold id, bootstrap id, params, and errors
   tuning_results <- purrr::map(dat$inner_resamples, function(dat, alg, params){
      
      param_names <- names(params)
      
      # loops each folds set of resamples, grid search, returen table of hyperparam combos and error values
      params_errors <- purrr::map_dfr(dat$splits, function(dat, alg, params){
         
         # split into analysis and assessment sets
         anal_df <- rsample::analysis(dat)
         ass_df <- rsample::assessment(dat)
         
         # as.h2o and h2o.grid have progress bars. That's too many of progress bars.
         h2o.no_progress()
         
         # send data to the h2o cluster
         anal_h2o <- as.h2o(anal_df)
         ass_h2o <- as.h2o(ass_df)
         
         y <- names(anal_h2o)[[ncol(anal_h2o)]]
         x <- setdiff(names(anal_h2o), y)
         
         h2o.show_progress()
         
         # need a unique grid id or h2o just gives you the same predictions over aand over
         gridId <- as.character(dqrng::dqrnorm(1))
         
         mod_grid <- h2o.grid(alg, x = x, y = y,
                              grid_id = gridId,
                              training_frame = anal_h2o,
                              validation_frame = ass_h2o,
                              hyper_params = params,
                              parallelism = 8)
         # results
         mod_gridperf <- h2o.getGrid(grid_id = gridId, sort_by = "mae")
         
         # Grab hyperparams, model_id, errors
         grid_results <- mod_gridperf@summary_table
         colnames(grid_results)[4] <- "error"
         
         # clean the cluster
         h2o.removeAll()
         
         grid_results
         
      }, alg, params, .id = "bootstrap") %>% 
         mutate(error = as.numeric(error)) %>% 
         group_by_at(vars(param_names)) %>%
         summarize(mean_error = mean(error, na.rm = TRUE)) %>% 
         ungroup()
         
   }, alg, params)
   
   
   best_hyper_vals <- tuning_results %>%
      map_df(function(dat) {
         dat[which.min(dat$mean_error),]
      }) %>%
      select(names(params)) %>% 
      # H2O makes params into char vars and adds brackets to the values. Guessing they're json.
      mutate_all(~stringr::str_remove_all(., "^\\[")) %>% 
      mutate_all(~stringr::str_remove_all(., "\\]")) %>%
      mutate_all(as.numeric)
   
   # fit models on the outer-loop folds using best hyperparams (e.g. 5 repeats, 10 folds = 50 models), returns numeric with error values
   outer_fold_error <- map2_dbl(dat$splits, 1:nrow(best_hyper_vals), function(dat, row) {
      params <- best_hyper_vals[row,]
      mod_error(params, mod_FUN, dat)
   })
   
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




##################################
# Compare algorithms
##################################


lol <- list(alg_list, params_list, mod_FUN_list)

# start the nested-cv
algorithm_comparison_ten <- pmap_dfr(lol, compare_algs, ncv_dat_10) %>%
   mutate(model = names(mod_FUN_list)) %>%
   select(model, everything())
toc(log = TRUE)


# log duration metric to MLflow
ncv_times <- tic.log(format = FALSE)
duration <- as.numeric(ncv_times[[1]]$toc - ncv_times[[1]]$tic)
mlflow_log_metric("duration", duration)
mlflow_set_tag("implementation", "h2o")
mlflow_set_tag("method", "kj")
mlflow_end_run()


# text me results
log.txt <- tic.log(format = TRUE)
msg <- glue("Using h2o-kj script: \n After running 10 fold,  {log.txt[[1]]}.")
pbPost("note", title="h2o-kj script finished", body=msg)
tic.clearlog()


# MLflow uses waitress for Windows. Killing it also kills mlflow.exe, python.exe, console window host processes
installr::kill_process(process = c("waitress-serve.exe"))

# shutdown cluster
h2o.shutdown(prompt = FALSE)
