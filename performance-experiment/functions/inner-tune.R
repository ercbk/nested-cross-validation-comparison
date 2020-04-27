# inner loop tuning function

# inputs:
# 1. ncv_dat = one ncv object from the list created by create-ncv-objects.R
# 2. mod_FUN_list = all the model objects created by create-models.R
# 3. params_list = all the hyperparameter grids created by create-grids.R
# 4. error_FUN = specified at the start of plan-<method>.R 

# outputs: df of hyperparameters for each fold that was chosen in the inner-loop



inner_tune <- function(ncv_dat, mod_FUN_list, params_list, error_FUN) {
   
   # inputs params, model, and resample, calls model and error functions, outputs error
   mod_error <- function(params, mod_FUN, dat) {
      y_col <- ncol(dat$data)
      y_obs <- rsample::assessment(dat)[y_col]
      mod <- mod_FUN(params, rsample::analysis(dat))
      pred <- predict(mod, rsample::assessment(dat))
      if (!is.data.frame(pred)) {
         pred <- pred$predictions
      }
      error <- error_FUN(y_obs, pred)
      error
   }
   
   # inputs resample, loops hyperparam grid values to model/error function, collects error value for hyperparam combo
   tune_over_params <- function(dat, mod_FUN, params) {
      params$error <- purrr::map_dbl(1:nrow(params), function(row) {
         params <- params[row,]
         mod_error(params, mod_FUN, dat)
      })
      params
   }
   
   # inputs and sends fold's resamples to tuning function, collects and averages fold's error for each hyperparameter combo
   summarize_tune_results <- function(dat, mod_FUN, params) {
      # Return row-bound tibble that has the 25 bootstrap results
      param_names <- names(params)
      furrr::future_map_dfr(dat$splits, tune_over_params, mod_FUN, params, .progress = FALSE) %>%
      lazy_dt(., key_by = param_names) %>% 
         # For each value of the tuning parameter, compute the
         # average <error> which is the inner bootstrap estimate.
         group_by_at(vars(param_names)) %>%
         summarize(mean_error = mean(error, na.rm = TRUE),
                   sd_error = sd(error, na.rm = TRUE),
                   n = length(error)) %>% 
         as_tibble()
   }
   
   tune_algorithms <- furrr::future_map2(mod_FUN_list, params_list, function(mod_FUN, params){
      tuning_results <- purrr::map(ncv_dat$inner_resamples, summarize_tune_results, mod_FUN, params)
      
      # Choose best hyperparameter combos across all the resamples for each fold (e.g. 5 repeats 10 folds = 50 best hyperparam combos)
      best_hyper_vals <- tuning_results %>%
         purrr::map_df(function(dat) {
            dat %>% 
               filter(mean_error == min(mean_error)) %>% 
               arrange(sd_error) %>% 
               slice(1)
         }) %>%
         select(names(params))
   })
}



