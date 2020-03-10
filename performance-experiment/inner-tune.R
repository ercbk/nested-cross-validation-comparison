# inner loop tuning function



pacman::p_load(dplyr, furrr, data.table, dtplyr)

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
            furrr::future_map_dfr(dat$splits, tune_over_params, mod_FUN, params, .progress = TRUE) %>%
                  lazy_dt(., key_by = param_names) %>% 
                  # For each value of the tuning parameter, compute the
                  # average <error> which is the inner bootstrap estimate.
                  group_by_at(vars(all_of(param_names))) %>%
                  summarize(mean_error = mean(error, na.rm = TRUE),
                            sd_error = sd(error, na.rm = TRUE),
                            n = length(error)) %>% 
                  as_tibble()
      }
      
      tune_algorithms <- purrr::map2(mod_FUN_list, params_list, function(mod_FUN, params){
         tuning_results <- purrr::map(ncv_dat$inner_resamples, summarize_tune_results, mod_FUN, params)
         
         # Choose best hyperparameter combos across all the resamples for each fold (e.g. 5 repeats 10 folds = 50 best hyperparam combos)
         best_hyper_vals <- tuning_results %>%
            purrr::map_df(function(dat) {
               dat %>% 
                  filter(mean_error == min(mean_error)) %>% 
                  arrange(sd_error) %>% 
                  slice(1)
            }) %>%
            select(all_of(names(params)))
      })
}


# chosen_hypervals <- inner_tune(ncv_dat = ncv_dat_list[[1]], mod_FUN_list = mod_FUN_list_ranger, params_list = params_list, error_FUN = error_FUN)

