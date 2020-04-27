# outer_cv function

# Takes the hyperparameter values that were chosen by the tuning process in the inner loop and uses them for cross-validation in the outer loop.

# inputs:
# 1. ncv_dat = ncv obj from list created by create-ncv-objects.R
# 2. best_hypervals_list = output from inner-tune.R
# 3. mod_FUN_list = output from create-models.R
# 4. error_FUN = error function given at start of plan_<method>.R
# 5. method = "kj" or "raschka"
# 6. train_dat = entire training set; output from mlbench-data.R
# 7. params_list = list of hyperparameter grids; output from create-grids.R

# output: df with chosen model, chosen hyperparameters, outer-fold error stats: mean, median, sd error for each algorithm




outer_cv <- function(ncv_dat, best_hypervals_list, mod_FUN_list, error_FUN, method, train_dat = NULL, params_list = NULL) {
   if (method == "raschka" & is.null(train_dat)) {
      stop("train_dat argument = NULL. Entire training set needs to be included for raschka method")
   }
   if (method == "raschka" &  is.null(params_list)) {
      stop("params_list argument = NULL. Hyperparameter grid list needs to be included for raschka method")
   }
   
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
   outer_stats <- furrr::future_map2(mod_FUN_list, best_hypervals_list, function(mod_FUN, best_hyper_vals){
      
      # fit models on the outer-loop folds using best hyperparams (e.g. 5 repeats, 10 folds = 50 models)
      outer_fold_error <- furrr::future_map2_dfr(ncv_dat$splits, 1:nrow(best_hyper_vals), function(dat, row) {
         params <- best_hyper_vals[row,]
         error <- mod_error(params, mod_FUN, dat)
         tibble(
            error = error
         )
      }, 
      # progress bar off when working with clusters
      .progress = FALSE) %>% 
         bind_cols(best_hyper_vals) %>% 
         mutate_all(~round(., 6))
      
      
      if (method == "kj") {
         # hyperparam values for final model will be the ones most selected to use on the outer-loop folds
         chosen_params <- best_hyper_vals %>% 
            group_by_all() %>% 
            tally() %>% 
            ungroup() %>% 
            filter(n == max(n)) %>% 
            slice(1)
         
         # if majority vote chooses more than one parameter set, then choose the set with the lowest error. And take the first row in case more than one set of params has min error.
         if (nrow(chosen_params) > 1) {
            chosen_params <- chosen_params %>%
               mutate_all(~round(., 6)) %>% 
               inner_join(outer_fold_error, by = c("mixture", "penalty")) %>%
               filter(error == min(error)) %>%
               slice(1) %>%
               select(names(best_hyper_vals))
         }
         
         # output error stats and chosen hyperparams
         tibble(
            mean_error = mean(outer_fold_error$error),
            median_error = median(outer_fold_error$error),
            sd_error = sd(outer_fold_error$error)
         ) %>% 
            bind_cols(chosen_params)
         
      } else if (method == "raschka") {
         tibble(
            mean_error = mean(outer_fold_error$error),
            median_error = median(outer_fold_error$error),
            sd_error = sd(outer_fold_error$error)
         )
         
      } else {
         stop("Need to specify method as kj or raschka", call. = FALSE)
      }
   })
   
   if (method == "raschka") {
      chosen_alg <- outer_stats %>% 
         bind_rows(.id = "model") %>% 
         filter(mean_error == min(mean_error)) %>% 
         pull(1)
      
      # Set inputs to chosen alg
      mod_FUN <- mod_FUN_list[[chosen_alg]]
      params <- params_list[[chosen_alg]]
      total_train <- rsample::vfold_cv(train_dat, v = 2)
      
      # tune chosen alg on the inner-loop cv strategy
      # code is an amalgam of funs: summarize_tune_results, tune_over_params
      tuning_results <- furrr::future_map(total_train$splits, function(dat, mod_FUN, params) {
         params$error <- furrr::future_map_dbl(1:nrow(params), function(row) {
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
         summarize(mean_error = mean(error, na.rm = TRUE),
                   sd_error = sd(error, na.rm = TRUE)) %>% 
         as_tibble()
      
      # Get best params from the tuning
      chosen_hyper_vals <- tuning_results %>%
         filter(mean_error == min(mean_error)) %>%
         arrange(sd_error) %>%
         slice(1) %>% 
         select(names(params))
      
      outer_stats <- outer_stats %>% 
         bind_rows(.id = "model") %>% 
         filter(model == chosen_alg) %>% 
         bind_cols(chosen_hyper_vals)
   }
   return(outer_stats)
}



