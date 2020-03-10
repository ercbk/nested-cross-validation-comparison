# ncv_compare function


# Chooses the best algorithm, fits best model on entire training set, predicts against large simulated data set


ncv_compare <- function(train_dat, large_dat, cv_stats, mod_FUN_list, params_list, error_FUN, method) {
   
   if (method == "kj") {
      # Choose alg with lowest avg error
      chosen_alg <- cv_stats %>%
         bind_rows(.id = "model") %>% 
         filter(mean_error == min(mean_error)) %>% 
         pull(model)
      
      # Set inputs to chosen alg
      mod_FUN <- mod_FUN_list[[chosen_alg]]
      params <- cv_stats[[chosen_alg]] %>%
         select(names(params_list[[chosen_alg]]))
      
   } else if (method == "raschka") {
      chosen_alg <- cv_stats %>% 
         pull(model)
      mod_FUN <- mod_FUN_list[[chosen_alg]]
      params <- cv_stats %>% 
         filter(model == chosen_alg) %>% 
         select(names(params_list[[chosen_alg]]))
   }
   
   fit <- mod_FUN(params, train_dat)
   
   # fit <- mod_FUN(params, ncv_dat_list$sim_data[[1]])
   preds <- predict(fit, large_dat)
   if (!is.data.frame(preds)) {
      preds <- preds$predictions
   }
   
   # calculate out-of-sample and retrieve nested-cv error
   y_col <- ncol(large_dat)
   y_obs <- large_dat[y_col]
   oos_error <- round(error_FUN(y_obs, preds), 5)
   
   if (method == "kj") {
      ncv_error <- cv_stats[[chosen_alg]] %>%
         mutate(mean_error = round(mean_error, 5)) %>% 
         pull(mean_error)
   } else if (method == "raschka") {
      ncv_error <- cv_stats %>%
         filter(model == chosen_alg) %>% 
         mutate(mean_error = round(mean_error, 5)) %>% 
         pull(mean_error)
   }
   
   # delta (the difference between errors) is how well the ncv estimated generalization performance
   ncv_perf <- bind_cols(oos_error = oos_error, ncv_error = ncv_error) %>% 
      mutate(method = method,
             delta_error = abs(oos_error - ncv_error),
             chosen_algorithm = chosen_alg) %>% 
      bind_cols(params) %>% 
      select(method, everything())
   
}


   


