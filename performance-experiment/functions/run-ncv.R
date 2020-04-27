# Runs nested cross-validation

# main function: compares the algorithms; choose the one with smallest error; predicts on a test set; calcs difference between test error and mean of outer-fold error

# calls inner-tune.R, outer-cv.R, and ncv-compare.R

# input and output description in the individual function scripts


run_ncv <- function(ncv_dat, sim_dat, large_dat, mod_FUN_list, params_list, error_FUN, method) {
   
   # output: hypervalues for each fold, chosen in the inner-loop
   best_hypervals_list <- inner_tune(
      ncv_dat = ncv_dat[[1]],
      mod_FUN_list = mod_FUN_list,
      params_list = params_list,
      error_FUN = error_FUN)
   
   # output: model, mean, median, sd error, and hyperparameter columns
   if (method == "raschka") {
      cv_stats <- outer_cv(
         ncv_dat = ncv_dat[[1]],
         best_hypervals_list = best_hypervals_list,
         mod_FUN_list = mod_FUN_list,
         error_FUN = error_FUN,
         method = method,
         train_dat = sim_dat,
         params_list = params_list)
   } else if (method == "kj") {
      cv_stats <- outer_cv(
         ncv_dat = ncv_dat[[1]],
         best_hypervals_list = best_hypervals_list,
         mod_FUN_list = mod_FUN_list,
         error_FUN = error_FUN,
         method = method)
   }
   
   # output: algorithm, hyperparams, and error values
   genl_perf_est <- ncv_compare(train_dat = sim_dat,
                                large_dat = large_dat,
                                cv_stats = cv_stats,
                                mod_FUN_list = mod_FUN_list,
                                params_list = params_list,
                                error_FUN = error_FUN,
                                method = method)
   
}