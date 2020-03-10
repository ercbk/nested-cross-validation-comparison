




source("performance-experiment/mlbench-data.R")
source("performance-experiment/create-ncv.R")
source("performance-experiment/create-models.R")
source("performance-experiment/create-grids.R")
source("performance-experiment/inner-tune.R")
source("performance-experiment/outer-cv.R")
source("performance-experiment/ncv-compare.R")

# options(error = function() { 
#       library(RPushbullet)
#       pbPost("note", "Error", geterrmessage())
#       if(!interactive()) stop(geterrmessage())
# })
# 
# 
# library(tictoc)
# tic()
# 
# 
# pacman::p_load(RPushbullet, glue)

set.seed(2019)

plan(multiprocess)

method <- "raschka"
# method <- "kj"
algorithms <- list("glmnet", "rf")

# sample_sizes <- c(100, 800, 2000, 5000, 10000)
# repeats <- seq(1:5)

sample_sizes <- 100
repeats <- 1

# method or method list?

large_dat <- mlbench_data(n = 10^5, noise_sd = 1, seed = 2019)

simdat_list <- purrr::map(sample_sizes, ~mlbench_data(.x))

ncv_dat_list <- create_ncv(dat = simdat_list, repeats = repeats, method = method)


error_FUN <- function(y_obs, y_hat){
      y_obs <- unlist(y_obs)
      y_hat <- unlist(y_hat)
      Metrics::mae(y_obs, y_hat)
}

mod_FUN_list <- create_models(algorithms)

params_list <- create_grids(algorithms, size = 100)

ncv_results <- purrr::map2_dfr(ncv_dat_list, simdat_list, function(ncv_dat, sim_dat) {
   
   best_hypervals_list <- inner_tune(
      ncv_dat = ncv_dat,
      mod_FUN_list = mod_FUN_list,
      params_list = params_list,
      error_FUN = error_FUN)
   
   # model, mean, median, sd error, and parameter columns
   if (method == "raschka") {
      cv_stats <- outer_cv(
         ncv_dat = ncv_dat,
         best_hypervals_list = best_hypervals_list,
         mod_FUN_list = mod_FUN_list,
         error_FUN = error_FUN,
         method = method,
         train_dat = sim_dat,
         params_list = params_list)
   } else if (method == "kj") {
      cv_stats <- outer_cv(
         ncv_dat = ncv_dat,
         best_hypervals_list = best_hypervals_list,
         mod_FUN_list = mod_FUN_list,
         error_FUN = error_FUN,
         method = method)
   }
   
   genl_perf_est <- ncv_compare(train_dat = sim_dat,
                                large_dat = large_dat,
                                cv_stats = cv_stats,
                                mod_FUN_list = mod_FUN_list,
                                params_list = params_list,
                                error_FUN = error_FUN,
                                method = method)
   
})

indices <- tidyr::crossing(sample_sizes, repeats)

perf_exp_results <- indices %>% 
   bind_cols(ncv_results)

