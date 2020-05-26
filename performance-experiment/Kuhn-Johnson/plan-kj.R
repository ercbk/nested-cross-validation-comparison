# Kuhn-Johnson drake plan


# Notes:
# 1. I broke the plan into units by sample size. I'm sure its possible to formulate the plan to perform the whole experiment by looping the kj and raschka method along with sample sizes into one large, more compact plan, but I wanted units that I could run overnight on my desktop.
# 2. sample_sizes: 100, 800, 2000, 5000, 10000 (maybe)
# 3. I'm trying to minimize the delta_error. Delta error is the absolute difference between the average error across the outer-folds of the nested cross-validation and the out-of-sample error which uses the chosen model and parameters to predict on a simulated 100K row dataset.





error_FUN <- function(y_obs, y_hat){
   y_obs <- unlist(y_obs)
   y_hat <- unlist(y_hat)
   Metrics::mae(y_obs, y_hat)
}

method <- "kj"
algorithms <- list("glmnet", "rf")
repeats <- seq(1:5)
grid_size <- 100

plan <- drake_plan(
   # model functions for each algorithm
   mod_FUN_list = create_models(algorithms),
   # data used to estimate out-of-sample error
   # noise_sd, seed settings are the defaults
   large_dat = mlbench_data(n = 10^5,
                            noise_sd = 1,
                            seed = 2019),
   # sample size = 100
   sim_dat_100 = mlbench_data(100),
   # hyperparameter grids for each algorithm
   # This probably doesn't need to be a "dynamic" target since mtry is only concerned about the number of columns in data (see script), but I'll do it anyways
   params_list_100 = create_grids(sim_dat_100,
                                  algorithms,
                                  size = grid_size),
   # create a separate ncv data object for each repeat value
   ncv_dat_100 = create_ncv_objects(sim_dat_100,
                                    repeats,
                                    method),
   # runs nested-cv and compares ncv error with out-of-sample error
   # outputs: ncv error, oos error, delta error, chosen algorithm, chosen hyperparameters 
   ncv_results_100 = target(
      run_ncv(ncv_dat_100,
              sim_dat_100,
              large_dat,
              mod_FUN_list,
              params_list_100,
              error_FUN,
              method),
      dynamic = map(ncv_dat_100)
   ),
   
   # repeat for the rest of the sample sizes
   # sample size = 800
   sim_dat_800 = mlbench_data(800),
   params_list_800 = create_grids(sim_dat_800,
                                  algorithms,
                                  size = grid_size),
   ncv_dat_800 = create_ncv_objects(sim_dat_800,
                                    repeats,
                                    method),
   ncv_results_800 = target(
      run_ncv(ncv_dat_800,
              sim_dat_800,
              large_dat,
              mod_FUN_list,
              params_list_800,
              error_FUN,
              method),
      dynamic = map(ncv_dat_800)
   ),

   # sample size = 2000
   sim_dat_2000 = mlbench_data(2000),
   params_list_2000 = create_grids(sim_dat_2000,
                                  algorithms,
                                  size = grid_size),
   ncv_dat_2000 = create_ncv_objects(sim_dat_2000,
                                    repeats,
                                    method),
   ncv_results_2000 = target(
      run_ncv(ncv_dat_2000,
              sim_dat_2000,
              large_dat,
              mod_FUN_list,
              params_list_2000,
              error_FUN,
              method),
      dynamic = map(ncv_dat_2000)
   ),

   # sample size = 5000
   sim_dat_5000 = mlbench_data(5000),
   params_list_5000 = create_grids(sim_dat_5000,
                                  algorithms,
                                  size = grid_size),
   ncv_dat_5000 = create_ncv_objects(sim_dat_5000,
                                    repeats,
                                    method),
   ncv_results_5000 = target(
      run_ncv(ncv_dat_5000,
              sim_dat_5000,
              large_dat,
              mod_FUN_list,
              params_list_5000,
              error_FUN,
              method),
      dynamic = map(ncv_dat_5000)
   )
   
)


