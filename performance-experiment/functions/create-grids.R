# Create Hyperparameter grid list


# input: 
# 1, size = number of rows
# 2. algorithms = list of algorithm abbreviations
# "rf" = Ranger Random Forest
# "glmnet" = Elastic Net regression
# "svm" = Support Vector Machines

# output: list of grid objects



create_grids <- function(sim_dat, algorithms, size = 100) {
      
      # Elastic Net Regression
      
      glm_params <- dials::grid_latin_hypercube(
            dials::mixture(),
            dials::penalty(),
            size = size
      )
      
      # Random Forest
      
      # Some of the parnsip model parameters have "unknown" for the default value ranges. finalize replaces the unknowns with values based on the data.
      mtry_updated <- dials::finalize(dials::mtry(), select(sim_dat, -ncol(sim_dat)))
      
      rf_params <- dials::grid_latin_hypercube(
            mtry_updated,
            dials::trees(),
            size = size 
      )
      
      # Support Vector Machines
      
      svm_params <- dials::grid_latin_hypercube(
            dials::cost(),
            dials::margin(),
            size = size 
      )
      
      # list of grid objects depending on the algorithms inputted (switch is pretty cool)
      # stop_glue throws error if algorithm inputted isn't available (Should be in glue pkg but isn't)
      grid_list <- purrr::map(algorithms, function(alg) {
            switch(alg,
                   rf = rf_params -> alg_grid,
                   glmnet = glm_params -> alg_grid,
                   svm = svm_params -> alg_grid,
                   infer:::stop_glue("{alg} grid not available."))
            alg_grid
            
      }) %>% 
            purrr::set_names(algorithms)
}

