# Create Hyperparameter grid list


# input: 
# 1, size = number of rows
# 2. algorithms = list of algorithm abbreviations
# "rf" = Ranger Random Forest
# "glmnet" = Elastic Net regression
# "svm" = Support Vector Machines

pacman::p_load(dplyr)

create_grids <- function(algorithms, size = 100) {
      
      # Elastic Net Regression
      
      glm_params <- dials::grid_latin_hypercube(
            dials::mixture(),
            dials::penalty(),
            size = size
      )
      
      # Random Forest
      
      rf_params <- dials::grid_latin_hypercube(
            dials::mtry(range = c(3, 4)),
            dials::trees(range = c(200, 300)),
            size = size 
      )
      
      # Support Vector Machines
      
      svm_params <- dials::grid_latin_hypercube(
            dials::cost(),
            dials::margin(),
            size = size 
      )
      
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

