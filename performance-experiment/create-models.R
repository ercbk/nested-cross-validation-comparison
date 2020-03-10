# Creates list of model functions

# input: list of algorithm abbreviations
# "rf" = Ranger Random Forest
# "glmnet" = Elastic Net regression
# "svm" = Support Vector Machines

# output: list of model functions

pacman::p_load(dplyr)

create_models <- function(algorithms) {
      
      # Random Forest
      
      ranger_FUN <- function(params, analysis_set) {
            mtry <- params$mtry[[1]]
            trees <- params$trees[[1]]
            model <- ranger::ranger(y ~ ., data = analysis_set, mtry = mtry, num.trees = trees)
            model
      }
      
      # Elastic Net Regression
      
      glm_FUN <- function(params, analysis_set) {
            alpha <- params$mixture[[1]]
            lambda <- params$penalty[[1]]
            model <- parsnip::linear_reg(mixture = alpha, penalty = lambda) %>%
                  parsnip::set_engine("glmnet") %>%
                  generics::fit(y ~ ., data = analysis_set)
            model
      }
      
      # Support Vector Machines
      
      svm_FUN <- function(params, analysis_set) {
            cost <- params$cost[[1]]
            model <- kernlab::ksvm(y ~ ., data = analysis_set,  C = cost)
            model
      }
      
      mod_FUN_list <- purrr::map(algorithms, function(alg) {
            switch(alg,
                   rf = ranger_FUN -> mod_fun,
                   glmnet = glm_FUN -> mod_fun,
                   svm = svm_FUN -> mod_fun,
                   infer:::stop_glue("{alg} model function not available."))
            mod_fun
            
      }) %>% 
            purrr::set_names(algorithms)
}


