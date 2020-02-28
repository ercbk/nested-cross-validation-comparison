# Nested cross-validation for tuning and algorithm comparison


# Raschka method
# mlr3



# Notes:
# 1. *** Make sure target variable is the last column***
# 2. As of 16Jan2020, folds are not fixed so each call to resample() creates a new nested_cv structure which means the algorithms are being compared on somewhat different data.
# 3. For the glm, lambda is not a tunable parameter in mlr3. So I had to use lambda_min_ratio and nlambda to get a grid that should be equivalent to a 100 row lambda grid.
# 4. The batch arg in the tuner function allows you to specify how you want to parallelize for each algorithm which is nice.


# Sections:
# 1. Set-Up and Data
# 2. Functions Used in the Loops
# 3. Model Functions; Hyperparameter Grids
# 4. Compare Algorithms
# 5. Tune and Score Chosen Algorithm





#####################################################
# set up and Data
#####################################################


# texts me when there's an error
options(error = function() { 
      library(RPushbullet)
      pbPost("note", "Error", geterrmessage())
      if(!interactive()) stop(geterrmessage())
})

# start MLflow server
sys::exec_background("mlflow server")
Sys.sleep(10)


library(tictoc)
tic()

pacman::p_load(glue, RPushbullet, dplyr, mlr3, mlr3learners, mlr3tuning, future, mlflow)


set.seed(2019)

# make explicit the name of the exeriement to record to
mlflow_set_experiment("ncv_duration")


plan(multiprocess)

# simulated data; generates 10 multi-patterned, numeric predictors plus outcome variable
sim_data <- function(n) {
      tmp <- mlbench::mlbench.friedman1(n, sd=1)
      tmp <- cbind(tmp$x, tmp$y)
      tmp <- as.data.frame(tmp)
      names(tmp)[ncol(tmp)] <- "y"
      tmp
}

# Use small data to tune and compare models
dat <- sim_data(5000)

train_idx <- caret::createDataPartition(y = dat$y, p = 0.80, list = FALSE)
train_dat <- dat[train_idx, ]
test_dat <- dat[-train_idx, ]




#####################################################
# Functions used in the loops
#####################################################


# task obj consists of the data and target variable
train_task <- TaskRegr$new(id = "train_dat", backend = train_dat, target = "y")

# inner-loop tuning
resampling_inner <- rsmp("cv", folds = 2)

# outer loop
resampling_outer_five <- rsmp("cv", folds = 5)

# error function
measures <- msr("regr.mae")

# required to specify an early stopping criteria. I don't want one.
terminator <- term("none")




#####################################################
# Model Functions; Hyperparameter Grids
#####################################################


# Model functions
rf_mod <- lrn("regr.ranger")
glm_mod <- lrn("regr.glmnet", nlambda = 1)

# available hyperparameters
# mlr3::mlr_learners$get("regr.ranger")$param_set
# mlr3::mlr_learners$get("regr.glmnet")$param_set

# Hyperparameter restrictions/conditions 
rf_params <- paradox::ParamSet$new(
      params = list(paradox::ParamInt$new("mtry", lower = 3, upper = 4),
                    paradox::ParamInt$new("num.trees", lower = 200, upper = 300)
      )
)

# values from for lambda.min.ratio come from the h2o docs about the parameter.
glm_params <- paradox::ParamSet$new(
      params = list(paradox::ParamDbl$new("alpha", lower = 0, upper = 1),
                    paradox::ParamDbl$new("lambda.min.ratio", lower = 0.0001, upper = 0.01)
      )
)



# generates a latin hypercube of values based on the restrictions above
# rf and glm grids consists of 100 rows
rf_grid <- paradox::generate_design_lhs(rf_params, 100)$data

glm_grid <- paradox::generate_design_lhs(glm_params, 100)$data



# Design_points is the grid strategy that allows me to make my own grids to experiment on. Batch_size indicates how many rows of the grid to tune simulataneously. Essentially how much to parallelize.
rf_tuner <- tnr("design_points", batch_size = 8, design = rf_grid)
glm_tuner <- tnr("design_points", batch_size = 8, design = glm_grid)


learner_list <- list(glm = glm_mod, rf = rf_mod)
params_list <- list(glm = glm_params, rf = rf_params)
tuner_list <- list(glm = glm_tuner, rf = rf_tuner)

lol <- list(learner_list, params_list, tuner_list)




#####################################################
# Compare algorithms
#####################################################


compare_algs <- function(mod, params, tuner, resampling_inner, resampling_outer, measures, terminator) {
      
      # Learner augmented with tuning. Tunes on the inner-loop bootstraps, then the best model will be used on the outer loop fold
      mod_tuned <- AutoTuner$new(learner = mod,
                                 resampling = resampling_inner,
                                 measures = measures,
                                 tune_ps = params,
                                 terminator = terminator,
                                 tuner = tuner)  
      
      # Initiates the ncv
      ncv_mod <- resample(task = train_task, learner = mod_tuned, resampling = resampling_outer)
      
      # pull() needs the name of the error function being used
      measure_name <- measures$id
      # Score for each fold
      outer_fold_error <- ncv_mod$score(measures) %>% 
            pull(measure_name)
      # Average score across outer loop folds
      mean_error <- ncv_mod$aggregate(measures)
      
      # output error stats and chosen hyperparams
      tibble(
            mean_error = mean_error,
            median_error = median(outer_fold_error),
            sd_error = sd(outer_fold_error)
      )
      
}


algorithm_comparison <- purrr::pmap_dfr(lol, compare_algs, resampling_inner, resampling_outer_five, measures, terminator) %>%
      mutate(model = names(learner_list)) %>%
      select(model, everything())




#####################################################
# Tune and Score the Chosen Algorithm
#####################################################


# Choose the best algorithm based on the lowest mean error on the outer loop folds
chosen_alg <- algorithm_comparison %>% 
      filter(mean_error == min(mean_error)) %>% 
      pull(1)

chosen_learner <- learner_list[[chosen_alg]]$reset()
chosen_grid <- params_list[[chosen_alg]]
chosen_tuner <- tuner_list[[chosen_alg]]

# Use inner-loop tuning strategy to tune the chosen model
chosen_tuned <- AutoTuner$new(learner = chosen_learner,
                           resampling = resampling_inner,
                           measures = measures,
                           tune_ps = chosen_grid,
                           terminator = terminator,
                           tuner = chosen_tuner)

# Tune the chosen model on the entire training set
chosen_train <- chosen_tuned$train(train_task)

toc(log = TRUE)



# log duration metric to MLflow
ncv_times <- tic.log(format = FALSE)
duration <- as.numeric(ncv_times[[1]]$toc - ncv_times[[1]]$tic)
mlflow_log_metric("duration", duration)
mlflow_set_tag("implementation", "mlr3")
mlflow_set_tag("method", "raschka")
mlflow_end_run()



# Score the best model on the entire training set and the held out test set
train_error <- round(chosen_train$predict(train_task)$score(measures), 4)
test_error <- round(chosen_train$predict_newdata(test_dat, task = train_task)$score(measures), 4)

# Average error across tuning folds and the parameter values chosen during tuning
kfold_error <- round(chosen_train$tuning_result$perf, 4)
best_params <- as.data.frame(chosen_train$tuning_result$params) %>%
      tidyr::pivot_longer(cols = names(.), names_to = "param", values_to = "value" ) %>% 
      glue_data("{param} = {value}") %>% 
      glue_collapse(sep = ",", last = " and ")


msg <- glue("Average of K-fold CV test folds: {kfold_error}
     Training Error: {train_error}
     Test Error: {test_error}
     Best Parameters for {chosen_alg}:
     {best_params}")

log.txt <- tic.log(format = TRUE)
text_msg <- glue("{log.txt[[1]]} for script to complete
                 Results:
                 {msg}")
pbPost("note", title="mlr3-raschka script finished", body=text_msg)
tic.clearlog()


# MLflow uses waitress for Windows. Killing it also kills mlflow.exe, python.exe, console window host processes
installr::kill_process(process = c("waitress-serve.exe"))


