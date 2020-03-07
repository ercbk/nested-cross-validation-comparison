# Nested Cross-Validation with sklearn models


# Raschka method
# python


#  Notes
# 1. *** make sure target variable is the last variable in the data.frame ***
# 2. Hyperparameter values were generated from the R package dials and saved to pickle format through reticulate
# 3. Data was simulated by R package mlbench and saved in pickle format through reticulate
# 4. Best to use reticulate::source_python('path/to/script.py', envir = NULL, convert = FALSE)
# (4. cont.) envir=NULL, convert=FALSE increases the speed
# 5. Bootstrap CV strategy isn't offered by Scikit Learn and I couldn't find any other Python packages offering it.
# 6. With n_iter set to the number of rows in the grids, RandomizedGridSearch just reshuffles the grids. 
# (6. cont.) Shouldn't be done this way in real life with Latin Hypercubes because I'd think shuffling
# (6. cont.) defeats the purpose of grid algorithm. Sklearn doesn't execute ParameterGrid in parallel though,
# (6. cont.) and I'm just worried about fairly testing the speed of implentations.


# Sections
# 1. Set-up
# 2. Data
# 3. Algorithms
# 4. Hyperparameter grids
# 5. Create inner-loop tuning strategy
# 6. Run nested-cv
# 7. Train and Score Chosen Algorithm





###################################
# Set-up
###################################


# # If running with reticulate::repl_python or using reticulate::source_python, necessary in order
# # to run in parallel.
# # Should be ran before other modules imported.
# # Updates executable path in sys module.
# import sys
# import os
# exe = os.path.join(sys.exec_prefix, "pythonw.exe")
# sys.executable = exe
# sys._base_executable = exe
# # update executable path in multiprocessing module
# import multiprocessing
# multiprocessing.set_executable(exe)


# # If running with reticulate::repl_python or using reticulate::source_python, necessary in order
# # start MLflow's server
# import subprocess
# import time
# subprocess.Popen('mlflow server')
# time.sleep(10)


from pytictoc import TicToc
t = TicToc()
t.tic()

from pushbullet import Pushbullet
import os
import mlflow
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

np.random.seed(2019)

# dotenv allows for persistent environment variables
from dotenv import load_dotenv
load_dotenv()
pb_token = os.getenv('PUSHBULLET_TOKEN')
pb = Pushbullet(pb_token)

# make explicit the name of the exeriement to record to
mlflow.set_experiment("ncv_duration")



###################################
# Data
###################################


# load simulated data
# r = read mode, b = binary; pickle is binary
with open('./data/fivek-simdat.pickle', 'rb') as fried:
      pdat = pickle.load(fried)

# load elastic net regression hyperparameter values
with open('./grids/elast-latin-params.pickle', 'rb') as elastp:
      elast_params = pickle.load(elastp)

# load random forest hyperparater values
with open('./grids/rf-latin-params.pickle', 'rb') as rfp:
      rf_params = pickle.load(rfp)


y_idx = len(pdat.columns) - 1
y = pdat.iloc[:, y_idx].values
X = pdat.iloc[:, 0:y_idx]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)




####################################
# Algorithms
####################################


# Elastic Net Regression
elast_est = ElasticNet(normalize = True,
                       fit_intercept = True)

# Random Forest
rf_est = RandomForestRegressor(criterion = "mae",
                               random_state = 1)


est_dict = {'Elastic Net': elast_est, 'Random Forest': rf_est}




####################################
# Hyperparameter grids
####################################

# elastic net regression
alpha = elast_params.pop('penalty').values
l1_ratio = elast_params.pop('mixture').values
elast_grid = [{'alpha': alpha,
               'l1_ratio': l1_ratio}]

# random forest
max_features = rf_params.pop('mtry').values
n_estimators = rf_params.pop('trees').values
rf_grid = [{'max_features': max_features,
            'n_estimators': n_estimators}]


grid_dict = {'Elastic Net': elast_grid, 'Random Forest': rf_grid}




####################################
# Create inner-loop tuning strategy
####################################


# vessel for my inner-loop grid search objects
gridcvs = {}

# shuffle = True required for setting random state
# setting random state makes sure all algorithms tuned on the same splits
inner_cv = KFold(n_splits = 2, shuffle = True, random_state = 1)

# Setting this parameter to the size of the grid tells Random Search to use every grid value once
elast_iter = len(elast_params)
rf_iter = len(rf_params)
iter_dict = {'Elastic Net': elast_iter, 'Random Forest': rf_iter}


# Setting up multiple RandomSearchCV objects, 1 for each algorithm
# Collecting them in the gridcvs dict
for pgrid, est, n_iter, name in zip((elast_grid, rf_grid),
                            (elast_est, rf_est),
                            (elast_iter, rf_iter),
                            ('Elastic Net', 'Random Forest')):
    gcv = RandomizedSearchCV(estimator = est,
                       param_distributions = pgrid,
                       n_iter = n_iter,
                       scoring = 'neg_mean_absolute_error',
                       n_jobs = -1,
                       cv = inner_cv,
                       verbose = 0,
                       refit = True)
    gridcvs[name] = gcv	




####################################
# Run nested-cv
####################################

# vessel for stats on the outer fold results
results = pd.DataFrame()

# The validation set scores of the outer loop folds will be used to choose the best algorithm
# loop the grid objects we created (1 for each algorithm)
for name, gs_est in sorted(gridcvs.items()):
      outer_scores = []
      # set the outer loop cv strategy
      outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
      
      # training set is split into train/valid as the outer-loop folds
      for train_idx, valid_idx in outer_cv.split(X_train, y_train):
            # training set is fed into inner-loop for tuning
            gridcvs[name].fit(X_train.values[train_idx], y_train[train_idx])
            # best model chosen to be scored on the outer-loop valid set
            outer_scores.append(gridcvs[name].best_estimator_.score(X_train.values[valid_idx], y_train[valid_idx]))
      
      # stats calc'd on outer fold scores
      fold_score = {'model': name,
                    'mean_error': np.mean(outer_scores),
                    'sd_error': np.std(outer_scores)}
      # 1 row per algorithm
      results = results.append(fold_score, ignore_index=True)

# Choose the best algorithm based on the lowest mean error on the outer loop folds
chosen_alg = results[results.mean_error == results.mean_error.min()]['model'][0]

chosen_est = est_dict[chosen_alg]
chosen_grid = grid_dict[chosen_alg]
chosen_iter = iter_dict[chosen_alg]




####################################
# Train and Score Chosen Algorithm
####################################


# Use inner-loop tuning strategy to tune the chosen model
gcv_model_select = RandomizedSearchCV(estimator = chosen_est,
                       param_distributions = chosen_grid,
                       n_iter = chosen_iter,
                       scoring = 'neg_mean_absolute_error',
                       n_jobs = -1,
                       cv = inner_cv,
                       verbose = 0,
                       refit = True)

# Tune the chosen model on the entire training set
gcv_model_select.fit(X_train, y_train)

# given in seconds
time_elapsed = round(t.tocvalue(), 2)

# log the metric on MLflow
mlflow.log_metric('duration', time_elapsed)
tags = {'implementation': 'python', 'method': 'raschka'}
mlflow.set_tags(tags)
mlflow.end_run()


best_model = gcv_model_select.best_estimator_

# Score the best model on the entire training set and the held out test set
train_error = round(mean_absolute_error(y_true=y_train, y_pred=best_model.predict(X_train)), 5)
test_error = round(mean_absolute_error(y_true=y_test, y_pred=best_model.predict(X_test)), 5)

# Average error across tuning folds and the parameter values chosen during tuning
k_fold_score = round(-1 * gcv_model_select.best_score_, 5)
outer_kfold_score = round(results[results.model == chosen_alg]['mean_error'][0], 5)
best_hyper_vals = gcv_model_select.best_params_

# best_hyper_vals is a dict. Use it to create the df, then add the errors.
model_stats = pd.DataFrame(data = best_hyper_vals, index = [1])
model_stats = model_stats.assign(kfold_error = k_fold_score, outer_fold_error = outer_kfold_score, train_error = train_error, test_error = test_error)


# evidently has to be written into a paragraph because print outputs can't be saved and there's no glue in python.
msg = f'Python script finished in {time_elapsed} seconds. The chosen algorithm was {chosen_alg} with parameters, {best_hyper_vals}. Avg score over cv\'s test folds was {k_fold_score}. Outer fold avg score was {outer_kfold_score}. Training Error: {train_error}, Test Error: {test_error}'

# text me the results
pb.push_note("Nested CV script finished", msg)


# # only necessary if running with reticulate::repl_python or using reticulate::source_python
# # MLflow uses waitress for Windows. Killing it also kills mlflow.exe, python.exe, console window host processes
# os.system('taskkill /f /im waitress-serve.exe')
# os.system('taskkill /f /im pythonw.exe')
