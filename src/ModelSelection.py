from ast import Raise
from operator import index
from matplotlib.pylab import f
from pathlib import Path
import random
import numpy as np
import math
import multiprocessing
import pandas
import itertools
import csv
import os
import ast
import pandas
import time

from MyProcess import *

from NeuralNetwork import NeuralNetwork
import ErrorFunctions

class ModelSelection:
    '''
    Implementation of the model selection algorithm
    
    Attributes:
    -----------
    backup_file: file
        file to backup the model selection's state

    '''

    def kf_train(model, data_set:np.ndarray, k:int, metrics:list, model_args:list):
        '''
        Compute the Backpropagation training algorithm on the NN for given a data set, estimating the hyperparameter performances
        trough validation in k folds of the data
        
        param data_set: a set of samples (pattern-target pairs) for supervised learning
        param k: number of data folds (data splits)
        param batch_size: parameter which determines the amount of training samples consumed in each iteration of the algorithm
            -> 1: Online
            -> 1 < batch_size < len(TR): Minibatch with minibatch size equals to batch_size
            -> len(TR): Batch
        param max_epochs: the maximum number of epochs (consumption of the whole training set) on which the algorithm will iterate
        param error_function: a string indicating the error function that the algorithm whould exploit when calculating the error distances between iterations
            -> "mee": Mean Euclidean Error
            -> "lms": Least Mean Square
        param error_decrease_tolerance: the errors difference (gain) value that the algorithm should consider as sufficiently low in order to stop training 
        param patience: the number of epochs to wait when a "no more significant error decrease" occurs
        param learning_rate: Eta hyperparameter to control the learning rate of the algorithm
        param lambda_tikhonov: Lambda hyperparameter to control the learning algorithm complexity (Tikhonov Regularization / Ridge Regression)
        param alpha_momentum: Momentum Hyperparameter

        return: mean and variance of the validation error
        '''
        
        # Computation of the size of each split
        data_len = len(data_set)
        split_size = math.floor(data_len/k)
        
        val_errors = np.empty((k, len(metrics)))
        tr_errors = np.empty(k)
        stats = {}
        split_index = 0
        
        # random shuffling the dataset to prevent biased dataset configuration
        np.random.shuffle(data_set)
        
        # At each iteration only one of the K subsets of data is used as the validation set, 
        # while all others are used for training the model validated on it.
        for i in range(k):
            split_index += split_size
            model.reset() # reset of the network to proceeds towards the next training (next fold)
            training_set = np.append(data_set[:split_size*i], data_set[split_size*(i + 1):], axis=0)
            validation_set = data_set[split_size*i : split_size*(i + 1)]
            
            new_stats = model.train(training_set, validation_set, *model_args)
            
            
            tr_errors[i] = new_stats['best_validation_training_error']
            if not stats: # first iteration
                for key in model.input_stats:
                    if key in new_stats:
                        stats[key] = new_stats[key]
                for key in model.train_stats:
                    if key in new_stats:
                        stats[key] = [new_stats[key]]
                for mes in metrics:
                    if 'training_' + mes.__name__ in new_stats:
                        stats['training_' + mes.__name__] = [new_stats['training_' + mes.__name__]]
                        stats['validation_' + mes.__name__] = [new_stats['validation_' + mes.__name__]]
                        # batch stats
                        stats['training_batch_' + mes.__name__] = [new_stats['training_batch_' + mes.__name__]]
                        stats['validation_batch_' + mes.__name__] = [new_stats['validation_batch_' + mes.__name__]]
            else: # other iterations
                for key in model.train_stats:
                    if key in new_stats:
                        stats[key].append(new_stats[key])
                for mes in metrics:
                    if 'training_' + mes.__name__ in new_stats:
                        stats['training_' + mes.__name__].append(new_stats['training_' + mes.__name__])
                        stats['validation_' + mes.__name__].append(new_stats['validation_' + mes.__name__])
                        # batch stats
                        stats['training_batch_' + mes.__name__].append(new_stats['training_batch_' + mes.__name__])
                        stats['validation_batch_' + mes.__name__].append(new_stats['validation_batch_' + mes.__name__])
                        
            outputs = model.predict_array(validation_set[:,:model.input_size])
            targets = validation_set[:,model.input_size:]
            for j, met in enumerate(metrics):
                val_errors[i, j] = met(outputs, targets)
                
                
        stats['mean_metrics'] = list(np.mean(val_errors, axis=0))
        stats['variance_metrics'] = list(np.var(val_errors, axis=0))
        stats['mean_best_validation_training_error'] = np.mean(tr_errors)

        return stats

    def __init__(self, cv_backup:str):
        '''
        Constructor of the class
        
        param cv_backup: file to backup the model selection's state
        
        return: -
        ''' 
        if cv_backup is not None:
            if cv_backup.endswith('.csv'):
                self.backup = cv_backup
            else:
                Raise(ValueError(' cv_backup extension must be .csv'))
                
        else:
            Raise(ValueError('Backup file missing'))
            
        self.partials_backup_prefix = 'tmp_'
        self.partials_backup_path = '../data/gs_data/backup'
        self.backup = cv_backup
        self.default_values =  {
        'range_min' : -0.75,
        'range_max' : 0.75,
        'fan_in' : True,
        'random_state' : None,

        'lambda_tikhonov' : 0.0,
        
        'learning_rate' : 0.1,
        'alpha_momentum' : 0.5,
        'lr_decay_tau' : 0,
        'eta_tau' : 0.01,
        'nesterov' : False,
        
        'adamax': False,
        'adamax_learning_rate': 0.002,
        'exp_decay_rate_1':0.9,
        'exp_decay_rate_2':0.999,
        
        'batch_size' : 1,
        'min_epochs' : 0,
        'max_epochs' : 100,
        'patience' : 10,
        'error_increase_tolerance' : 0.0001,
        'retraing_es_error': -1,

        'metrics':[ErrorFunctions.mean_squared_error, ],      
        'topology': {}, # must be inizialized

        'collect_data':False, 
        'collect_data_batch':False, 
        'verbose':False
        }
        self.inzialization_arg_names = ['topology', 'range_min', 'range_max', 'fan_in', 'random_state']
        self.train_arg_names = ['batch_size', 'max_epochs', 'retraing_es_error', 'error_increase_tolerance', 'patience', 'min_epochs', 
                       'learning_rate', 'lr_decay_tau', 'eta_tau',  'lambda_tikhonov', 'alpha_momentum', 'nesterov', 
                       'adamax', 'adamax_learning_rate', 'exp_decay_rate_1', 'exp_decay_rate_2',
                       'metrics', 'collect_data', 
                        'collect_data_batch', 'verbose']
    
    def __restore_backup(self, hyperparameters:list = None):
        '''
        Restore model selection's state from a backup file (csv format)

        param backup_file: backup file list to be used to restore the state
        
        return: -
        '''
        
        restore_file = self.__merge_csv_file(os.path.join(self.partials_backup_path, self.partials_backup_prefix + "0.csv"))
        
        csv = pandas.read_csv(restore_file)

        stats_columns = csv.columns.get_loc("stats")
        columns_to_drop = list(csv.columns[stats_columns:])
        csv = csv.drop(columns_to_drop, axis=1)
        
        backup_hyperparameters = csv.columns.values.tolist()

        if hyperparameters is not None:
            if backup_hyperparameters != hyperparameters: 
                return None, False
        
        done_configurations = csv.values.tolist()
        
        return done_configurations, True

    def __get_configurations(self, hyperparameters:dict, constraints:dict = {}, recovery:bool = False):
        '''
        Get all the possible configurations of the hyperparameters

        param hyperparameters: dictionary with the hyperparameters to be tested

        return: list of all the possible configurations and list of the hyperparameters' names
        '''
        done_configurations = []
        if recovery:
            done_configurations, success = self.__restore_backup(list(hyperparameters.keys()))
            
            if not success:
                Raise(ValueError('The specified hyperparameters not correspond to backup data found'))
        
        configurations = []
        names = list(hyperparameters.keys())

        for hyper_param in hyperparameters:
            if hyper_param in self.default_values.keys():
                configurations.append(hyperparameters[hyper_param])

        configurations = [list(item) for item in list(itertools.product(*configurations))]
        
        for c in configurations:
            for key in constraints:
                constraint_func = constraints[key][0]
                if constraint_func(c[names.index(key)]):
                    not_valid = constraints[key][1]
                else:
                    not_valid = constraints[key][2]

                for constraint in not_valid:
                    c[names.index(constraint)] = self.default_values[constraint]

        configurations.sort()
        configurations = list(k for k,_ in itertools.groupby(configurations))

        print("Already done: ", len(done_configurations))
        print('tot conf:', len(configurations))
        configurations = list(filter(lambda x: x not in done_configurations, configurations))
        print('remaining conf:', len(configurations))
        
        random.shuffle(configurations)
        
        return configurations, names

    def __merge_csv_file(self, results_file_name:str):
        '''
        Merge the results of the processes in a single file

        param results_file_name: name of the file obtained after the merge
        param n_proc: number of processes

        return: -
        '''
        
        backup_file = [f for f in os.listdir(self.partials_backup_path) if f.startswith(self.partials_backup_prefix)]

        backup_file = list(map(lambda f: os.path.join(self.partials_backup_path, f), backup_file))
        
        to_concat = [pandas.read_csv(f, header = 0) for f in backup_file]
        if to_concat:
            df = pandas.concat([pandas.read_csv(f, header = 0) for f in backup_file], ignore_index=True)
            
            for file in backup_file:
                os.remove(file)
                
            df.to_csv(results_file_name, index=False)

        return results_file_name

    def __process_task_trainKF(self, data_set:np.ndarray, hyperparameters:list, hyperparameters_name:list, 
                        k_folds:int = 1, backup:str = None, verbose:bool = False):
        '''
        Train the model with the given hyperparameters and the number of folds

        param dataset: dataset to be used for K-Fold cross validation
        param hyperparameters: dict of hyperparameters' configurations to be used for validation
        param hyperparameters_name: list of hyperparameters' names
        param k_folds: number of folds to be used in the cross validation
        param topology: topology of the neural network
        param backup: backup file to be used to write the results

        return: -

        '''
        # metric is a default param
        metrics_name = [m.__name__ for m in self.default_values['metrics']]
        if not os.path.isfile(backup): 
            back_up = open(backup, 'a+') 
            writer = csv.writer(back_up)
            writer.writerow(hyperparameters_name + 
                                       ['stats'] +
                                       ['mean_' + x for x in metrics_name] + 
                                       ['var_' + x for x in metrics_name] + 
                                       ['mean_best_validation_training_error'])
            back_up.flush()
        else: # if file exists i only add more data
            back_up = open(backup, 'a') 
            writer = csv.writer(back_up)
            

        # for every configuration create a new clean model and train it
        for index_con, configuration in enumerate(hyperparameters):
            grid_val = self.default_values.copy()
            for i, hyper_param in enumerate(configuration): 
                grid_val[hyperparameters_name[i]] = hyper_param

            # create a new model
            grid_val['topology'] = ast.literal_eval(grid_val['topology'])
            args_init = [grid_val[key] for key in self.inzialization_arg_names]
            nn = NeuralNetwork(*args_init)
            
            
            # train the model
            grid_val['learning_rate'] = grid_val['learning_rate'] / grid_val['batch_size']
            grid_val['adamax_learning_rate'] = grid_val['adamax_learning_rate'] / grid_val['batch_size']
            grid_val['eta_tau'] = grid_val['learning_rate']/100 # eta tau more or less 1% of eta_0
            args_train = [grid_val[key] for key in self.train_arg_names] 
            if verbose: print("Training a new model : ", args_train)
            
            try:
                
                print('pid:', os.getpid(), ' started new kfold' , index_con + 1, '/', len(hyperparameters))
                stats = ModelSelection.kf_train(nn, data_set, k_folds, grid_val['metrics'], args_train)
                
                list_to_write =(list(configuration) + 
                                [stats] + 
                                [x for x in stats['mean_metrics']] + 
                                [x for x in stats['variance_metrics']] + 
                                [stats['mean_best_validation_training_error']])
                writer.writerow(list_to_write)
            
            
            except Exception:
                writer.writerow(list(configuration) + [None, None, None, None, None]) 
            
            back_up.flush()

        back_up.close()

    def grid_searchKF(self, data_set:np.ndarray, hyperparameters:dict = {}, k_folds:int = 2, n_proc:int = 1, recovery:bool = False, constraints:dict = {}, ):
        '''
        Implementation of the grid search algorithm

        param data_set: training set to be used in the grid search
        param hyperparameters: dictionary with the hyperparameters to be tested
        param k_folds: number of folds to be used in the cross validation
        param topology: topology of the neural network
        param topology_name: name of the network topology
        param n_proc: number of processes to be used in the grid search

        return: the best hyperparameters' configuration
        '''
        
        hyperparameters = dict(sorted(hyperparameters.items()))
        configurations, names = self.__get_configurations(hyperparameters, constraints, recovery)
        print('tot conf to do:', len(configurations))
        if n_proc == 1: # sequential execution
            self.__process_task_trainKF(data_set, configurations, names, k_folds, self.backup)
            return
        
        remainder = len(configurations) % n_proc
        single_conf_size = int(len(configurations) / n_proc)
        start = end = 0
        j = 0
        proc_pool = []
        partial_data_dir = Path(self.partials_backup_path).absolute()

        if not os.path.exists(partial_data_dir):
            os.makedirs(partial_data_dir)

        for i in range(n_proc): # distribute equally the workload among the processes
            time.sleep(1)
            start = end
            if remainder > 0:
                end += single_conf_size + 1
            else:
                end += single_conf_size
            
            j = i+1
            process = multiprocessing.Process(target=self.__process_task_trainKF, 
                                              args=(data_set, 
                                                    configurations[start:end],
                                                    names, 
                                                    k_folds, 
                                                    os.path.join(partial_data_dir, f''+ self.partials_backup_prefix +f'{j}.csv')),
                                              daemon=True)
            proc_pool.append(process)
            process.start()
            
            remainder -= 1
           
        for process in proc_pool: # join all the terminated processes
            process.join()

        self.__merge_csv_file(self.backup)
    
    def __process_task_trainHO(self, training_set:np.ndarray, validation_set:np.ndarray, hyperparameters:list, 
                        hyperparameters_name:list, backup:str = None, verbose:bool = True):
        
        '''
        Train the model with the given configuration of hyperparameters

        param training_set: training set to be used for hold out validation
        param validation_set: validation set to be used for hold out validation
        param hyperparameters: list of hyperparameters' configurations to be used for validation
        param hyperparameters_name: list of hyperparameters' names
        param: backup: backup file
        param: verbose: verbosity of the algorithm

        return: -
        '''

        if os.path.isfile(backup):
            back_up = open(backup, 'a')
            writer = csv.writer(back_up)
        else:
            back_up = open(backup, 'a+')
            writer = csv.writer(back_up)
            writer.writerow(hyperparameters_name + ['stats', 'metrics_names', 'errors'])

        # for every configuration create a new clean model and train it
        for configuration in hyperparameters:
            grid_val = self.default_values.copy()
            for i, key in enumerate(hyperparameters_name): grid_val[key] = configuration[i]

            # create a new model
            metrics_name = [m.__name__ for m in grid_val['metrics']]
            grid_val['topology'] = ast.literal_eval(grid_val['topology'])
            args_init = [grid_val[key] for key in self.inzialization_arg_names]
            nn = NeuralNetwork(*args_init)
            # train the model
            args_train = [grid_val[key] for key in self.train_arg_names]
            

            stats = nn.train(training_set, validation_set, *args_train)
            metrics_error = [stats['validation_' + mes.__name__][-1] for mes in grid_val['metrics']]
            writer.writerow(list(configuration) + [stats, metrics_name, ])
            back_up.flush()
        
        back_up.close()

    def grid_searchHO(self, training_set:np.ndarray, validation_set:np.ndarray, hyperparameters:dict, 
                      n_proc:int = 1, recovery:bool = False):
        '''
        Implementation of the grid search algorithm using hold out validation

        param training_set: training set to be used in the grid search
        param validation_set: validation set to be used in the grid search
        param hyperparameters: dictionary with the hyperparameters to be tested
        param topology: topology of the neural network
        param n_proc: number of processes to be used in the grid search
        param topology_name: name of the network topology

        return: the best hyperparameters' configuration
        
        '''
        
        hyperparameters = dict(sorted(hyperparameters.items()))
        configurations, names = self.__get_configurations(hyperparameters, recovery)

        if n_proc == 1: # sequential execution
            self.__process_task_trainHO(training_set, validation_set, configurations, names, self.backup)
            return

        remainder = len(configurations) % n_proc
        single_conf_size = int(len(configurations) / n_proc)
        start = end = 0
        j = 0
        proc_pool = []
        
        partial_data_dir = Path(self.partials_backup_path).absolute()
        if not os.path.exists(partial_data_dir):
            os.makedirs(partial_data_dir)

        for i in range(n_proc): # distribute equally the workload among the processes
            start = end
            if remainder > 0:
                end += single_conf_size + 1
            else:
                end += single_conf_size
            j = i+1
            process = multiprocessing.Process(target=self.__process_task_trainHO, args=(training_set, validation_set, configurations[start:end],
                                                                                 names, os.path.join(partial_data_dir, f''+ self.partials_backup_prefix +f'{j}.csv'), ))
            proc_pool.append(process)
            process.start()
            
            remainder -= 1
           
        for process in proc_pool: # join all the terminated processes
            process.join()

        self.__merge_csv_file(self.backup)
