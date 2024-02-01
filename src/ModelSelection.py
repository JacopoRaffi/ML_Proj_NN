from ast import Raise
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

from NeuralNetwork import NeuralNetwork
import ErrorFunctions

class ModelSelection:
    '''
    Implementation of the model selection algorithm
    
    Attributes:
    -----------
    self.partials_backup_prefix: str
        the prefix of the backup file path
    self.partials_backup_path: str
        the foldere where to store the backup
    self.backup: file
        the backup file of the istance
    self.default_values: dict
        the default input values used when training new models
    self.inzialization_arg_names: list of string
        the ordered list that contains the input of the NeuralNetwork's constructor
    self.train_arg_names: list of string
        the ordered list that contains the input of the train method of NeuralNetwork
    '''

    def kf_train(model, data_set:np.ndarray, k:int, metrics:list, model_args:list):
        '''
        Compute the Backpropagation training algorithm on the NN for a given data set, estimating the hyperparameter performances
        trough k-fold validation
        
        Parameters
        ----------
        data_set: np.ndarray
            A set of samples (pattern-target pairs) for supervised learning
        k: int
            Number of data folds (data splits)
        metrics: list of callable
            List of metrics used in the training of the model
        model_args: list of arguments given in input when training a the model

        Returns
        -------
        return: a dct containing all the stats accumulated by the model during training, plus mean and variance
                of the final evaluation metrics computed for every fold
        '''
        
        # Computation of the size of each split
        data_len = len(data_set)
        split_size = math.floor(data_len/k)
        
        # inizializing usefoul variables
        val_errors = np.empty((k, len(metrics)))
        tr_errors = np.empty(k)
        # return val
        stats = {}
        split_index = 0
        
        # random shuffling the dataset to prevent biased dataset configuration
        np.random.shuffle(data_set)
        
        # At each iteration only one of the K subsets of data is used as the validation set, 
        # while all others are used for training the model validated on it.
        for i in range(k):
            # index to compute the validation splits
            split_index += split_size
            model.reset() # reset of the network to proceeds towards the next training (next fold)
            # computing the new training annd validation sets
            training_set = np.append(data_set[:split_size*i], data_set[split_size*(i + 1):], axis=0)
            validation_set = data_set[split_size*i : split_size*(i + 1)]
            
            # training and gathering stats
            new_stats = model.train(training_set, validation_set, *model_args)
            
            # here the stats are accumulated intelligently
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
                        
            # we compute the final valdation metrics
            outputs = model.predict_array(validation_set[:,:model.input_size])
            targets = validation_set[:,model.input_size:]

            for j, met in enumerate(metrics):
                metr = met(outputs, targets)
                val_errors[i, j] = metr
            
        # practical stats change
        stats['mean_metrics'] = list(np.mean(val_errors, axis=0))
        stats['variance_metrics'] = list(np.var(val_errors, axis=0))
        stats['mean_best_validation_training_error'] = np.mean(tr_errors)

        return stats

    def __init__(self, cv_backup:str):
        '''
        Constructor of the class
        
        Parameters
        ----------
        cv_backup: str
            File to backup the state of the computations
        
        Returns
        -------
        return: -
        ''' 
        # if backup is not good we raise an exeption
        if cv_backup is not None:
            if cv_backup.endswith('.csv'):
                self.backup = cv_backup
            else:
                Raise(ValueError(' cv_backup extension must be .csv'))
        else:
            Raise(ValueError('Backup file missing'))
        
        # inziializing usefoul variables
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
        'alpha_momentum' : 0.0,
        'lr_decay_tau' : 0,
        'eta_tau' : 0.01,
        'nesterov' : False,
        
        'adamax': False,
        'adamax_learning_rate': 0.002,
        'exp_decay_rate_1':0.9,
        'exp_decay_rate_2':0.999,
        
        'batch_size' : 1,
        'min_epochs' : 0,
        'max_epochs' : 512,
        'patience' : 5,
        'error_increase_tolerance' : math.inf,
        'retrainig_es_error': -1,

        'metrics':[ErrorFunctions.mean_squared_error, ],      
        'topology': {}, # must be inizialized

        'collect_data':False, 
        'collect_data_batch':False, 
        'verbose':False
        }
        self.inzialization_arg_names = NeuralNetwork.inzialization_input.copy()
        self.train_arg_names = ['batch_size', 
                                'max_epochs', 
                                'min_epochs',
                                'retrainig_es_error',
                                'patience', 
                                'error_increase_tolerance', 
                                
                                'lambda_tikhonov',
                                
                                'adamax',
                                'adamax_learning_rate',
                                'exp_decay_rate_1',
                                'exp_decay_rate_2',
                                
                                'learning_rate',
                                'lr_decay_tau',
                                'eta_tau',         
                                'alpha_momentum', 
                                'nesterov',
                                
                                'metrics', 
                                'collect_data', 
                                'collect_data_batch', 
                                'verbose']

    def __restore_backup(self, hyperparameters:list = None):
        '''
        Restore model selection's state from a backup folder

        Parameters
        ----------
        hyperparameters: list
            List of hyperparameters
        
        Returns
        ------- 
        return: -
        '''
        
        # merge all the useful files
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

        Parameters
        ----------
        hyperparameters: dict
            Dictionary with the hyperparameters to be tested
        constraints: dict
            Dictionary that contains the constraints (see grid_search_kfold)
        recovery: bool
            If a previous saved state is being recovered

        Returns
        -------
        return: list
            List of all the possible configurations and list of the hyperparameters' names
        '''
        # if the istance is starting from a pre-existing backup, not all configuration needs to be tested
        done_configurations = []
        if recovery:
            done_configurations, success = self.__restore_backup(list(hyperparameters.keys()))
            
            if not success:
                Raise(ValueError('The specified hyperparameters not correspond to backup data found'))
        
        configurations = []
        names = list(hyperparameters.keys())

        # here all the possible combination are computed
        for hyper_param in hyperparameters:
            if hyper_param in self.default_values.keys():
                configurations.append(hyperparameters[hyper_param])
        configurations = [list(item) for item in list(itertools.product(*configurations))]
        
        # here the constrains dict is used to delete all the matching configurations
        for c in configurations:
            for key in constraints:
                constraint_func = constraints[key][0]
                if constraint_func(c[names.index(key)]):
                    not_valid = constraints[key][1]
                else:
                    not_valid = constraints[key][2]

                for constraint in not_valid:
                    c[names.index(constraint)] = self.default_values[constraint]

        # here the duplicated and modified configuration are eliminated
        configurations.sort()
        configurations = list(k for k,_ in itertools.groupby(configurations))

        print("Already done: ", len(done_configurations))
        print('tot conf:', len(configurations))
        configurations = list(filter(lambda x: x not in done_configurations, configurations))
        print('remaining conf:', len(configurations))

        # we shuffle the configuration to equally distribute them through all the processes
        random.shuffle(configurations)
        
        return configurations, names

    def __merge_csv_file(self, results_file_name:str):
        '''
        Merge the results of the processes in a single file

        Parameters
        ----------
        results_file_name: str
            Name of the file obtained after the merge
            
        Returns
        -------
        return: -
        '''
        
        # all the configuration are taken from the files, one for each process
        backup_file = [f for f in os.listdir(self.partials_backup_path) if f.startswith(self.partials_backup_prefix)]
        backup_file = list(map(lambda f: os.path.join(self.partials_backup_path, f), backup_file))
        
        # merge them in a single dataframe and then save them in a sngle file called tmp_0 or the final backup
        to_concat = [pandas.read_csv(f, header = 0) for f in backup_file]
        if to_concat:
            df = pandas.concat([pandas.read_csv(f, header = 0) for f in backup_file], ignore_index=True)
            
            for file in backup_file:
                os.remove(file)
                
            df.to_csv(results_file_name, index=False)

        return results_file_name

    def __process_task_trainKF(self, 
                               data_set:np.ndarray, 
                               hyperparameters:list, 
                               hyperparameters_name:list, 
                               k_folds:int = 1, 
                               backup:str = None):
        '''
        Train the model with the given hyperparameters and the number of folds

        Parameters
        ----------
        data_set: np.ndarray 
            Dataset used for the K-Fold cross validation
        hyperparameters: dict
            Dict of hyperparameters' configurations to be used for validation
        hyperparameters_name: 
            List of hyperparameters' names
        k_folds: int
            number of folds to be used in the cross validation
        backup: str
            Backup file to be used to write the results

        Returns
        -------
        return: -

        '''
        # metric is a default param
        metrics_name = [m.__name__ for m in self.default_values['metrics']]
        # the header of the csv is written to the file
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
            
            # the hyperparameters are processed and used to start the model
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
            
            # if the execution fails the search does not stop
            try:
                # print some information at each new train
                print('pid:', os.getpid(), ' started new kfold' , index_con + 1, '/', len(hyperparameters))
                stats = ModelSelection.kf_train(nn, data_set, k_folds, grid_val['metrics'], args_train)
                
                # save the train results at every iteration
                list_to_write =(list(configuration) + 
                                [stats] + 
                                [x for x in stats['mean_metrics']] + 
                                [x for x in stats['variance_metrics']] + 
                                [stats['mean_best_validation_training_error']])
                writer.writerow(list_to_write)
            #  of the execution fails the configuration is spared regardless
            except Exception:
                writer.writerow(list(configuration) + [None, None, None, None, None]) 
            back_up.flush()

        back_up.close()

    def grid_searchKF(self, 
                      data_set:np.ndarray, 
                      hyperparameters:dict = {}, 
                      k_folds:int = 2, 
                      n_proc:int = 1, 
                      recovery:bool = False, 
                      constraints:dict = {}):
        '''
        Implementation of a completely configurable grid search

        Parameters
        ----------
        data_set: np.ndarray
            Training + validation to be used in the grid search
        hyperparameters: dict
            Dictionary with the hyperparameters to be tested
            
                eg: {'hyp_param_1':[val_1, val_2, val_3, ...], 
                        'hyp_param_2':[val_1, val_2, ...],         
                        'hyp_param_3':[val_1, ... ], ... }
                        
        k_folds: int
            Number of folds to be used in the cross validation
        n_proc: int
            Number of processes to be used in the grid search
        recovery: bool
            If to recover a previous computation interrupted before the finish
        constraints: dict
            Dictionary with the constrains to discard useless configuration
            
                eg: {'hyp_param_1': (fun_1, 
                                     ['hyp_param_2', 'hyp_param_3', 'hyp_param_4', ...], 
                                     ['hyp_param_5', 'hyp_param_6', ...]),
                     'hyp_param_7': (fun_2,
                                     ['hyp_param_8', 'hyp_param_9', ...],
                                     [...]), ... }
                            
                fun is a function applied to the values of 'hyp_param_1' and if the result is True then 
                the hyperparameters inside the first list are set to default values, if False the parameters 
                inside the second list are set to default values
        
        Returns
        -------
        return: -
        '''
        # the input hyperparameters grid is processed
        hyperparameters = dict(sorted(hyperparameters.items()))
        configurations, names = self.__get_configurations(hyperparameters, constraints, recovery)
        print('tot conf to do:', len(configurations))
        if n_proc == 1: # sequential execution if process is 1
            self.__process_task_trainKF(data_set, configurations, names, k_folds, self.backup)
            return
        
        # creation of useful variables to start the processes
        remainder = len(configurations) % n_proc
        single_conf_size = int(len(configurations) / n_proc)
        start = end = 0
        j = 0
        proc_pool = []
        partial_data_dir = Path(self.partials_backup_path).absolute()

        # creation of the backup folder
        if not os.path.exists(partial_data_dir):
            os.makedirs(partial_data_dir)
            
        # distribute equally the workload among the processes
        for i in range(n_proc): 
            # to give some time to the machine (and not make orrible print at the start)
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
           
        # join all the terminated processes
        for process in proc_pool: 
            process.join()

        self.__merge_csv_file(self.backup)