from ast import Raise
import numpy 
import math
import multiprocessing
import pandas
import itertools
from NeuralNetwork import NeuralNetwork
import ErrorFunctions
import csv
import json


class ModelSelection:
    '''
    Implementation of the model selection algorithm
    
    Attributes:
    -----------
    backup_file: file
        file to backup the model selection's state

    '''

    def __init__(self, cv_backup:str = None, topology_backup:str = None):
        if cv_backup is not None and topology_backup is not None:
            if cv_backup.endswith('.csv'):
                self.backup = open(cv_backup, 'w+') # create the backup file
            else:
                Raise(ValueError(' cv_backup extension must be .csv'))

            if topology_backup.endswith('.json'):
                self.topology_backup = open(topology_backup, 'w+')
            else:
                Raise(ValueError('topology_backup extension must be .json'))
        else:
            Raise(ValueError('Backup file missing'))


    def __train_modelKF(self, data_set:numpy.ndarray, hyperparameters:list, hyperparameters_name:list, 
                        k_folds:int = 1, topology:dict = {}, topology_name:str = 'standard', lock:multiprocessing.Lock = None, backup=None):
        '''
        Train the model with the given hyperparameters and the number of folds

        param dataset: dataset to be used for K-Fold cross validation
        param hyperparameters: dict of hyperparameters' configurations to be used for validation
        param k_folds: number of folds to be used in the cross validation

        '''

        lambda_tikhonov = 0.0
        alpha_momentum = 0.0
        learning_rate = 0.1
        batch_size = 1
        max_epochs = 100
        error_decrease_tolerance = 0.0001
        patience = 10
        min_epochs = 10
        range_min = -0.75
        range_max = 0.75
        fan_in = True
        random_state = 42
        metrics = [ErrorFunctions.mean_squared_error, ]
        writer = csv.writer(backup)
        print("HYPERPARAMETERS: ", hyperparameters_name)

        # for every configuration create a new clean model and train it
        for configuration in hyperparameters:
            for index, hyper_param in enumerate(configuration):
                if hyperparameters_name[index] == 'lambda_tikhonov':
                    lambda_tikhonov = hyper_param

                elif hyperparameters_name[index] == 'alpha_momentum':
                    alpha_momentum = hyper_param

                elif hyperparameters_name[index] == 'learning_rate':
                    learning_rate = hyper_param

                elif hyperparameters_name[index] == 'batch_size':
                    batch_size = hyper_param
                
                elif hyperparameters_name[index] == 'max_epochs':
                    max_epochs = hyper_param

                elif hyperparameters_name[index] == 'error_decrease_tolerance':
                    error_decrease_tolerance = hyper_param

                elif hyperparameters_name[index] == 'patience':
                    patience = hyper_param

                elif hyperparameters_name[index] == 'min_epochs':
                    min_epochs = hyper_param
            
            # create a new model
            nn = NeuralNetwork(topology, range_min, range_max, fan_in, random_state)
            mean, var = nn.kf_train(data_set, k_folds, batch_size, max_epochs, error_decrease_tolerance, 
                                    patience, min_epochs, learning_rate, lambda_tikhonov, alpha_momentum, metrics)
            
            print("CONFIGURATION: ", configuration)
            print("MEAN: ", mean, "VAR: ", var)
            lock.acquire()
            writer.writerow(list(configuration) + [topology_name, mean, var])
            lock.release()


    def grid_searchHO(self, training_set:numpy.ndarray, validation_set:numpy.ndarray, n_proc:int = 1):
        pass

    def grid_searchKF(self, data_set:numpy.ndarray, hyperparameters:dict = {}, k_folds:int = 3, 
                      n_proc:int = 1, topology:dict = {}, topology_name:str = 'standard'):
        '''
        Implementation of the grid search algorithm

        param data_set: training set to be used in the grid search
        param hyperparameters: dictionary with the hyperparameters to be tested
        param k_folds: number of folds to be used in the cross validation
        param n_proc: number of processes to be used in the grid search

        return: 
        '''

        # controllare che in hyperparameters ci sia la chiave network

        configurations = []
        names = list(hyperparameters.keys())

        for hyper_param in hyperparameters:
            if hyper_param == 'lambda_tikhonov':
                configurations.append(hyperparameters[hyper_param])

            elif hyper_param == 'alpha_momentum':
                configurations.append(hyperparameters[hyper_param])

            elif hyper_param == 'learning_rate':
                configurations.append(hyperparameters[hyper_param])

            elif hyper_param == 'batch_size':
                configurations.append(hyperparameters[hyper_param])

            elif hyper_param == 'max_epochs':
                configurations.append(hyperparameters[hyper_param])

            elif hyper_param == 'error_decrease_tolerance':
                configurations.append(hyperparameters[hyper_param])

            elif hyper_param == 'patience':
                configurations.append(hyperparameters[hyper_param])

            elif hyper_param == 'min_epochs':
                configurations.append(hyperparameters[hyper_param])

        configurations = list(itertools.product(*configurations))

        remainder = len(configurations) % n_proc
        single_conf_size = int(len(configurations) / n_proc)
        start = end = 0
        proc_pool = []

        file_lock = multiprocessing.Lock()
        
        writer = csv.writer(self.backup)
        writer.writerow(names + ['topology', 'mean', 'var'])

        for i in range(n_proc): # distribute equally the workload among the processes
            start = end
            if remainder > 0:
                end += single_conf_size + 1
            else:
                end += single_conf_size
            
            process = multiprocessing.Process(target=self.__train_modelKF, args=(data_set, configurations[start:end], 
                                                                                 names, k_folds, topology, topology_name, file_lock, self.backup))
            proc_pool.append(process)
            process.start()
            
            remainder -= 1
           
        for process in proc_pool: # join all the terminated processes
            process.join()

        
        # TODO: estrai la(o le) migliori configurazioni di ipermarametri 

    #TODO: grid search con lista di topologie

    def restore_backup(self, backup_file:str):
        '''
        Restore model selection's state from a backup file (JSON format)

        '''
        pass

