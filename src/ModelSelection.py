from ast import Raise
import numpy 
import math
import multiprocessing
import pandas
import itertools


class ModelSelection:
    '''
    Implementation of the model selection algorithm
    
    Attributes:
    -----------
    backup_file: file
        file to backup the model selection's state

    '''

    def __init__(self, file_backup:str = None):
        if file_backup is not None:
            if file_backup.endswith('.csv'):
                self.backup = open(file_backup, 'w+') # create the backup file
            else:
                Raise(ValueError('Extension must be .csv'))


    def train_modelKF(self, data_set:numpy.ndarray, hyperparameters:list, hyperparameters_name:list, k_folds:int = 1):
        '''
        Train the model with the given hyperparameters and the number of folds

        param dataset: dataset to be used for K-Fold cross validation
        param hyperparameters: dict of hyperparameters' configurations to be used for validation
        param k_folds: number of folds to be used in the cross validation

        '''
        
        pass
        # for every configuration create a new clean model and train it
        

    def grid_searchKF(self, data_set:numpy.ndarray, hyperparameters:dict = {}, k_folds:int = 3, n_proc:int = 1):
        '''
        Implementation of the grid search algorithm

        param data_set: training set to be used in the grid search
        param hyperparameters: dictionary with the hyperparameters to be tested
        param k_folds: number of folds to be used in the cross validation
        param n_proc: number of processes to be used in the grid search

        return: 
        '''

        configurations = []
        names = list(hyperparameters.keys())
        print(names)

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
        print(configurations)
        data = numpy.array([[1,2,3], [1,2,3]])
        
           


    def restore_backup(self, backup_file:str):
        '''
        Restore model selection's state from a backup file (JSON format)

        '''
        pass

