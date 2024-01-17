from ast import Raise
from email import header
from operator import index
from matplotlib.pylab import f
import numpy 
import multiprocessing
import pandas
import itertools
from NeuralNetwork import NeuralNetwork
import ErrorFunctions
import csv
import os

#TODO implementare il ripristino (il main ricomputerà le configurazioni eliminando quelle già eseguite e salvate nei backup.)
#                                   le restanti saranno ridistribuite trai processi)
#TODO fare in modo che i backup_file si salvino periodicamente su disco e non solo alla fine del processo
#TODO controllare che la merge funzioni correttamente
#TODO ripristinare lo stato dei backup
#TODO (opzionale) ottimizzare


class ModelSelection:
    '''
    Implementation of the model selection algorithm
    
    Attributes:
    -----------
    backup_file: file
        file to backup the model selection's state

    '''


    def __init__(self, cv_backup:str = None, topology_backup:str = None):
        '''
        Constructor of the class
        
        param cv_backup: file to backup the model selection's state
        
        return: -
        ''' 

        self

        if cv_backup is not None and topology_backup is not None:
            if cv_backup.endswith('.csv'):
                self.backup = cv_backup
            else:
                Raise(ValueError(' cv_backup extension must be .csv'))

            if topology_backup.endswith('.json'):
                pass
                #self.topology_backup = open(topology_backup, 'w+')
            else:
                Raise(ValueError('topology_backup extension must be .json'))
        else:
            Raise(ValueError('Backup file missing'))


    def __train_modelKF(self, data_set:numpy.ndarray, hyperparameters:list, hyperparameters_name:list, 
                        k_folds:int = 1, topology:dict = {}, topology_name:str = 'standard', backup:str = None):
        '''
        Train the model with the given hyperparameters and the number of folds

        param dataset: dataset to be used for K-Fold cross validation
        param hyperparameters: dict of hyperparameters' configurations to be used for validation
        param hyperparameters_name: list of hyperparameters' names
        param k_folds: number of folds to be used in the cross validation
        param topology: topology of the neural network
        param topology_name: name of the topology
        param backup: backup file to be used to write the results

        return: -

        '''

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
        metrics = [ErrorFunctions.mean_squared_error, ]'''
        
        values_to_use =  {
        'range_min' : -0.75,
        'range_max' : 0.75,
        'fan_in' : True,
        'random_state' : None,

        'lambda_tikhonov' : 0.0,
        'alpha_momentum' : 0.5,
        'learning_rate' : 0.1,
        'batch_size' : 1,
        'max_epochs' : 100,
        'error_decrease_tolerance' : 0.0001,
        'patience' : 10,
        'min_epochs' : 0,
        'metrics':[ErrorFunctions.mean_squared_error, ],

        'collect_data':True, 
        'collect_data_batch':False, 
        'verbose':False
        }
        inzialization_arg_names = ['range_min', 'range_max', 'fan_in', 'random_state']
        train_arg_names = ['batch_size', 'max_epochs', 'error_decrease_tolerance', 'patience', 'min_epochs', 
                       'learning_rate', 'lambda_tikhonov', 'alpha_momentum', 'metrics', 'collect_data', 
                        'collect_data_batch', 'verbose']
        

        if os.path.isfile(backup): # if file exists i only add more data
            back_up = open(backup, 'a')
            writer = csv.writer(back_up)
        else:
            back_up = open(backup, 'w+') # if file doesn't exist i create it adding the header
            writer = csv.writer(back_up)
            writer.writerow(hyperparameters_name + ['topology','validation_error_mean', 'validation_error_variance'])

        # for every configuration create a new clean model and train it
        for configuration in hyperparameters:
            '''
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
                    min_epochs = hyper_param'''
            
            
            for i, key in enumerate(hyperparameters_name): values_to_use[key] = configuration[i]

            # create a new model
            args_init = [values_to_use[key] for key in inzialization_arg_names]
            nn = NeuralNetwork(topology, *args_init)
            # train the model
            args_train = [values_to_use[key] for key in train_arg_names]
            print("\nTraining a new model : ", args_train)
            mean, var = nn.kf_train(data_set, k_folds, *args_train)
            
            writer.writerow(list(configuration) + [topology_name, mean, var])
            back_up.flush()
        
        back_up.close()

    def __train_modelHO(self, training_set:numpy.ndarray, validation_set:numpy.ndarray, hyperparameters:list, 
                        hyperparameters_name:list, topology:dict = {}, topology_name:str = 'standard', 
                        backup:str = None):
        
        '''
        Train the model with the given configuration of hyperparameters

        param training_set: training set to be used for hold out validation
        param validation_set: validation set to be used for hold out validation
        param hyperparameters: list of hyperparameters' configurations to be used for validation
        param hyperparameters_name: list of hyperparameters' names
        param topology: topology of the neural network
        param topology_name: name of the topology
        param lock: lock to be used to write on the backup file
        param: backup: backup file

        return: -
        '''
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
        metrics = [ErrorFunctions.mean_squared_error, ]'''
        
        values_to_use = {
        'range_min' : -0.75,
        'range_max' : 0.75,
        'fan_in' : True,
        'random_state' : None,

        'lambda_tikhonov' : 0.0,
        'alpha_momentum' : 0.5,
        'learning_rate' : 0.1,
        'batch_size' : 1,
        'max_epochs' : 100,
        'error_decrease_tolerance' : 0.0001,
        'patience' : 10,
        'min_epochs' : 0,
        'metrics':[ErrorFunctions.mean_squared_error, ],

        'collect_data':True, 
        'collect_data_batch':False, 
        'verbose':False
        }
        inzialization_arg_names = ['range_min', 'range_max', 'fan_in', 'random_state']
        train_arg_names = ['batch_size', 'max_epochs', 'error_decrease_tolerance', 'patience', 'min_epochs', 
                       'learning_rate', 'lambda_tikhonov', 'alpha_momentum', 'metrics', 'collect_data', 
                        'collect_data_batch', 'verbose']

        if os.path.isfile(backup):
            back_up = open(backup, 'a')
            writer = csv.writer(back_up)
        else:
            back_up = open(backup, 'w+')
            writer = csv.writer(back_up)
            writer.writerow(hyperparameters_name + ['topology', 'validation_error_mean', 'validation_error_variance'])

        # for every configuration create a new clean model and train it
        for configuration in hyperparameters:
            '''for index, hyper_param in enumerate(configuration):
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
                    min_epochs = hyper_param'''
            
            for i, key in enumerate(hyperparameters_name): values_to_use[key] = configuration[i]

            # create a new model
            args_init = [values_to_use[key] for key in inzialization_arg_names]
            nn = NeuralNetwork(topology, *args_init)
            # train the model
            args_train = [values_to_use[key] for key in train_arg_names]
            

            stats = nn.ho_train(training_set, validation_set, *args_train)      
            writer.writerow(list(configuration) + [topology_name, stats['validation_mean_squared_error'][-1], 0])
            back_up.flush()

    def __get_configurations(self, hyperparameters:dict):
        '''
        Get all the possible configurations of the hyperparameters

        param hyperparameters: dictionary with the hyperparameters to be tested

        return: list of all the possible configurations and list of the hyperparameters' names
        '''

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

        return configurations, names

    def __get_best_hyperparameters(self, num:int = 1):
        '''
        Get the best hyperparameters' configuration(s)

        param num: number of best configurations to be returned

        return: dictionary with the best hyperparameters' configuration
        '''

        #TODO: prendere i migliori iperparametri

        pass

    def merge_csv_file(self, results_file_name:str, n_proc:int = 1):
        '''
        Merge the results of the processes in a single file

        param results_file_name: name of the file obtained after the merge
        param n_proc: number of processes

        return: -
        '''

        backup_file = [f'backup_{i}.csv' for i in range(n_proc)]
        df = pandas.concat([pandas.read_csv(f, header = 0) for f in backup_file], ignore_index=True)
        df.to_csv(results_file_name, index=False)

        for file in backup_file:
            os.remove(file)

    def grid_searchHO(self, training_set:numpy.ndarray, validation_set:numpy.ndarray, hyperparameters:dict, 
                      topology:dict, n_proc:int = 1, topology_name:str = 'standard'):
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
        #TODO: ottimizzare caso con singolo processo, chiama direttamente la funzione _train_modelHO
        configurations, names = self.__get_configurations(hyperparameters)

        remainder = len(configurations) % n_proc
        single_conf_size = int(len(configurations) / n_proc)
        start = end = 0
        proc_pool = []
        
        for i in range(n_proc): # distribute equally the workload among the processes
            start = end
            if remainder > 0:
                end += single_conf_size + 1
            else:
                end += single_conf_size
            
            process = multiprocessing.Process(target=self.__train_modelHO, args=(training_set, validation_set, configurations[start:end],
                                                                                 names, topology, topology_name,
                                                                                 f'backup_{i}.csv',))
            proc_pool.append(process)
            process.start()
            
            remainder -= 1
           
        for process in proc_pool: # join all the terminated processes
            process.join()

        self.__merge_csv_file(self.backup, n_proc)

    def grid_searchKF(self, data_set:numpy.ndarray, hyperparameters:dict = {}, k_folds:int = 3, 
                      topology:dict = {}, topology_name:str = 'standard', n_proc:int = 1):
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
        #TODO: ottimizzare caso con singolo processo, chiama direttamente la funzione _train_modelKF
        configurations, names = self.__get_configurations(hyperparameters)

        remainder = len(configurations) % n_proc
        single_conf_size = int(len(configurations) / n_proc)
        start = end = 0
        proc_pool = []

        file_lock = multiprocessing.Lock()
        
        #writer = csv.writer(self.backup)
        #writer.writerow(names + ['topology', 'mean', 'var'])

        for i in range(n_proc): # distribute equally the workload among the processes
            start = end
            if remainder > 0:
                end += single_conf_size + 1
            else:
                end += single_conf_size
            
            process = multiprocessing.Process(target=self.__train_modelKF, args=(data_set, configurations[start:end], 
                                                                                 names, k_folds, topology, topology_name, 
                                                                                 f'backup_{i}.csv',))
            proc_pool.append(process)
            process.start()
            
            remainder -= 1
           
        for process in proc_pool: # join all the terminated processes
            process.join()

        #final_results = open(self.backup, 'a+')
        #writer = csv.writer(final_results)

        self.__merge_csv_file(self.backup, n_proc)

    def restore_backup(self, backup_file:str):
        '''
        Restore model selection's state from a backup file (csv format)

        param backup_file: backup file to be used to restore the state
        
        return: -
        '''
        pass

