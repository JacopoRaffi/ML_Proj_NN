import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append(os.path.abspath('../src/'))
from ModelSelection import *
from ActivationFunctions import *
from NeuralNetwork import *
from MyUtils import *


tr_norm_df = pd.read_csv('../data/divided_std_train_0_8.csv')
test_norm_df = pd.read_csv('../data/divided_std_test_0_2.csv')

training_len = len(tr_norm_df)
test_len = len(test_norm_df)

INPUT = 10
OUTPUT = 3

hidden_len = 32
hidden_fun = 'sigmoid'
output_fun = 'identity'
sigmoid_l1 = create_stratified_topology([INPUT,hidden_len,OUTPUT], 
                                      [[None,[]]]*INPUT + [[hidden_fun, [1]]]*hidden_len + [[output_fun, []]]*OUTPUT)


MS = ModelSelection('../data/gs_data/1l_sigmoid_5_fg_gauss.csv')
MS.default_values['metrics'] = [ErrorFunctions.mean_euclidean_error, ErrorFunctions.mean_squared_error]
hyperparam_grid = {
    'lambda_tikhonov':[0.000000001, 0.00000001, 0.0000001],
    'batch_size':[6, 8, 11],
    'min_epochs': [150],
    'max_epochs':[500],
    
    'learning_rate':[0.09],
    'lr_decay_tau':[145, 165, 185, 200],
    'alpha_momentum':[0.85, 0.92, 0.95],
    
    'error_increase_tolerance':[0.000001],
    'patience':[5],
    'topology': [str(sigmoid_l1)],
    
    'adamax' : [False],
}
MS.grid_searchKF(tr_norm_df.values, hyperparam_grid,  3, 6, False, {})