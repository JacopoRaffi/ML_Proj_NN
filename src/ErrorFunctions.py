import numpy as np
import math
import sklearn.metrics as skm

'''The collections of all implemented error functions'''

def mean_euclidean_error(outputs:np.ndarray, targets:np.ndarray):
    '''
    Calculates the Mean Euclidean Error for a given learning set of patterns
    
    param outputs: the predicted NN's outputs
    param targets: the targhet values

    return: the MEE value    
    '''

    sum = 0
    for diff in (outputs-targets):
        sum += math.sqrt(np.sum(diff**2))

    return sum/len(outputs)

def mean_squared_error(outputs:np.ndarray, targets:np.ndarray):
    '''
    Calculates the Mean Squared Error for a given learning set of patterns
    
    param outputs: the predicted NN's outputs
    param targets: the targhet values

    return: the MSE value (or RMSE value)
    '''

    error = np.mean(np.sum((targets-outputs)**2, axis=1))
    if np.isnan(error):
        print("targets:", targets)
        print("outs:", outputs)

    return error

def accuracy(outputs:np.ndarray, targets:np.ndarray):
    '''
    Calculates the accuracy for a given learning set of patterns, computed as:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    param outputs: the predicted NN's outputs, boolean values or the probability of the positive class
    param targets: the targhet values, trictly boolean values

    return: the accuracy value
    '''
    N = targets.shape[0]
    if outputs.shape[1] > 1:
        raise Exception('only 1-dimension array are allowed')
    outputs = outputs.flatten()
    targets = targets.flatten()
    predictions = np.array([1 if x > 0.5 else 0 for x in outputs])
    
    ret = sum((targets) == predictions) / N
    return ret 
