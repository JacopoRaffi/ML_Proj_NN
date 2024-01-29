import numpy as np
import math
import sklearn.metrics as skm

'''The collections of all implemented error functions and related methods'''

def mean_euclidean_error(outputs:np.ndarray, targets:np.ndarray):
    '''
    Calculates the Mean Euclidean Error for a given learning set of patterns
    
    param outputs: the predicted NN's outputs
    param targets: the actual targets

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
    param targets: the actual targets

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
    
    param outputs: the predicted NN's outputs, only 1-dimension array are allowed
    param targets: the actual targets

    return: the accuracy value
    '''
    if outputs.shape[1] > 1:
        raise Exception('only 1-dimension array are allowed')
    outputs = outputs.flatten()
    targets = targets.flatten()
    N = targets.shape[0]
    predictions = np.array([1 if x > 0.5 else 0 for x in outputs])
    
    ret = sum((targets) == predictions) / N
    return ret

def f1_score(outputs:np.ndarray, targets:np.ndarray):
    '''
    Calculates the accuracy for a given learning set of patterns, computed as:
        F1_SCORE = (2 * TP) / (2 * TP + FN + FP)
    
    param outputs: the predicted NN's outputs, only 1 dimension array are allowed
    param targets: the actual targets

    return: the accuracy value
    '''
    if outputs.shape[1] > 1:
        raise Exception('only 1-dimension array are allowed')
    outputs = outputs.flatten()
    targets = targets.flatten()
    predictions = np.array([1 if x > 0.5 else 0 for x in outputs])
    
    ret = skm.f1_score(targets, predictions, average='macro')
    return ret 

if __name__ == '__main__':
    outputs = np.array([[3.2],[3.3],[3.3],[3.4],[3.5],[3.54]])
    targets = np.array([[5.1212],[5.1212],[5.1212],[5.121],[5.12121],[5.12]])

    print(mean_euclidean_error(outputs, targets), np.mean(np.linalg.norm(outputs-targets, axis = 1)))
