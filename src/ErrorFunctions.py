import numpy
import math

'''The collections of all implemented error functions and related methods'''

def mean_euclidean_error(outputs:numpy.ndarray, targets:numpy.ndarray):
    '''
    Calculates the Mean Euclidean Error for a given learning set of patterns
    
    :param outputs: the predicted NN's outputs
    :param targets: the actual targets

    :return: the MEE value
    '''
    
    mean = numpy.mean(numpy.linalg.norm(outputs-targets, axis = 1))

    return mean

def mean_squared_error(outputs:numpy.ndarray, targets:numpy.ndarray, root:bool = False):
    '''
    Calculates the Mean Squared Error for a given learning set of patterns
    
    :param outputs: the predicted NN's outputs
    :param targets: the actual targets
    :param root: if True returns MSE, if False returns RMSE

    :return: the MSE value (or RMSE value)
    '''

    error = numpy.mean(numpy.sum((targets-outputs)**2, axis=1))
    
    if root:
        error = math.sqrt(error)

    return error