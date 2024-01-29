import numpy
import math

'''The collections of all implemented error functions and related methods'''

def mean_euclidean_error(outputs:numpy.ndarray, targets:numpy.ndarray):
    '''
    Calculates the Mean Euclidean Error for a given learning set of patterns
    
    param outputs: the predicted NN's outputs
    param targets: the actual targets

    return: the MEE value    
    '''

    sum = 0
    for diff in (outputs-targets):
        sum += math.sqrt(numpy.sum(diff**2))

    return sum/len(outputs)

def mean_squared_error(outputs:numpy.ndarray, targets:numpy.ndarray):
    '''
    Calculates the Mean Squared Error for a given learning set of patterns
    
    param outputs: the predicted NN's outputs
    param targets: the actual targets
    param root: if True returns MSE, if False returns RMSE

    return: the MSE value (or RMSE value)
    '''

    error = numpy.mean(numpy.sum((targets-outputs)**2, axis=1))
    if numpy.isnan(error):
        print("targets:", targets)
        print("outs:", outputs)

    return error

def accuracy(outputs:numpy.ndarray, targets:numpy.ndarray):
    '''
    Calculates the accuracy for a given learning set of patterns, computed as:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    param outputs: the predicted NN's outputs
    param targets: the actual targets
    param root: if True returns MSE, if False returns RMSE

    return: the MSE value (or RMSE value)
    '''

    error = numpy.mean(numpy.sum((targets-outputs)**2, axis=1))
    if numpy.isnan(error):
        print("targets:", targets)
        print("outs:", outputs)

    return error

if __name__ == '__main__':
    outputs = numpy.array([[3.2],[3.3],[3.3],[3.4],[3.5],[3.54]])
    targets = numpy.array([[5.1212],[5.1212],[5.1212],[5.121],[5.12121],[5.12]])

    print(mean_euclidean_error(outputs, targets), numpy.mean(numpy.linalg.norm(outputs-targets, axis = 1)))
