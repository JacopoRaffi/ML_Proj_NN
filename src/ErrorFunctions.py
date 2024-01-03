import numpy
#from sklearn.metrics import mean_squared_error
import math

class ErrorFunctions:
    '''The collections of all implemented error functions and related methods'''
    
    def mean_euclidean_error(outputs:numpy.ndarray, targets:numpy.ndarray):
        '''
        Calculates the Mean Euclidean Error for a given learning set of patterns
        
        :param outputs: the predicted NN's outputs
        :param targets: the actual targets

        :return: the Mean Euclidean Error
        '''
        #norm_vector = numpy.linalg.norm(outputs-targets, axis = 1)
        sum = numpy.sum(numpy.linalg.norm(outputs-targets, axis = 1))
        output_length = len(outputs)

        return sum/output_length
    
    def mean_squared_error(outputs:numpy.ndarray, targets:numpy.ndarray):
        '''
        Calculates the Mean Squared Error for a given learning set of patterns
        
        :param outputs: the predicted NN's outputs
        :param targets: the actual targets

        :return: the Mean Squared Error
        '''
        n_samples = outputs.shape[0]
        features = outputs.shape[1]
        error = (numpy.sum(numpy.sum((targets-outputs)**2, axis=1))) / (n_samples*features)
        
        return error
    


if __name__== '__main__':
    out = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tar = numpy.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]])

    #print("MSE: ", ErrorFunctions.mean_squared_error(out, tar), " ", mean_squared_error(tar, out), " ", ErrorFunctions.mean_squared_error(out, tar) == mean_squared_error(out, tar))
    #print("MEE: ", ErrorFunctions.mean_euclidean_error(out, tar))
