import math
import numpy

class ActivationFunctions:
    '''The collections of all implemented activation functions'''

    def identity(input, *args):
        '''
        The identity function
        
        :param input: the input of the function
        :param args[0]: ignored
        :return: the results of the identity function
        '''
        return input
    
    def sigmoid(input, *args):
        '''
        The sigmoidal logistic function
        
        :param input: the input of the function
        :param args[0]: the slope parameter of the sigmoid function
        :return: the results of the sigmoid function
        '''

        return 1/(1 + math.exp(-(input*args[0])))
    
    def tanh(input, *args):
        '''
        The hyperbolic tangent function
        
        :param input: the input of the function
        :param args[0]: the slope parameter of the hyperbolic tangent function, the more alpha increases, the more skewed the function becomes
        :return: the results of the sigmoid function
        '''

        return numpy.tanh((input*args[0])/2)

    def softplus(input, *args):
        '''
        The softplus function 
        
        :param input: the input of the function
        :param args[0]: ignored
        :return: the results of the sigmoid function
        '''

        # a safe softplus for large input
        return math.log1p(math.exp(-abs(input))) + max(input, 0)
    
    def gaussian(input, *args):
        '''
        The gaussian function
        
        :param input: the input of the function
        :param args[0]: the slope parameter of the gaussian function
        :return: the results of the sigmoid function
        '''

        return math.exp(-args[0]*(input**2))