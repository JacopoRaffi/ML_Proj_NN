import math
import numpy

class ActivationFunctions:
    '''The collections of all implemented activation functions'''

    def identity(input):
        '''
        The identity function
        
        :param input: the input of the function
        :return: the results of the identity function
        '''
        return input
    
    def sigmoid(input, alpha):
        '''
        The sigmoidal logistic function
        
        :param input: the input of the function
        :param alpha: the slope parameter of the sigmoid function
        :return: the results of the sigmoid function
        '''

        return 1/(1 + math.exp(-(input*alpha)))
    
    def tanh(input, alpha):
        '''
        The hyperbolic tangent function
        
        :param input: the input of the function
        :param alpha: the slope parameter of the hyperbolic tangent function, the more alpha increases, the more skewed the function becomes
        :return: the results of the sigmoid function
        '''

        return numpy.tanh((input*alpha)/2)

    def softplus(input):
        '''
        The softplus function 
        
        :param input: the input of the function
        :return: the results of the sigmoid function
        '''

        # a safe softplus for large input
        return math.log1p(math.exp(-abs(input))) + max(input, 0)
    
    def gaussian(input, alpha):
        '''
        The gaussian function
        
        :param input: the input of the function
        :param alpha: the slope parameter of the gaussian function
        :return: the results of the sigmoid function
        '''

        return math.exp(-alpha*(input**2))