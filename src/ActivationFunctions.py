import math
import numpy

class ActivationFunctions:
    '''The collections of all implemented activation functions and related methods'''
    
    def derivative(fun:callable, input:float, *args):
        '''
        Method which calculates the first order derivative of the given function for the given input
        
        :param fun: a differentiable function
        :param input: the input on which the derivative is calculated
        :return: fun's derivative calculated on the input
        '''
        
        x = numpy.array([input-0.0001, input, input+0.0001])
        y = numpy.array([fun(x_elem, *args) for x_elem in x])

        return numpy.gradient(y, x, edge_order=2)[1]

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