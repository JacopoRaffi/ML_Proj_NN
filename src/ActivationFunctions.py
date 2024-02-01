import math
import numpy


'''
The collections of all implemented activation functions and related methods
'''

def derivative(fun:callable, input:float, *args):
    '''
    Method which calculates the first order derivative of the given function for the given input
    
    Parameters
    ----------
    fun: callable
        The differentiable function whose derivative is computed
    input: float
        the input on which the derivative is calculated

    Returns
    -------
    return: float
        fun's derivative calculated on the input
    '''

    # to correct compute the relud derivate around and in 0
    if fun == ReLU:
        if input > 0: return 1 # constant 1
        else: return 0 # costant 0
    
    # to speed up the identity derivate that is always 1
    elif fun == identity:
        return 1

    # we compute the derivate of the others functions with numpy.gradient
    else:
        x = numpy.array([input-0.001, input, input+0.001])
        y = numpy.array([fun(x_elem, *args) for x_elem in x])

    return numpy.gradient(y, x, edge_order=2)[1]

def identity(input:float, *args):
    '''
    The identity function
    
    Parameters
    ----------
    input: float
        the input of the function
    
    Returns
    -------
    return: float
        the results of the identity function
    '''
    return input

def sigmoid(input, slope, *args):
    '''
    The sigmoidal logistic function
    
    Parameters
    ----------
    input: float
        the input of the function
    slope: float
        the slope parameter of the sigmoid function
    
    Returns
    -------
    return: float
        the results of the sigmoid function
    '''
    scarto = 5
    if input > scarto:
        return 1
    elif input < -scarto:
        return 0
    else: 
        return 1/(1 + math.exp(-(input*slope)))

def tanh(input, slope, *args):
    '''
    The hyperbolic tangent function
    
    Parameters
    ----------
    input: float
        the input of the function
    slope: float
        the slope parameter of the hyperbolic tangent function, the more alpha increases, the more skewed the function becomes
    
    Returns
    -------
    return: float
        the results of the sigmoid function
    '''

    return numpy.tanh((input*slope)/2)

def softplus(input, *args):
    '''
    The softplus function 
    
    Parameters
    ----------
    input: float
        the input of the function

    Returns
    -------
    return: float
        the results of the sigmoid function
    '''

    # a safe softplus for large input
    return math.log1p(math.exp(-abs(input))) + max(input, 0)

def ReLU(input, *args):
    '''
    The ReLU function
    
    Parameters
    ----------
    input: float
        the input of the function

    Returns
    -------
    return: float
        the results of the sigmoid function
    '''

    return max(0.0, input)