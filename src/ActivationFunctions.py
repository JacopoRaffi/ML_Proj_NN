import math

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
        The sigmoidal logistic function
        
        :param input: the input of the function
        :param alpha: the slope parameter of the sigmoid function
        :return: the results of the sigmoid function
        '''

        return 1/(1 + math.exp(-(input*alpha)))



    """
    Adds two numbers and returns the result.

    This add two real numbers and return a real result. You will want to
    use this function in any place you would usually use the ``+`` operator
    but requires a functional equivalent.

    :param a: The first number to add
    :param b: The second number to add
    :type a: int
    :type b: int
    :return: The result of the addition
    :rtype: int

    :Example:

    >>> add(1, 1)
    2
    >>> add(2.1, 3.4)  # all int compatible types work
    5.5

    .. seealso:: sub(), div(), mul()
    .. warnings:: This is a completly useless function. Use it only in a 
            tutorial unless you want to look like a fool.
    .. note:: You may want to use a lambda function instead of this.
    .. todo:: Delete this function. Then masturbate with olive oil.
    """