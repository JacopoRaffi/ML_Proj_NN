from ABCNeuron import ABCNeuron
from InputNeuron import InputNeuron
from HiddenNeuron import HiddenNeuron
from OutputNeuron import OutputNeuron
from ActivationFunctions import ActivationFunctions


class NeuralNetwork:
    #TODO: implementare il metodo __construct_from_dict e dopo il metodo __topological_sort
    '''
    Implementation of the neural network that includes neurons and all method to use and train the NN
    
    Attributes
    ----------
    

    '''

    def __get_function_from_string(self, name:str):
        '''
        Map the function name to the corresponding callable variable
        
        :param name: the name of the function to map
        :return: the function corrisponding to the name as a callable
        '''
        if name == 'identity':
            fun = ActivationFunctions.identity

        elif name == 'sigmoid':
            fun = ActivationFunctions.sigmoid

        elif name == 'tanh':
            fun = ActivationFunctions.tanh

        elif name == 'softplus':
            fun = ActivationFunctions.softplus

        elif name == 'gaussian':
            fun = ActivationFunctions.gaussian
        
        else:
            raise ValueError(f"Activation function {name} not found")

        return fun

    def __construct_from_dict(self, topology:dict, rand_range_min:float = -1, rand_range_max:float = 1, fan_in:bool = True, *args):
        '''
        Builds a Neural Network of ABCNeuron's objects from the topology
        
        :param topology: the graph structure described by a dictionary (see __init__ comments)
        :param rand_range_min: minimum value for random weights initialisation range
        :param rand_range_max: maximum value for random weights initialisation range
        :param fan_in: if the weights'initialisation should also consider the Neuron's fan-in
        :return: the dictionary that represents the Neural Network of ABCNeuron's objects
        '''
        graph = {}
        return graph
    
    def __topological_sort(self, topology:dict, rand_range_min:float = -1, rand_range_max:float = 1, fan_in:bool = True, *args):
        '''
        Sort on a topological order the neurons of the Neural Network
        
        :param topology: the graph structure described by a dictionary (see __init__ comments)
        :param rand_range_min: minimum value for random weights initialisation range
        :param rand_range_max: maximum value for random weights initialisation range
        :param fan_in: if the weights'initialisation should also consider the Neuron's fan-in
        :return: the list of neurons sorted in topological order
        '''
        graph = self.__construct_from_dict(topology, rand_range_min, rand_range_max, fan_in, *args)
        neurons = []
        return neurons

    def __init__(self, topology:dict = {}):
        '''
        Neural Network inizialization
        
        :param topology: the graph structure is described by a dictionary that has a key for each unit in the network, 
            and for each key contains a list of unit type (input, hidden, output), activation function, 
            and list of nodes where an outgoing arc terminates.
            
            eg: {'A': ['input', 'None', ['C', 'D', 'E']], 
                 'B': ['input', 'None', ['C', 'D', 'E']],
                 'C': ['hidden', 'sigmoid', ['F', 'G']],
                 'D': ['hidden', 'sigmoid', ['F', 'G']],
                 'E': ['hidden', 'sigmoid', ['F', 'G']],
                 'F': ['output', 'identity', []],
                 'G': ['output', 'identity', []]}

            
        :return: -
        '''

        
        self.neurons = self.__topological_sort(topology) # list of neurons sorted in topological order
    
    def __str__(self):
        '''
        Return a string that describe the internal state of the neural network
        
        :return: the description of the internal rapresentation
        '''
        attributes = ", ".join(f"{attr}={getattr(self, attr)}" for attr in vars(self))
        return f"{self.__class__.__name__}({attributes})"
    
    
    def train(self, ):
        return
    