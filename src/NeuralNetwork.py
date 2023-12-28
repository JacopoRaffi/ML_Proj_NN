
class NeuralNetwork:
    '''
    Implementation of the neural network that includes neurons and all method to use and train the NN
    
    Attributes
    ----------
    

    '''

    
    def __construct_from_dict(self, topology):
        return

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
                 'F': ['output', 'linear', []],
                 'G': ['output', 'linear', []]}

            
        :return: -
        '''

        

        self.neurons = [] # maybe ordered in topographical order

        self.__construct_from_dict(topology)
        return
    
    def __str__(self):
        '''
        Return a string that describe the internal state of the neuron
        
        :return: the description of the internal rapresentation
        '''
        attributes = ", ".join(f"{attr}={getattr(self, attr)}" for attr in vars(self))
        return f"{self.__class__.__name__}({attributes})"
    
    
    def train(self, ):
        return
    