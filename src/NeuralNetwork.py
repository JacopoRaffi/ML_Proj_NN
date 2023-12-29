from ABCNeuron import ABCNeuron
from InputNeuron import InputNeuron
from HiddenNeuron import HiddenNeuron
from OutputNeuron import OutputNeuron
from ActivationFunctions import ActivationFunctions


class NeuralNetwork:
    '''
    Implementation of the neural network that includes neurons and all method to use and train the NN
    
    Attributes
    ----------
    neurons: list of ABCNeuron
        the list of neurons, sorted in topological order, composing the NN
    

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

    def __construct_from_dict(self, topology:dict, rand_range_min:float = -1, rand_range_max:float = 1, fan_in:bool = True):
        '''
        Builds a Neural Network of ABCNeuron's objects from the topology
        
        :param topology: the graph structure described by a dictionary (see __init__ comments)
        :param rand_range_min: minimum value for random weights initialisation range
        :param rand_range_max: maximum value for random weights initialisation range
        :param fan_in: if the weights'initialisation should also consider the Neuron's fan-in
        :return: the list of Neurons that compose the Neural Network
        '''
        units = []
        unit_type = ''
        unit_activation_function = ''
        unit_activation_function_args = []
        unit_successors = []
        unit_index = 0
        
        # All Neurons are initialised without synapses (successors/predecessors dependencies)
        for node in topology:
            unit_index = int(node)
            unit_type = topology[node][0]
            unit_activation_function = topology[node][1]
            unit_activation_function_args = [float(a) for a in topology[node][2]]
            
            if unit_type == 'input':
                units.append(InputNeuron(unit_index))
                
            elif unit_type == 'hidden':
                units.append(HiddenNeuron(unit_index, rand_range_min, rand_range_max, fan_in, self.__get_function_from_string(unit_activation_function), unit_activation_function_args))
                
            elif unit_type == 'output': # Fan-in is fixed as False for output units so to prevent Delta (Backpropagation Error Signal) to be a low value 
                units.append(OutputNeuron(unit_index, rand_range_min, rand_range_max, False, self.__get_function_from_string(unit_activation_function), unit_activation_function_args))
            
        # All Neurons' dependecies of successors and predecessors are filled inside the objects
        for node in topology:
            unit_type = topology[node][0]
            
            if unit_type != 'output': # Output units have no successors
                unit_successors = [units[int(u)] for u in topology[node][3]]
                units[int(node)].extend_successors(unit_successors)
        
        # All Neurons weights vectors are initialised
        for neuron in units:
            if neuron.type != 'input':
                neuron.initialise_weights(rand_range_min, rand_range_max, fan_in)

        return units
    
    def __topological_sort_util(self, index:int, visited:list, ordered:list):
        '''
        Recursive function that builds the topological(inverse) order updating ordered list 
        
        :param: index: the index of the neuron to visit
        :param: visited: the list of visited neurons
        :param: ordered: the list of ordered neurons (inverse topological order)
        :return: -
        '''
        visited[index] = True
        
        if self.neurons[index].type != 'output':
            for succ in self.neurons[index].successors:
                if not visited[succ.index]:
                    self.__topological_sort_util(succ.index, visited, ordered)
        
        ordered.append(self.neurons[index])
    
    def __topological_sort(self):
        '''
        Sort on a topological order the neurons of the Neural Network
        
        :param: -
        :return: -
        '''
        n_neurons = len(self.neurons)
        visited = [False]*n_neurons
        ordered = []
        
        for i in range(n_neurons):
            if not visited[i]:
                self.__topological_sort_util(i, visited, ordered)
        
        self.neurons = ordered[::-1]
                
    def __init__(self, topology:dict = {}, rand_range_min:float = -1, rand_range_max:float = 1, fan_in:bool = True):
        '''
        Neural Network inizialization
        
        :param topology: the graph structure is described by a dictionary that has a key for each unit in the network, 
            and for each key contains a list of unit type (input, hidden, output), activation function, parameters of activation functions
            and list of nodes where an outgoing arc terminates.
            
            eg: {'0': ['input', 'None', [], ['2', '3', '4']], 
                 '1': ['input', 'None', [], ['2', '3', '4']],
                 '2': ['hidden', 'sigmoid', [fun_args...], ['5', '6']],
                 '3': ['hidden', 'sigmoid', [fun_args...], ['5', '6']],
                 '4': ['hidden', 'sigmoid', [fun_args...], ['5', '6']],
                 '5': ['output', 'identity', [fun_args...], []],
                 '6': ['output', 'identity', [fun_args...], []]}

        :param: rand_range_min: minimum value for random weights initialisation range
        :param: rand_range_max: maximum value for random weights initialisation range
        :param: fan_in: if the weights'initialisation should also consider the Neuron's fan-in
        :return: -
        '''

        self.neurons = self.__construct_from_dict(topology, rand_range_min, rand_range_max, fan_in)
        self.__topological_sort() # The NN keeps its neurons in topological order
    
    def __str__(self):
        '''
        Return a string that describe the internal state of the neural network
        
        :return: the description of the internal rapresentation
        '''
        attributes = ", ".join(f"{attr}={getattr(self, attr)}" for attr in vars(self))
        return f"{self.__class__.__name__}({attributes})"
    
    
    def train(self, ):
        return
    


if __name__ == '__main__':
    topology = {'0': ['input', 'None', [], ['2', '3', '4', '5']], 
                '1': ['input', 'None', [], ['2', '3', '4', '6']],
                '2': ['hidden', 'sigmoid', ['1'], ['5', '6', '4']],
                '3': ['hidden', 'sigmoid', ['1'], ['5', '6']],
                '4': ['hidden', 'sigmoid', ['1'], ['5', '6']],
                '5': ['output', 'identity', [], []],
                '6': ['output', 'identity', [], []]}
    
    nn = NeuralNetwork(topology)
    for neuron in nn.neurons:
        if neuron.type != 'input':
            print(neuron.index, " ", neuron.type, " ", neuron.w)