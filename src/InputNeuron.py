
from ActivationFunctions import ActivationFunctions

from ABCNeuron import ABCNeuron

class InputNeuron(ABCNeuron):
    '''
    Implementation of an input neuron composing the NN
    
    Attributes
    ----------
    index : int
        the index of the neuron in the NN
    type : str
        the type of the neuron
    successors : list of neurons
        list of neurons receiving this neuron's outputs
    n_successors: int
        number of units linked as successors to this neuron
    output_list : list of float
        list of the previous outputs of the neuron (instance variable exploited to store outputs for training scope)
    last_predict : float
        output of the neuron (instance variable exploited for predictions out of training)
    '''

    def __init__(self, index:int):
        '''
        Neuron initialisation

        :param index: the index of the neuron in the NN
        :return: -
        '''
        self.index = index
        self.type = 'input'
        self.successors = [] # list of neurons receiving this neuron's outputs
        self.n_successors = 0
        
        self.output_list = [] # creates the output list (instance variable exploited to store outputs for training scope)
        self.last_predict = 0.0 # output of the neuron (instance variable exploited for predictions out of training)
        
    def forward(self, input:float, training:bool):
        '''
        Calculates the Neuron's output on the inputs incoming from the other units, adding the output in the output_list
        
        :param input: Neuron's input vector
        :param training: flag which determines the neuron behaviour in storing data for training
        :return: the Neuron's output
        '''
        output_value = input
        if training:
            self.output_list.append(output_value)
        else:
            self.last_predict = output_value
    
    def add_successor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's successors and
        update the predecessors' list of the successor neuron with the current neuron
        
        :param neuron: the Neuron to add to the list of successors
        :return: -
        '''
        self.successors.append(neuron)
        self.n_successors += 1
        neuron.add_predecessor(self)
    
    def extend_successors(self, neurons:list):
        '''
        Extends the list of the Neuron's successors and
        update the predecessors' list of the successors neurons with the current neuron
        
        :param neurons: the list of Neurons to add to the list of successors
        :return: -
        '''
        self.successors.extend(neurons)
        self.n_successors += len(neurons)
        for successor in neurons:
            successor.add_predecessor(self)

    def reset_neuron_history(self):
        '''
        Resets the history of the neuron by clearing the list of previous outputs
        
        :return: -
        '''
        self.output_list = []

    
            