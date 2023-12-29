
from ActivationFunctions import ActivationFunctions

from ABCNeuron import ABCNeuron

class InputNeuron(ABCNeuron):
    '''
    Implementation of an input neuron composing the NN
    
    Attributes
    ----------
    successors : list of neurons
        list of neurons receiving this neuron's outputs
    output_list : list of float
        list of the previous output of the neuron
    '''

    def __init__(self, index:int):
        '''
        Neuron initialisation

        :return: -
        '''
        self.index = index
        self.successors = [] # list of neurons receiving this neuron's outputs
        self.output_list = [] # creates the output list        
        
    def forward(self, input:float):
        '''
        Inizializate the input in the net
        
        :param input: Neuron's input value
        :return: the Neuron's output
        '''
        self.output_list.append(input)
        return input
    
    def add_successor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's successors
        
        :param neuron: the Neuron to add to the list of successors
        :return: -
        '''
        self.successors.append(neuron)
        neuron.add_predecessor(self)
    
    def extend_successors(self, neurons:list):
        '''
        Extends the list of the Neuron's successors
        
        :param neurons: the list of Neurons to add to the list of successors
        :return: -
        '''
        self.successors.extend(neurons)
        for successor in neurons:
            successor.add_predecessor(self)

    def reset_neuron_history(self):
        '''
        Resets the history of the neuron by clearing the list of previous outputs
        
        :return: -
        '''
        self.output_list = []

    
            