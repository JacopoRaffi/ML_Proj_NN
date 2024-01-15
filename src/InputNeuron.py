
#from ActivationFunctions import ActivationFunctions
import ActivationFunctions
import numpy as np
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
    last_predict : float
        output of the neuron (instance variable exploited for predictions out of training)
    '''

    def __init__(self, index:int):
        '''
        Neuron initialisation

        param index: the index of the neuron in the NN

        return: -
        '''
        self.index = index
        self.type = 'input'
        self.successors = [] # list of neurons receiving this neuron's outputs
        self.n_successors = 0
        
        self.last_predict = 0.0 # output of the neuron (instance variable exploited for predictions out of training)
        # just to semplify stats collection
        self.delta_error = 0 
        self.old_weight_update = 0
        self.w = np.array([0])
        
    def forward(self, input:float):
        '''
        Calculates the Neuron's output on the inputs incoming from the other units, adding the output in the output_list
        
        param input: Neuron's input vector

        return: the Neuron's output
        '''   
        self.last_predict = input
        #print(self.index, self.last_predict)
    
    def add_successor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's successors and
        update the predecessors' list of the successor neuron with the current neuron
        
        param neuron: the Neuron to add to the list of successors

        return: -
        '''
        self.successors.append(neuron)
        self.n_successors += 1
        neuron.add_predecessor(self)
    
    def extend_successors(self, neurons:list):
        '''
        Extends the list of the Neuron's successors and
        update the predecessors' list of the successors neurons with the current neuron
        
        param neurons: the list of Neurons to add to the list of successors
        
        return: -
        '''
        self.successors.extend(neurons)
        self.n_successors += len(neurons)
        for successor in neurons:
            successor.add_predecessor(self)

    
            