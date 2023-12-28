from abc import ABC
import numpy

class ABCNeuron(ABC):

    def __str__(self):
        '''
        Return a string that describe the internal state of the neuron
        
        :return: the description of the internal rapresentation
        '''
        attributes = ", ".join(f"{attr}={getattr(self, attr)}" for attr in vars(self))
        return f"{self.__class__.__name__}({attributes})"    
    
    def forward(self, input):
        pass