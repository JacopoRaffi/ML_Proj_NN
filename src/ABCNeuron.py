from abc import ABC

class ABCNeuron(ABC):

    def __str__(self):
        '''
        Return a string that describe the internal state of the neuron
        
        :return: the description of the internal rapresentation
        '''
        attributes = ", ".join(f"{attr}={getattr(self, attr)}" for attr in vars(self))
        return f"{self.__class__.__name__}({attributes})"    
    
    def __repr__(self):
        attributes = ''
        for attr in vars(self):
            if attr == 'predecessors' or attr == 'successors':
                attributes += f"{attr}=["
                for i in getattr(self, attr):
                    attributes += str(i.index) + ', '
                attributes += '],\n\t'
            else:
                attributes += f"{attr}={getattr(self, attr)}" + ',\n\t' 


        return f"{self.__class__.__name__}({attributes})"
