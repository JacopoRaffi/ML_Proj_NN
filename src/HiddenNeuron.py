import numpy as np
#from ActivationFunctions import ActivationFunctions
import ActivationFunctions
import math
from ABCNeuron import ABCNeuron

class HiddenNeuron(ABCNeuron):
    '''
    Implementation of an hidden neuron composing the NN
    
    Attributes
    ----------
    index : int
        the index of the neuron in the NN
    type : str
        the type of the neuron
    predecessors : list of neurons
        list of neurons sending their outputs in input to this neuron
    n_predecessors: int
        number of units linked as predecessors to this neuron
    successors : list of neurons
        list of neurons receiving this neuron's outputs
    n_successors: int
        number of units linked as successors to this neuron
    w : array of float
        weights vector
    f : callable
        activation function
    f_parameters : list of float
        the list for the additional (optional) parameters of the activation function
    net : float
        inner product between the weight vector and the unit's input at a given iteration
    steps : int
        number of learning steps (weight update) undergone by the neuron
    last_predict : float
        output of the neuron (instance variable exploited for predictions out of training)
    partial_weight_update : array of float
        the partial sum (on the minibatch) that will compose the DeltaW weight update value
    old_weight_update : array of float
        the old weight update value DeltaW
    partial_successors_weighted_errors : float
        the partial sum of successors' errors values weighted by the weight of the link bethween the two units

    '''

    def __init__(self, index:int, activation_fun:callable = ActivationFunctions.sigmoid, *args):
        '''
        Neuron initialisation
        
        param index: the index of the neuron in the NN
        param n_input: the number of inputs receivable by the Neuron
        param activation_fun: the Neuron's actviation function
        param rand_range_min: minimum value for random weights initialisation range
        param rand_range_max: maximum value for random weights initialisation range
        param fan_in: if the weights'initialisation should also consider the Neuron's fan-in
        param args: additional (optional) parameters of the activation function

        :return: -
        '''
        self.index = index
        self.type = 'hidden'
        self.predecessors = [] # list of neurons sending their outputs in input to this neuron
        self.n_predecessors = 0
        self.successors = [] # list of neurons receiving this neuron's outputs
        self.n_successors = 0
        self.w = np.array([]) # weights vector (initialised later)
        self.f = activation_fun # activation function
        self.f_parameters = list(args) # creates the list for the additional (optional) parameters of the activation function
        self.net = 0.0 # inner product between the weight vector and the unit's input at a given iteration
        
        self.steps = 0 # number of learning steps (weight update) undergone by the neuron
        
        self.last_predict = 0.0 # output of the neuron (instance variable exploited for predictions out of training)
        self.partial_weight_update = np.array([]) # the partial sum (on the minibatch) that will compose the DeltaW weight update value
        self.old_weight_update = np.array([]) # the old weight update value DeltaW
        self.partial_successors_weighted_errors = 0.0 # the partial sum of successors' errors values weighted by the weight of the link bethween the two units
        
        
        self.exponentially_weighted_infinity_norm = 1 # variable used in for the adamax weight update
        
        
        # the creation of the variable is not necessary because can be created in any moment, just having the istance of the object but
        # the None value can help in preventing error, also resetting the variable can help in this sense
    
    def add_nesterov_momentum(self, alpha_momentum:float = 0.0):
        '''
        Updates the weight vector (w) of the Neuron with Nesterov's Momentum
        this update should be done before the next minibatch learning iteration
        
        :param alpha_momentum: Nertov's Momentum Hyperparameter
        :return: -
        '''
        
        self.w = self.w + alpha_momentum*self.old_weight_update
      
    def increase_steps(self):
        self.steps += 1 
       
    def update_weights(self, learning_rate:float = 0.01, lr_decay_tau:int = 0, 
                       eta_tau:float = 0.0, lambda_tikhonov:float = 0.0, alpha_momentum:float = 0.0, 
                       nesterov_momentum:bool = False):
        '''
        Updates the weight vector (w) of the Neuron
        
        param learning_rate: Eta hyperparameter to control the learning rate of the algorithm
        param lr_dacay_tau: Number of iterations (tau) if the learning rate decay procedure is adopted
        param eta_tau: Eta hyperparameter at iteration tau if the learning rate decay procedure is adopted
        param lambda_tikhonov: Lambda hyperparameter to control the learning algorithm complexity (Tikhonov Regularization / Ridge Regression)
        param alpha_momentum: Momentum Hyperparameter

        return: -
        '''
        # if the learning rate decay is active, the learning step is adjusted depending on the iteration number
        # so to slow the intensities of weights update as the algorithm proceeds (recommended in minibatch)
        # self.steps += 1
        eta = learning_rate
        if self.steps < lr_decay_tau:
            alpha = self.steps/lr_decay_tau
            eta = learning_rate * (1 - alpha) + alpha * eta_tau
        elif lr_decay_tau > 0:
            eta = eta_tau
            
        # here is the final gradient multiplied by the learning rate
        weight_update = (eta * self.partial_weight_update)
        
        # if nesterov momentum is exploited, we must apply the weight update on the original weight vector
        self.w -= alpha_momentum * self.old_weight_update * nesterov_momentum 
        
        # here we add the tikhonov regularization
        tmp = np.copy(self.w)
        tmp[0] = 0 # avoid to regularize the bias
        weight_update = weight_update - (lambda_tikhonov * tmp)
        
        # here we add the momentum influence on the final weight update
        weight_update = weight_update + (self.old_weight_update * alpha_momentum)

        
        self.w += weight_update
        
        # if nesterov momentum is exploited, we add now the momentum to calculate the right gradient
        self.w += alpha_momentum * self.old_weight_update * nesterov_momentum 
        
        # reset of every accumulative variable used
        self.old_weight_update = weight_update.copy()
        self.partial_weight_update = np.zeros(self.n_predecessors + 1)
        self.partial_successors_weighted_errors = 0.0
        
        if sum(np.isinf(self.w)): raise Exception('Execution Failed')
     
    def update_weights_adamax(self, learning_rate:float = 0.002, exp_decay_rates_1:float = 0.9, exp_decay_rates_2:float = 0.999,
                              lambda_tikhonov:float = 0.0):
        '''
        Updates the weight vector (w) of the Neuron using Adamax Algorithm
        
        param learning_rate: Eta hyperparameter to control the learning rate of the algorithm
        param exp_decay_rates_1: Exponential decay rates for the momentum
        param exp_decay_rates_2: Exponential decay rates for the infinite norm
        
        return: -
        '''
        
        self.steps += 1
        
        # our gradient is already multiplyed by -1 !!!!
        gradient = self.partial_weight_update * -1
        
        # Update biased first moment estimate
        momentum = exp_decay_rates_1 * self.old_weight_update + (1 - exp_decay_rates_1) * gradient
        
        # Update the exponentially weighted infinity norm
        self.exponentially_weighted_infinity_norm = max(self.exponentially_weighted_infinity_norm * exp_decay_rates_2, 
                                                        np.linalg.norm(gradient, ord=np.inf))
        
        # Update parameters
        dummy_1 = momentum/self.exponentially_weighted_infinity_norm
        dummy_2 = (1 - math.pow(exp_decay_rates_1, self.steps))
        weight_update = -(learning_rate/dummy_2)*dummy_1
        
        
        # here we add the tikhonov regularization
        tmp = np.copy(self.w)
        #tmp[0] = 0 # avoid to regularize the bias
        weight_update = weight_update - (lambda_tikhonov * tmp)
        
        self.w += weight_update
        
        # reset of every accumulative variable used
        self.old_weight_update = weight_update.copy()
        self.partial_weight_update = np.zeros(self.n_predecessors + 1)
        self.partial_successors_weighted_errors = 0.0
        
        if sum(np.isinf(self.w)): raise Exception('Execution Failed')
        
    def initialise_weights(self, rand_range_min:float, rand_range_max:float, fan_in:bool, random_generator:np.random.Generator):
        '''
        Initialises the Neuron's weights vector (w).
        Updates the unit's numbers of predecessors and successors (the network has already been completely linked together)
        
        param rand_range_min: minimum value for random weights initialisation range
        param rand_range_max: maximum value for random weights initialisation range
        param fan_in: if the weights'initialisation should also consider the Neuron's fan-in

        return: -
        '''

        self.n_predecessors = len(self.predecessors)
        self.n_successors = len(self.successors)
        self.old_weight_update = np.zeros(self.n_predecessors + 1) # bias
        self.partial_weight_update = np.zeros(self.n_predecessors + 1) # bias
        
        self.w = random_generator.uniform(rand_range_min, rand_range_max, self.n_predecessors + 1) # bias
        
        self.steps = 0
        self.exponentially_weighted_infinity_norm = 1
        
        if fan_in:
            self.w = self.w * 2/(self.n_predecessors + 1)
        
    def forward(self):
        '''
        Calculates the Neuron's output on the inputs incoming from the other units
        
        param input: Neuron's input vector

        return: -
        '''
        input = np.empty(self.n_predecessors + 1)# bias
        input[0] = 1 # bias
        for index, p in enumerate(self.predecessors):
            input[index + 1] = p.last_predict

        self.net = np.inner(self.w, input)
        self.last_predict = self.f(self.net, *self.f_parameters)
     
    def accumulate_weighted_error(self, delta: float, weight: float):
        '''
        function used by successors, accumulate the weighted error of the successors

        param delta: the error of the successor
        param weight: the weight of the successor that correspond to the output of this unit

        return: -
        '''

        # summation
        self.partial_successors_weighted_errors += (delta * weight)

    def backward(self):
        '''
        Calculates the Neuron's error contribute for a given learning pattern
        Calculates a partial weight update for the Neuron (Partial Backpropagation)
        
        return: -
        '''
        
        predecessors_outputs = np.zeros(self.n_predecessors + 1) # bias
        
        # computing delta error of the unit (before we have accumulated the successor's deltas in 'partial_successors_weighted_errors')
        delta_error = self.partial_successors_weighted_errors * ActivationFunctions.derivative(self.f, self.net, *self.f_parameters)
        
        predecessors_outputs[0] = 1 # bias
        for index, p in enumerate(self.predecessors):
            predecessors_outputs[index + 1] = p.last_predict # bias
            if p.type != "input":
                p.accumulate_weighted_error(delta_error, self.w[index + 1]) # bias

        self.partial_weight_update += (delta_error * predecessors_outputs)# accumulating weight updates for the batch version

    def add_successor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's successors and
        update the predecessors' list of the successor neuron with the current neuron
        
        param neuron: the Neuron to add to the list of successors

        return: -
        '''
        self.successors.append(neuron)
        neuron.add_predecessor(self)
    
    def extend_successors(self, neurons:list):
        '''
        Extends the list of the Neuron's successors and
        update the predecessors' list of the successors neurons with the current neuron
        
        param neurons: the list of Neurons to add to the list of successors

        return: -
        '''
        self.successors.extend(neurons)
        for successor in neurons:
            successor.add_predecessor(self)

    def add_predecessor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's predecessors
        
        param neuron: the Neuron to add to the list of predecessors

        return: -
        '''
        self.predecessors.append(neuron)
