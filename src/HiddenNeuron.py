import numpy as np
import ActivationFunctions
import math

class HiddenNeuron():
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
        number of neurons in self.precedessors
    successors : list of neurons
        list of neurons receiving this neuron's outputs
    n_successors: int
        number of neurons in self.successors
    w : array of float
        weights vector
    f : callable
        activation function
    f_parameters : list of float
        list of additional (optional) args of self.f
    net : float
        inner product between the weight vector and the unit's input at a given iteration
    steps : int
        number of epochs undergone by the neuron in the training
    last_predict : float
        output of the neuron (instance variable exploited for predictions out of training)
    partial_weight_update : array of float
        the partial sum (on the minibatch) that will compose the DeltaW weight update value
    old_weight_update : array of float
        the old weight update value DeltaWs
    partial_successors_weighted_errors : float
        the partial sum of successors' errors values weighted by the weight of the link bethween the two units

    '''

    def __init__(self, index:int, activation_fun:callable = ActivationFunctions.sigmoid, *args):
        '''
        Neuron constructor
        
        Parameters
        ----------
        index: int
            the index of the neuron in the NN
        activation_fun: callable
            the Neuron's actviation function
        args: additional (optional) parameters of the activation function

        Returns
        -------
        return: -
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
        # inizialized to zero to prevent bad behaviors when combinated with the ReLU (zero derivate)
        
        # the creation of the variables is not necessary, but can help in preventing error, 
        # also resetting the variable can help in this sense
    
    def increase_steps(self):
        '''
        Increase by one self.steps, is called at every epoch to notify the neuron
        
        Returns
        -------
        return: -
        '''
        self.steps += 1 
       
    def update_weights(self, learning_rate:float = 0.01, lr_decay_tau:int = 0, 
                       eta_tau:float = 0.0, lambda_tikhonov:float = 0.0, alpha_momentum:float = 0.0, 
                       nesterov_momentum:bool = False):
        '''
        Updates the weight vector (w) of the Neuron using, if active, nesterov momentum, standard momentum
        tikhonov regularization and learning rate decay
        
            w_t+1 = w_t + gradient_based_update + momentum_based_update + tikhonov_based_update
            
            -> gradient_based_update depends on learning_rate, eta_tau, lr_decay_tau and the gradient
            -> momentum_based_update depends alpha momentum
            -> tikhonov_based_update depends on lambda_tikhonov
        
        Parameters
        ----------
        learning_rate: float
            Eta hyperparameter to control the learning rate of the algorithm
        lr_decay_tau: int
            Number of epochs after which the learning rate stop decreasing, before which the learning rate decay
        eta_tau: float
            Learning rate after iteration tau if lr_decay_tau > 0, before is used to 
                        make the learnig rate decay
        lambda_tikhonov: float
            Lambda hyperparameter to control the learning algorithm complexity (Tikhonov Regularization / Ridge Regression)
        alpha_momentum: float
            Momentum Hyperparameter

        Returns
        -------
        return: -
        '''
        # if the learning rate decay is active, the learning step is adjusted depending on the iteration number
        # so to slow the intensities of weights update as the algorithm proceeds
        eta = learning_rate
        if self.steps < lr_decay_tau:
            alpha = self.steps/lr_decay_tau
            eta = learning_rate * (1 - alpha) + alpha * eta_tau
        elif lr_decay_tau > 0:
            eta = eta_tau
            
        # here the final gradient is multiplied by the learning rate
        weight_update = (eta * self.partial_weight_update)
        
        # if nesterov momentum is exploited, we must apply the weight update on the original weight vector
        self.w -= alpha_momentum * self.old_weight_update * nesterov_momentum 
        
        # here we add the tikhonov regularization
        tmp = np.copy(self.w)
        tmp[0] = 0 # avoid to regularize the bias
        weight_update = weight_update - (lambda_tikhonov * tmp)
        
        # here we add the momentum influence on the final weight update
        weight_update = weight_update + (self.old_weight_update * alpha_momentum)

        # weight update
        self.w += weight_update
        
        # if nesterov momentum is exploited, we add now the momentum to calculate the right gradient
        self.w += alpha_momentum * self.old_weight_update * nesterov_momentum 
        
        # reset of every accumulative variable used
        self.old_weight_update = weight_update.copy()
        self.partial_weight_update = np.zeros(self.n_predecessors + 1)
        self.partial_successors_weighted_errors = 0.0
        
        # a fail fast approach
        if sum(np.isinf(self.w)): raise Exception('Execution Failed, w:' + str(self.w))
     
    def update_weights_adamax(self, learning_rate:float = 0.002, exp_decay_rates_1:float = 0.9, exp_decay_rates_2:float = 0.999,
                              lambda_tikhonov:float = 0.0):
        '''
        Updates the weight vector (w) of the Neuron using Adamax
        is the application of the Adam algoritm regularized with the infinity norm of the gradient

            alpha: learning_rate
            beta1,beta2 : exp_decay_rates_1, exp_decay_rates_2
            grad(theta): actual gradient (self.partial_weight_update * -1)
            theta_0: Initial parameter vector
            
            weight update:
                t ← t + 1
                g_t ← grad(theta_t-1)                               (Get gradients at timestep t)
                m_t ← beta1 * m_t-1 + (1 - beta1) · g_t             (Update biased first moment estimate)
                u_t ← max(beta2 * u_t/1 , |g_t|)                    (Update the exponentially weighted infinity norm)
                
                tmp = beta1 ** t                                     (beta1 to the power of t)
                theta_t ← theta_t/1 / (alpha / (1 / tmp)) ·m_t/u_t   (Update parameters)

        Parameters
        ----------
        learning_rate: float
            Eta hyperparameter to control the learning rate of the algorithm
        exp_decay_rates_1: float
            Exponential decay rates for the momentum
        exp_decay_rates_2: float
            Exponential decay rates for the infinite norm
        lambda_tikhonov: float
            Lambda hyperparameter to control the learning algorithm complexity (Tikhonov Regularization / Ridge Regression)
        
        Returns
        -------
        return: -
        '''
        
        # our gradient is already multiplyed by -1 so we revers the sign
        gradient = self.partial_weight_update * -1
        
        # update biased first moment estimate
        momentum = exp_decay_rates_1 * self.old_weight_update + (1 - exp_decay_rates_1) * gradient
        
        # update the exponentially weighted infinity norm
        self.exponentially_weighted_infinity_norm = max(self.exponentially_weighted_infinity_norm * exp_decay_rates_2, 
                                                        np.linalg.norm(gradient, ord=np.inf))
        
        # compute the final weight update
        # momentum influence
        dummy_1 = momentum/self.exponentially_weighted_infinity_norm
        # learning rate decay influence
        dummy_2 = (1 - math.pow(exp_decay_rates_1, (self.steps + 1)))
        # steps + 1 because in the first iteration, per the standard weight decay steps needs to be 0 but here 1
        # weight update
        weight_update = (learning_rate/dummy_2)*dummy_1
        
        
        # here we add the tikhonov regularization
        tmp = np.copy(self.w)
        weight_update = weight_update + (lambda_tikhonov * tmp)
        
        # here the weights are finally updated
        self.w -= weight_update
        
        # reset of every accumulative variable used
        self.old_weight_update = weight_update.copy()
        self.partial_weight_update = np.zeros(self.n_predecessors + 1)
        self.partial_successors_weighted_errors = 0.0
        
        # a fail fast approach
        if sum(np.isinf(self.w)): raise Exception('Execution Failed, w:' + str(self.w))
        
    def initialise_weights(self, rand_range_min:float, rand_range_max:float, fan_in:bool, random_generator:np.random.Generator):
        '''
        Initialises the Neuron's weights vector (w).
        Updates the unit's numbers of predecessors and successors (the network has already been completely linked together)
        
        Parameters
        ----------
        rand_range_min: float
            minimum value for random weights initialisation range
        rand_range_max: float
            maximum value for random weights initialisation range
        fan_in: bool
            if the weights'initialisation should also consider the Neuron's fan-in

        Returns
        -------
        return: -
        '''

        # here the predeccessors are counted
        self.n_predecessors = len(self.predecessors)
        self.n_successors = len(self.successors)
        # here are created vectors of the correct len, counting the bias
        self.old_weight_update = np.zeros(self.n_predecessors + 1) # bias
        self.partial_weight_update = np.zeros(self.n_predecessors + 1) # bias
        
        # the weights are chosen uniformly at random in the range taken in input
        self.w = random_generator.uniform(rand_range_min, rand_range_max, self.n_predecessors + 1) # bias
        
        # reset of every accumulative variable
        self.steps = 0
        self.exponentially_weighted_infinity_norm = 1
        
        # if fan_in the weights are reduced accordingly
        if fan_in:
            self.w = self.w * 2/(self.n_predecessors + 1)
        
    def forward(self):
        '''
        Calculates the Neuron's output on the inputs incoming from the other units
        
        Returns
        -------
        return: -
        '''
        # the input vector is initialized and computed
        input = np.empty(self.n_predecessors + 1)# bias
        input[0] = 1 # bias
        for index, p in enumerate(self.predecessors):
            input[index + 1] = p.last_predict

        self.net = np.inner(self.w, input)
        self.last_predict = self.f(self.net, *self.f_parameters)
        # the prediction is stored so that other neurons can use the value
     
    def accumulate_weighted_error(self, delta: float, weight: float):
        '''
        Function used by successors, accumulate the weighted error of the successors.

        Parameters
        ----------
        delta: float
            the error of the successor
        weight: float
            the weight of the successor that correspond to the output of this unit

        Returns
        -------
        return: -
        '''

        # summation
        self.partial_successors_weighted_errors += (delta * weight)

    def backward(self):
        '''
        Calculates the Neuron's error contribute for a given learning pattern
        Calculates a partial weight update for the Neuron (Partial Backpropagation)
        
        Returns
        -------
        return: -
        '''
        # here we create the vector
        predecessors_outputs = np.empty(self.n_predecessors + 1) # bias
        
        # computing delta error of the unit (before we have accumulated the successor's deltas in 'partial_successors_weighted_errors')
        delta_error = self.partial_successors_weighted_errors * ActivationFunctions.derivative(self.f, self.net, *self.f_parameters)
        
        # here we compute the sum derived by the predeccessors
        predecessors_outputs[0] = 1 # bias
        for index, p in enumerate(self.predecessors):
            predecessors_outputs[index + 1] = p.last_predict # bias
            if p.type != "input":
                p.accumulate_weighted_error(delta_error, self.w[index + 1]) # bias

        # the summation is multiplied by delta error to get the partial weight update
        self.partial_weight_update += (delta_error * predecessors_outputs)

    def add_successor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's successors and
        update the predecessors' list of the successor neuron with the current neuron
        
        Parameters
        ----------
        neuron: HiddenNeuron or OutputNeuron
            the Neuron to add to the list of successors

        Returns
        -------
        return: -
        '''
        self.successors.append(neuron)
        neuron.add_predecessor(self)

    def add_predecessor(self, neuron):
        '''
        Adds a neuron to the list of the Neuron's predecessors
        
        Parameters
        ----------
        neuron: HiddenNeuron or InputNeuron
            the Neuron to add to the list of predecessors

        Returns
        -------
        return: -
        '''
        self.predecessors.append(neuron)

    def extend_successors(self, neurons:list):
        '''
        Extends the list of the Neuron's successors and
        update the predecessors' list of the successors neurons with the current neuron
        
        Parameters
        ----------
        neurons: list of HiddenNeuron or list of OutputNeuronthe 
            list of Neurons to add to the list of successors
        
        Returns
        -------
        return: -
        '''
        self.successors.extend(neurons)
        for successor in neurons:
            successor.add_predecessor(self)