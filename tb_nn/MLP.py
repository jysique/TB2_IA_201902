from Perceptron import Perceptron, add_bias_node
import numpy as np


def _linear_delta(targets, outputs, nobs):
    return (targets - outputs) / nobs


def _logistic_delta(targets, outputs, *args):
    return (targets - outputs) * outputs


def _softmax_delta(targets, outputs, nobs):
    return (targets - outputs) / nobs


_calc_deltao = {
    "linear": _linear_delta,
    "logistic": _logistic_delta,
    "softmax": _softmax_delta
}


def _linear_activation(outputs, *args):
    return outputs


def _logistic_activation(outputs, beta, *args):
    return 1 / (1 + np.exp(-beta * outputs))


def _softmax_activation(outputs, *args):
    # this is multinomial logit
    eX = np.exp(outputs)
    return eX / eX.sum(axis=1)[:, None]


_activation_funcs = {
    "linear": _linear_activation,
    "logistic": _logistic_activation,
    "softmax": _softmax_activation,
}


class MLP(Perceptron):
    """
    A Multi-Layer Perceptron
    """

    def __init__(self, nhidden, eta, beta=1, momentum=0.9, outtype='logistic'):
        # Set up network size
        self.nhidden = nhidden
        self.eta = eta

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

    def _init_weights(self):
        # Initialise network
        weights1 = np.random.rand(self.m + 1, self.nhidden) - 0.5
        weights1 *= 2 / np.sqrt(self.m)
        weights2 = np.random.rand(self.nhidden + 1, self.n) - 0.5
        weights2 *= 2 / np.sqrt(self.nhidden)

        self.weights1 = weights1
        self.weights2 = weights2

    def earlystopping(self, inputs, targets, valid_input, valid_target,
                      max_iter=100, epsilon=1e-3, disp=True):

        self._initialize(inputs, targets)
        valid_input = add_bias_node(valid_input)

        # 2 iterations ago, last iteration, current iteration
        # current iteration, last iteration, 2 iterations ago
        last_errors = [0, np.inf, np.inf]

        count = 0

        while np.any(np.diff(last_errors) > epsilon):
            count += 1
            if disp:
                print(count)

            # train the network
            self.fit(inputs, targets, max_iter, init=False, disp=disp)
            last_errors[2] = last_errors[1]
            last_errors[1] = last_errors[0]

            # check on the validation set
            valid_output = self.predict(valid_input, add_bias=False)
            errors = valid_target - valid_output
            last_errors[0] = 0.5 * np.sum(errors ** 2)

        if disp:
            print("Stopped in %d iterations" % count, last_errors)
        return last_errors[0]

    def fit(self, inputs, targets, max_iter, disp=True, init=True):
        if init:
            self._initialize(inputs, targets)
        inputs = self.inputs
        targets = self.targets
        weights1 = self.weights1
        weights2 = self.weights2
        eta = self.eta
        momentum = self.momentum
        nobs = self.nobs

        outtype = self.outtype

        # Add the inputs that match the bias node
        inputs = add_bias_node(inputs)
        change = list(range(self.nobs))

        updatew1 = np.zeros_like(weights1)
        updatew2 = np.zeros_like(weights2)

        for n in range(1, max_iter + 1):
            # predict attaches hidden
            outputs = self.predict(inputs, add_bias=False)

            error = targets - outputs
            obj = .5 * np.sum(error ** 2)

            # Different types of output neurons
            deltao = _calc_deltao[outtype](targets, outputs, nobs)
            hidden = self.hidden
            deltah = hidden * (1. - hidden) * np.dot(deltao, weights2.T)

            updatew1 = (eta * (np.dot(inputs.T, deltah[:, :-1])) +
                        momentum * updatew1)
            updatew2 = (eta * (np.dot(self.hidden.T, deltao)) +
                        momentum * updatew2)
            weights1 += updatew1
            weights2 += updatew2

            # Randomise order of inputs
            np.random.shuffle(change)
            inputs = inputs[change, :]
            targets = targets[change, :]

        if disp:
            print("Iteration: ", n, " Objective: ", obj)

        # attach results
        self.weights1 = weights1
        self.weights2 = weights2
        self.outputs = outputs

    def predict(self, inputs=None, add_bias=True):
        inputs = inputs.astype(float)
        if inputs is None:
            inputs = self.inputs

        if add_bias:
            inputs = add_bias_node(inputs)

        hidden = np.dot(inputs, self.weights1)
        hidden = _activation_funcs["logistic"](hidden, self.beta)
        hidden = add_bias_node(hidden)
        self.hidden = hidden

        outputs = np.dot(self.hidden, self.weights2)

        outtype = self.outtype

        # Different types of output neurons
        self.outputsInfernce = _activation_funcs[self.outtype](outputs, self.beta)
        # return _activation_funcs[self.outtype](outputs, self.beta)
        return self.outputsInfernce

    def think(self, inputs):
        if self.outputsInfernce.argmax() == 0:
            stringOut = "Acido"
        if self.outputsInfernce.argmax() == 1:
            stringOut = "Basico"
        else:
            stringOut = "Neutro"
        return stringOut

