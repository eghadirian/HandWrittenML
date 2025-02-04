import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0

def dot(v,w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def perceptron_output(weights, bias, x):
    calculation = dot(weights, x) + bias
    return step_function(calculation)

# FFN
def sigmoid(t):
    return  1/(1+np.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    outputs = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)
        input_vector = outputs
    return outputs

def back_propagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)
    output_deltas = [output * (1-output) * (output-target) # derivative of sigmoid * (output-target)
                     for output, target in zip(outputs, targets)]
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i]*hidden_output
    hidden_deltas = [hidden_output * (1-hidden_output)
                     *dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -=hidden_deltas[i]*input

