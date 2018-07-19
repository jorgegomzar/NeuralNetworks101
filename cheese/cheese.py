from numpy import exp, array, random, dot
import matplotlib.pyplot as plt

training_set_inputs = array([[0,0,0], [1, 1, 1], [1, 0, 0], [1, 1, 0]])
training_set_outputs = array([[0, 1, 0, 1]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1

for iteration in range(10000):
	output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
	synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))

print("Answer with 1 (yes) or 0 (no)")
x1 = float(input('Nice weather? '))
x2 = float(input('Companion? '))
x3 = float(input('Public transport? '))
if ((1 / (1 + exp(-(dot(array([x1, x2 ,x3]), synaptic_weights))))) > 0.5):
	print("I'm going.")
else: 
	print("I can miss this one.")