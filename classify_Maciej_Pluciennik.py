import numpy as np


# Sigmoid activation function
def activation_function(x):
    return 1 / (1 + np.exp(-x * 0.3))  # lambda = 0.3


# Derivative
def activation_derivative(x):
    return x * (1 - x)


# Implementation of perceptron learning algorithm
def perceptron_learning(letter_set, learning_rate_modifier, upper_error, max_of_cycles):
    input_size = len(letter_set[0][0])

    # Initialize weights in range [-1,1]
    weights = np.random.uniform(low=-1, high=1, size=input_size)

    cycle_count = 0

    while cycle_count < max_of_cycles:
        total_error = 0

        for example, target in letter_set:
            # Calculate input signal
            input_signal = np.dot(example, weights)

            # Put it through activation function
            output = activation_function(input_signal)

            # Calculate error
            error = 0.5 * (target - output) ** 2
            total_error += error

            # Update weights
            weights += learning_rate_modifier * (target - output) * activation_derivative(output) * example

        cycle_count += 1

        # Condition for end of learning process
        if total_error < upper_error:
            print(f'Uczenie zakończone po {cycle_count} cyklach.')
            break

    return weights


# Define learning set
training_set = [
    (np.array([
        1, 1, 1, 1,
        1, 0, 0, 1,
        1, 1, 1, 1,
        1, 0, 0, 1,
        1, 0, 0, 1
    ]), 1),  # Letter A
    (np.array([
        1, 1, 1, 1,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 1, 1, 1
    ]), 0),  # Letter C
]

# Parameters for algorithm
learning_rate = 0.4
max_error = 0.01
max_cycles = 1000

learned_weights = perceptron_learning(training_set, learning_rate, max_error, max_cycles)

# Test on sample image
test_example = np.array([
    0, 1, 1, 1,
    0, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
    0, 1, 1, 1
])
test_input_signal = np.dot(test_example, learned_weights)
prediction = activation_function(test_input_signal)

# Results of classification
if prediction == 1:
    print("Image shows letter A.")
else:
    print("Image shows letter C.")
