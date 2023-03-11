#https://www.glennklockwood.com/data-intensive/analysis/perceptron.html
import pandas
import numpy
#import matplotlib 
import matplotlib.pyplot
import time
inputs = pandas.DataFrame(
    [[0, 0],
    [0, 1],
    [1, 0],
    [1, 1]],
    columns=["input 1", "input 2"])
print(inputs)
inputs.index.name = "observation #"

ground_truth = pandas.Series([0, 1, 1, 1], name="true output", index=inputs.index)

print("Inputs and true outputs (truth table) are:")
print(pandas.concat((inputs, ground_truth), axis=1))


def linear(x, weights, bias):
    """Linear model function
    """
    return numpy.dot(x, weights) + bias


def sigmoid(x):
    """Activation function
    """
    return 1.0 / (1.0 + numpy.exp(-x))

matplotlib.pyplot.plot(numpy.arange(-10, 10, 0.1), sigmoid(numpy.arange(-10, 10, 0.1)), label="sigmoid")
matplotlib.pyplot.grid()
matplotlib.pyplot.title("Sigmoid activation function")
pass

numpy.random.seed(seed=1)
print("Setting initial weights to random values.")
weights = numpy.random.rand(inputs.shape[1], 1)

print(pandas.DataFrame(weights, index=inputs.columns, columns=["weight"]))

bias = numpy.random.rand(1)[0]
print("Setting starting bias to a random value: {:4f}".format(bias))

learning_rate = 0.05
print("Setting learning rate to {} based on prior experience.".format(learning_rate))

NUM_ITERATIONS = 10000

# convert dataframe bits into arrays for processing
inputs_array = inputs.to_numpy()
truth_array = ground_truth.to_numpy().reshape(-1, 1)

x = inputs_array
t0 = time.time()
for i in range(NUM_ITERATIONS):
    y = linear(x, weights, bias)
    f = sigmoid(y)

    error = numpy.abs(f - truth_array)

    # calculate out partial derivatives for each input
    dE_df = error/(f - truth_array)
    df_dy = sigmoid(y) * (1.0 - sigmoid(y))
    dy_dw = x
    dE_dy = dE_df * df_dy
    dE_dw = numpy.dot(dy_dw.T, dE_dy)  # dy_dw = x

    # update weights and biases - the error is the sum of error over each input
    weights -= learning_rate * dE_dw
    bias -= learning_rate * dE_dy.sum()

    if i % (NUM_ITERATIONS / 10) == 0:
        print("error at step {:5d}: {:10.2e}".format(i, error.sum()))

print("Final weights: {}".format(weights.flatten()))
print("Final bias:    {}".format(bias))
print("{:d} iterations took {:.1f} seconds".format(NUM_ITERATIONS, time.time() - t0))


predicted_output = sigmoid(numpy.dot(x, weights) + bias)
predicted_output = pandas.DataFrame(
    predicted_output,
    columns=["predicted output"],
    index=inputs.index)

print(predicted_output)

print(pandas.concat((
    inputs,
    ground_truth,
    predicted_output),
    axis=1))