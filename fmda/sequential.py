import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Input
from fullrnn import FullSimpleRNN

def rnn_model_sequential(time_steps, input_dim, rnn_units, output_units):
    model = Sequential([
        Input(shape=(time_steps, input_dim)),
        FullSimpleRNN(units=rnn_units),
        layers.Dense(output_units, activation='softmax')
    ])
    return model

def manual_rnn_output(inputs, rnn_weights, rnn_bias, dense_weights, dense_bias):
    initial_state = np.zeros((inputs.shape[0], rnn_units))
    concatenated_inputs = np.concatenate([inputs, np.expand_dims(initial_state, 1)], axis=-1)
    rnn_output = np.tanh(np.matmul(concatenated_inputs, rnn_weights) + rnn_bias)
    
    dense_input = rnn_output[:, -1, :]
    dense_output = np.matmul(dense_input, dense_weights) + dense_bias
    softmax_output = np.exp(dense_output) / np.sum(np.exp(dense_output), axis=1, keepdims=True)
    
    return softmax_output

def create_and_test_model():
    time_steps = 10
    input_dim = 8
    rnn_units = 32
    output_units = 1
    batch_size = 5

    # Create model
    model = rnn_model_sequential(time_steps, input_dim, rnn_units, output_units)

    # Generate random input data
    x = np.random.random((batch_size, time_steps, input_dim))

    # Predict using the model
    model_output = model.predict(x)

    # Manually calculate the output
    initial_state = np.zeros((batch_size, rnn_units))
    kernel = model.get_layer('full_simple_rnn').kernel.numpy()
    bias = model.get_layer('full_simple_rnn').bias.numpy()

    # Manually calculate the output for each time step
    for t in range(time_steps):
        input_t = x[:, t, :]
        output_t, initial_state = plain_python_full_simple_rnn(input_t, initial_state, kernel, bias)

    manual_output = output_t[:, np.newaxis, :]

    # Compare outputs
    difference = np.max(np.abs(model_output - manual_output))
    print("Difference between model.predict and manual output:", difference)

if __name__ == "__main__":
    create_and_test_model()
