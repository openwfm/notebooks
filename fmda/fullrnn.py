# This code is partially based on a conversation with ChatGPT, 
# an AI language model trained by OpenAI. 

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Input

class FullSimpleRNN(layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim + self.units, self.units),
                                      initializer='glorot_uniform',
                                      name='kernel')
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    name='bias')
        super().build(input_shape)

    def call(self, inputs, initial_state=None):
        if initial_state is None:
            initial_state = tf.zeros((tf.shape(inputs)[0], self.units))

        output = tf.tanh(tf.matmul(tf.concat([inputs, tf.expand_dims(initial_state, 1)], axis=-1), self.kernel) + self.bias)
        return output, initial_state

def plain_python_full_simple_rnn(inputs, initial_state, kernel, bias):
    # assume that inputs has dimensions (batch_size, timesteps, input_dim) 
    # and initial_state has dimensions (batch_size, units).
    # We first concatenate inputs and initial_state along the last axis, 
    # resulting in an array with dimensions (batch_size, timesteps, input_dim + units). 
    # Next, we multiply this arrats by a weight matrix kernel that has dimensions
    # (input_dim + units, units) which results in an array with dimensions
    # (batch_size, timesteps, units). Finally, we add a bias term bias that has 
    # dimensions (units,) to the result element-wise resulting in an array with
    # dimensions (batch_size, timesteps, units). 

    print('inputs.shape=(batch_size, timesteps, input_dim):',inputs.shape)
    print('initial_state.shape=(batch_size, units):',initial_state.shape)
    print('kernel.shape=(input_dim + units, units):',kernel.shape)
    print('bias.shape=(units,):',bias.shape)
    initial_state_expanded = np.expand_dims(initial_state, 1)
    print('initial_state_expanded.shape=(batch_size, 1, input_dim):',initial_state_expanded.shape)
    inputs_and_initial_state_expanded = np.concatenate([inputs,initial_state_expanded], axis=-1)
    print('inputs_and_initial_state_expanded.shape=(batch_size,timesteps+1,batch_size)',inputs_and_initial_state_expanded.shape)
    #output = np.tanh(np.matmul(np.concatenate([inputs, np.expand_dims(initial_state, 1)], axis=-1), kernel) + bias)
    output = np.tanh(np.matmul(inputs_and_initial_state_expanded, kernel) + bias)
    print('output.shape=(units,):',output.shape)
    return output

def test_full_simple_rnn_functional_model():
    units = 60
    timesteps = 1
    input_dim = 2

    x = np.random.random((32, timesteps, input_dim))
    initial_state = np.random.random((32, units))

    inputs = Input(shape=(timesteps, input_dim), name="input_1")
    initial_state_input = Input(shape=(units,), name="input_2")
    rnn_layer = FullSimpleRNN(units, activation="tanh")
    rnn_output, _ = rnn_layer(inputs, initial_state_input)
    model = Model([inputs, initial_state_input], rnn_output)

    rnn_output = model.predict({"input_1": x, "input_2": initial_state})
    # print("Output from model.predict:\n", rnn_output)

    kernel = rnn_layer.kernel.numpy()
    bias = rnn_layer.bias.numpy()

    plain_python_output = plain_python_full_simple_rnn(x, initial_state, kernel, bias)
    # print("Output from plain Python code:\n", plain_python_output)

    difference = np.max(np.abs(rnn_output - plain_python_output))
    print("Difference between model.predict and plain Python code output: ", difference)

def test_full_simple_rnn_dims():
    # Test the initialization and build methods of the FullSimpleRNN class
    layer = FullSimpleRNN(units=3)
    input_shape = (None, 1, 2)
    layer.build(input_shape)
    print(layer.kernel)
    print(layer.bias)
    
    # Test the call method with some sample inputs
    inputs = tf.random.normal((32, 1, 2))
    initial_state = tf.random.normal((32, 3))
    output, _ = layer(inputs, initial_state)
    print(output.shape)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Input

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
    model.summary()

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
    #test_full_simple_rnn_dims()
    test_full_simple_rnn_functional_model()
    #create_and_test_model()
