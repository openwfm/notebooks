import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class FullSimpleRNN(Layer):
    def __init__(self, units, activation='tanh', use_bias=True, **kwargs):
        super(FullSimpleRNN, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.W = self.add_weight(name='W',
                                 shape=(self.input_dim, self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.U = self.add_weight(name='U',
                                 shape=(self.units, self.units),
                                 initializer='orthogonal',
                                 trainable=True)
        if self.use_bias:
            self.b = self.add_weight(name='b',
                                     shape=(self.units,),
                                     initializer='zeros',
                                     trainable=True)
        else:
            self.b = None
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = tf.matmul(inputs, self.W) + tf.matmul(prev_output, self.U)
        if self.use_bias:
            h = h + self.b
        output = self.activation(h)
        return output, [output]

    def get_config(self):
        config = super(FullSimpleRNN, self).get_config()
        config.update({'units': self.units,
                       'activation': tf.keras.activations.serialize(self.activation),
                       'use_bias': self.use_bias})
        return config

def FullSimpleRnn_test():
    
    # Define test dimensions
    seq_length = 10
    batch_size = 5
    input_dim = 3
    hidden_dim = 2
    
    # Set up test data and initial state
    x = np.random.randn(seq_length, batch_size, input_dim)
    h0 = np.random.randn(batch_size, hidden_dim)
    
    # Instantiate FullSimpleRNN layer
    rnn = FullSimpleRNN(units=hidden_dim)
    
    # Compute forward pass
    output, state = rnn(tf.constant(x), [tf.constant(h0)])
    
    # Verify output values
    for t in range(seq_length):
        expected_output_t = np.tanh(np.dot(x[t], rnn.get_weights()[0]) +
                                    np.dot(h0, rnn.get_weights()[1]) +
                                    rnn.get_weights()[2])
        np.testing.assert_allclose(output[t].numpy(), expected_output_t, rtol=1e-5)
     
