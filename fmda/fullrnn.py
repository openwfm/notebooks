import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class FullSimpleRNN(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(self.input_dim, self.units),
                                 initializer="glorot_uniform",
                                 dtype=tf.float32)  # Cast to float32
        self.U = self.add_weight(shape=(self.units, self.units),
                                 initializer="orthogonal",
                                 dtype=tf.float32)  # Cast to float32
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros",
                                 dtype=tf.float32)  # Cast to float32

    def call(self, inputs, states):
        prev_output = states[0] if isinstance(states, list) else states
        h = tf.matmul(inputs, self.W) + tf.matmul(prev_output, self.U)
        output = tf.tanh(h + self.b)
        return output, [output]

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
     
