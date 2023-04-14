
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Input

batch_size=4   # number of samples
hidden_units=5
features=2
timesteps=3

# The following code is partially based on 
# https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/
# modified from fmda_kf_rnn_orig.ipynb 
 

def create_RNN(hidden_units, dense_units, input_shape, activation):
    print('create_RNN: hidden_units=',hidden_units,'dense_units=',dense_units,
        'input_shape=',input_shape,'activation=',activation)
    inputs = tf.keras.Input(shape=input_shape)
    initial_state = tf.keras.Input(shape=(hidden_units,))
    x, state = tf.keras.layers.SimpleRNN(hidden_units, input_shape=input_shape,
         activation=activation[0], return_state=True)(inputs, initial_state=initial_state)
    outputs = tf.keras.layers.Dense(dense_units, activation=activation[1])(x)
    model = tf.keras.Model(inputs=[inputs, initial_state], outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def SimpleRNN_test():
    # Demo example
    print('SimpleRNN_test')

    demo_model = create_RNN(hidden_units=hidden_units, dense_units=1, 
                            input_shape=(timesteps,features), 
                            activation=['linear', 'linear'])
    print(demo_model.summary())
    w = demo_model.get_weights()
    #print(len(w),' weight arrays:',w)
    wname=('wx','wh','bh','wy','by','wz','bz')
    for i in range(len(w)):
      print(i,':',wname[i],'shape=',w[i].shape)
    wx, wh, bh, wy, by = w
    
    #  input batch_size x timesteps x features 
    x = tf.reshape(tf.range(batch_size*timesteps*features),
        [batch_size,timesteps,features])
    print('input x.shape=(batch_size,timesteps,features):',x.shape)
    #print('model.predict start')
    initial_state = np.zeros((batch_size, hidden_units))
    y_pred_model = demo_model.predict([x,initial_state])

    #print('model.predict end')
    
    o=np.zeros([batch_size,1])
    for i in range(batch_size):
      h = np.zeros(hidden_units)
      for j in range(timesteps):
          h = np.dot(x[i,j,:], wx) + np.dot(h,wh) + bh
      o[i,0] = np.dot(h, wy) + by
    
    difference = np.max(np.abs(y_pred_model - o))
    print("SimpleRNN model difference between model.predict and manual output:", difference)


# The following code is partially based on a conversation with ChatGPT, 
# an AI language model trained by OpenAI. 

from tensorflow.keras import layers

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
        self.activation_fn = tf.keras.activations.get(self.activation)
        super().build(input_shape)

    def step(self, state, inputs_t):
        output = self.activation_fn(tf.matmul(tf.concat([inputs_t, state], axis=-1), self.kernel) + self.bias)
        return output

    def call(self, inputs, initial_state=None):
        if initial_state is None:
            initial_state = tf.zeros((tf.shape(inputs)[0], self.units))

        inputs_t = tf.transpose(inputs, perm=[1, 0, 2])
        outputs = tf.scan(self.step, elems=inputs_t, initializer=initial_state)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])

        return outputs, outputs[:, -1, :]


def plain_python_full_simple_rnn(inputs, initial_state, kernel, bias, activation_fn=np.tanh):
    batch_size, timesteps, features = inputs.shape
    units = initial_state.shape[1]

    outputs = []
    state = initial_state
    for t in range(timesteps):
        output = activation_fn(np.matmul(np.concatenate([inputs[:, t, :], state], axis=-1), kernel) + bias)
        outputs.append(output)
        state = output

    outputs = np.stack(outputs, axis=1)
    return outputs, state

def test_full_simple_rnn_functional_model():
    print('test_full_simple_rnn_functional_model')

    x = np.random.random((batch_size, timesteps, features))
    initial_state = np.random.random((batch_size, hidden_units))

    inputs = Input(shape=(timesteps, features), name="input_1")
    initial_state_input = Input(shape=(hidden_units,), name="input_2")
    rnn_layer = FullSimpleRNN(hidden_units, activation="tanh")
    rnn_output, _ = rnn_layer(inputs, initial_state_input)
    model = Model([inputs, initial_state_input], rnn_output)
    print(model.summary())

    rnn_output = model.predict({"input_1": x, "input_2": initial_state})
    # print("Output from model.predict:\n", rnn_output)

    kernel = rnn_layer.kernel.numpy()
    bias = rnn_layer.bias.numpy()

    plain_python_output, _  = plain_python_full_simple_rnn(x, initial_state, kernel, bias)
    # print("Output from plain Python code:\n", plain_python_output)

    difference = np.max(np.abs(rnn_output - plain_python_output))
    print("Difference between model.predict and plain Python code output: ", difference)

if __name__ == "__main__":
    SimpleRNN_test()
    test_full_simple_rnn_functional_model()
