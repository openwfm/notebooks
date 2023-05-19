# Reference
# =========
# Why does deep and cheap learning work so well?∗
# Henry W. Lin, Max Tegmark, and David Rolnick
# Dept. of Physics, Harvard University, Cambridge, MA 02138
# Dept. of Physics, Massachusetts Institute of Technology, Cambridge, MA 02139 and Dept. of Mathematics, Massachusetts Institute of Technology, Cambridge, MA 02139
# Here the x input takes two numbers and produces the multiplication as output
import numpy as np

# multiplication approximator
lambda_val = 0.000001
sigmoid = lambda x : 1./(1 + np.exp(-x))
sigmoid_sec = lambda x: sigmoid(x)*(1 - sigmoid(x))*(1 - 2*sigmoid(x))
bias1 = 2.3787

# provide the input here, it works for two numbers as of now.
# It should not work for very large numbers. In that case,
# place lambda_val to a smaller value. But it does work really well 
# till 1e4 ~ 1e5 range.
x = np.array([12, 45]).reshape(-1,1)
W1 = np.array([[lambda_val, lambda_val], [-lambda_val, -lambda_val], [lambda_val, -lambda_val], [-lambda_val, lambda_val]])
b1 = np.array([bias1, bias1, bias1, bias1]).reshape(-1,1)

mu = 1/((lambda_val**2)*4*sigmoid_sec(bias1))

W2 = np.array([mu, mu, -mu, -mu])
z1 = np.dot(W1, x) + b1
a1 = sigmoid(z1)
z2 = np.dot(W2, a1)

print(z2[0])