# make session reproducible
import os
print('setting TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0')
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'
def set_seeds():
    import random
    random.seed(123)
    import numpy as np
    np.random.seed(123)
    import tensorflow as tf
    tf.random.set_seed(123)
    tf.keras.utils.set_random_seed(123)
    print('resetting random seeds')
set_seed()

