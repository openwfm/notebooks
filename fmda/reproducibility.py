# make session reproducible
import os
print('setting TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0')
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'
def set_seed(seed=123):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    print('resetting random seed do %i' % seed)
set_seed()
print('call set_seed() or set_seed(seed=value) to reset')

