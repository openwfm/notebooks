# make session reproducible
import os
environ={'TF_DETERMINISTIC_OPS':'1','PYTHONHASHSEED':'0','TF_CPP_MIN_LOG_LEVEL':'2'}
# print('setting',environ)
os.environ.update(environ)
def set_seed(seed=123):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    print('resetting random seeds to %i' % seed)
# set_seed()
# print('call set_seed() or set_seed(seed=value) to reset')

