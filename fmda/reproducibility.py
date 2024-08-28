# make session reproducible
import os
import random
import numpy as np
import tensorflow as tf
environ={'TF_DETERMINISTIC_OPS':'1','PYTHONHASHSEED':'0','TF_CPP_MIN_LOG_LEVEL':'2'}
# print('setting',environ)
os.environ.update(environ)
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    print('resetting random seeds to %i' % seed)
# set_seed()
# print('call set_seed() or set_seed(seed=value) to reset')

