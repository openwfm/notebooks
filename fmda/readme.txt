## Testing

The testing data lives in testing_dict.pickle

To validate:

    1. Start jupyter lab or jupyter notebook in directory fmda

    2. Run presentations/fmda_kf_rnn_orig.ipynb Note the hash after Check 5
       This creates file data/rnn_orig.json, check its timestamp if new.
       
    3. Run data/create_test_dict.ipynb
       This creates file data/testing_dict.pickle and data/errors.json, check timestamps.

    4. Run rmda_rnn_rain.ipynb
       The hash after Check 5 should match

Curently, the Check 5 hash value is 5.55077327554663e+19 (Python 3.10.9)

## Batching and training

in moisture_rnn.py commit e483fd27b04a0165f0c251cbff951287700a87f2 2024/1/2

original staircase() invoked by batch_type = 1
sequences example: [1 2 3 4 5] [2 3 4 5 6] etc
orig. no batching specified, left to keras
sequences in a batch executed in parallel => not stateful 

new staircase_2() invoked by batch_type = 2
sequences: [[1 2 3 4 5]  [2 3 4 5 6] ...  [5 6 7 8 9]
            [6 7 8 9 10] [7 8 9 10 11] ...
sequence in each batch continues same sequence from previous batch => stateful
flatted to pass to model.fit
then break into batches by specifying batch_size when training=5
           
better (future) single function controlled by batch_size :
batch 1:
[[1 2 3 4 5]  [2 3 4 5 6] ...  [5 6 7 8 9] [6 7 8 9 10] ...  up to batch_size sequences
batch 2:
[[6 7 8 9 10].... continue sequences, stop when runs out
etc

batch_size large or inf: all in one batch - pass batch_size = sequences to model.fit
batch_size = None: same, pass batch_size = None to model.fit 
batch_size < timesteps: error
batch_size = anything else: pass batch_size to model.fit

batch_size = inf is same as previous batch_type = 1
batch_size = timesteps is same as previous batch_type = 2







