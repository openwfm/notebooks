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

