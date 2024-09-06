import numpy as np
from moisture_rnn import staircase_spatial


# Test Function Notes:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Testing for a single Functions:
# 1. Helper function to create data objects, eg X & y
# 2. Function to print X & y in human readable way
# 3. Call target function eg staircase_spatial to create XX, yy, n_seqs
# 4. Print previous output in human readable way (as similar to images in Overleaf FMDA w Recurrent Notebooks - Example 4)
# 5. Compare to expectations* (eg see Overleaf FMDA w Recurrent Notebooks - Example 4) human or code



def staircase_spatial_test(total_time_steps, features, batch_size, timesteps, n_locs = 10):
    """
    Parameters:
    -----------

    
    
    Run staircase_spatial with:
    -----------
    X : list of numpy arrays of length n_locs
        A list where each element is a numpy array containing features for a specific location. The shape of each array is `(total_time_steps, features)`.

    y : list of numpy arrays of length n_locs
        A list where each element is a numpy array containing the target values for a specific location. The shape of each array is `(total_time_steps,)`.   


    Returns:
    -----------
    
    """

    
    # 
    print("staircase_spatial_test inputs")
    print(f"total_time_steps: {total_time_steps}")
    print(f"features: {features}")
    print(f"batch_size: {batch_size}")
    print(f"timesteps: {timesteps}")
    print(f"n_locs: {n_locs}")


    # Create test arrays
    X = []
    y = []

    for i in range(0, n_locs):
        Xi = np.arange(i, i+total_time_steps)
        yi = np.arange(i, i+total_time_steps)
        X.append(Xi)
        y.append(yi)


    print(X)
    print(y)





