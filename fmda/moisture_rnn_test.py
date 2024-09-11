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


# DRAFT UTIL FOR PRINTING:
# rounded_padded_X0 = round_and_pad(X[0,:,:])
# rounded_padded_X1 = round_and_pad(X[params['batch_size']-1,:,:])
# sample_0_str = f"[{' '.join(rounded_padded_X0[0])}]"
# sample_1_str = f"[{' '.join(rounded_padded_X1[0])}]"
# row_length = len(f"{sample_0_str}  ...  {sample_1_str}")


# # Manually adjust padding to match the row length
# total_header_length = len("Sample 0") + len(f"Sample {params['batch_size']-1}") + len("...")
# spaces_needed = row_length - total_header_length
# padding = " " * (spaces_needed // 2)

# print("Batch 0:")
# print(f"Sample 0{padding}...{padding}Sample {params['batch_size']-1}")

# for row0, row1 in zip(rounded_padded_X0, rounded_padded_X1):
#     print(f"[{' '.join(row0)}]  ...  [{' '.join(row1)}]")



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





