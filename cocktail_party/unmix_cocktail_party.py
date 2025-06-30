import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from RICA import Optimizer, get_rel_err, M_IplusE
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

from sample_different_S import SignalGenerator


def main():
    # Load the signal from file
    X = np.load("observation_cocktail_party.npy")
    n, d = X.shape

    # use only the first n = 500000 samples
    n = 3_000_000
    X = X[:n, :]
    
    print("loaded X. X shape: ", X.shape)
    t = np.linspace(0, 1, n)

    # Define A_true only for testing different methods at the end
    A_true = np.array([[1,  2, 0],
              [0, 1,  2],
              [0, 0, 1]])
    
    A_inv = np.linalg.inv(A_true)

    optimizer = Optimizer(n,d, MC_SAM_LEN=50)
    I_x = optimizer.RICA(X)
    
    
    # Compare with FastICA
    ica = FastICA(n_components=d, random_state=42)
    X_transformed = ica.fit_transform(X)
    W_fastica = ica.components_


    # Compare contrasts for different methods using A_true at the end
    print("\n\n\n", "-"*10, "comparing the contrasts for different methods", "-"*10)
    print("I_X @ A : \n", I_x @ A_true)
    print("W_fastica @ A: \n", W_fastica @ A_true)
    
    rel_erorr_rica = get_rel_err(I_x , A_inv, *M_IplusE(I_x, A_true))
    rel_error_fastica = get_rel_err(W_fastica, A_inv, *M_IplusE(W_fastica, A_true))
    print("Relative error for RICA: ", rel_erorr_rica)
    print("Relative error for FastICA: ", rel_error_fastica)

    # compute the estimated signals S_hat_1 =  X @ I_x.T, S_hat_2  = X @ W_fastica.T, S_hat_3 = X @ I_x_GD.T and save them to npy files
    X_full = np.load("observation_cocktail_party.npy")
    S_hat_1 = X_full @ I_x.T
    S_hat_2 = X_full @ W_fastica.T
    
    np.save("S_hat_rica.npy", S_hat_1)
    np.save("S_hat_fast.npy", S_hat_2)



if __name__ == "__main__":
    main()
