import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import random
from scipy.integrate import odeint  # (if you wish to use an ODE solver)

from RICA import Optimizer, ContrastCalculator
from sample_different_S import SignalGenerator

np.set_printoptions(precision=2)


def main():
    """
    Main function to execute the perturbation and delta calculations.
    """
    
    nr_of_mc_samples_for_moment_computation = 1_000_000
    
    d = 3
    MC_SAM_LEN = 15
    # nr_of_experiments = 4
    
    available_conf_type = ["two_groups", "common_corruptor", "talking_pairs", "multiplicative"]
    conf_strenghts = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025] 
    ts_type = 'MA'  # Type of time series to sample, e.g., 'iid', 'gumbelMA', 'OU', 'MA'
    delta_for_two_groups = []
    delta_for_common_corruptor = []
    delta_for_talking_pairs = []
    delta_for_multiplicative = []
    
    signal_generator = SignalGenerator()
    # 1 
    # S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='iid')
    # 2
    # S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='gumbelMA') # easy to get under eps0
    # # 3
    # S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='OU')
    # length_of_S = S.shape[0]
    # burnin = int(0.1 * length_of_S)  # 10% burnin
    # S = S[burnin:]
    # # EXAMPLE 4: MA
    S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='MA')
         
    S_pure = S
    n = S.shape[0]
    contrastcalculator = ContrastCalculator(n, d, MC_SAM_LEN, check_identifiability_criteria=False)

    delta_pure = contrastcalculator.compute_delta(S_pure)
    print("delta for pure signal: ", delta_pure)
    # --- confound the pure signal ----------------------------------------
    for con_type in available_conf_type:
        
        delta_hist = []
        for conf_str in conf_strenghts:
            # --- confound the pure signal ----------------------------------------
            # Create a fresh copy for each confounding operation
            S_copy = S_pure.copy()
            S_corrupted = signal_generator.confound_pure_signal(S_copy, con_type, conf_str)
            contrastcalculator = ContrastCalculator(n, d, MC_SAM_LEN, check_identifiability_criteria=False)

            delta = contrastcalculator.compute_delta(S_corrupted)
            print("delta for conf type:", con_type, " and conf strength:", conf_str, " = ", delta)
            delta_hist.append(delta)

        if con_type == "two_groups":
            delta_for_two_groups = delta_hist
        elif con_type == "common_corruptor":
            delta_for_common_corruptor = delta_hist
        elif con_type == "talking_pairs":
            delta_for_talking_pairs = delta_hist
        elif con_type == "multiplicative":
            delta_for_multiplicative = delta_hist
        print("delta for conf type:", con_type, " = ", delta)

    # --- plot the results ----------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(conf_strenghts, delta_for_two_groups, label='Two Groups', marker='o')
    plt.plot(conf_strenghts, delta_for_common_corruptor, label='Common Corruptor', marker='o')
    plt.plot(conf_strenghts, delta_for_talking_pairs, label='Talking Pairs', marker='o')
    plt.plot(conf_strenghts, delta_for_multiplicative, label='Multiplicative', marker='o')
    plt.axhline(y=delta_pure, color='r', linestyle='--', label='Pure Signal Delta')
    plt.xlabel('Confounder Strength')
    plt.ylabel('Delta Value')
    plt.title(f'Delta Values for Different Confounder Types, source type: {ts_type}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("delta_vs_confounder_strength.png")
    plt.show()

if __name__ == "__main__":
    main()