import numpy as np
import jax
from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import FastICA
from RICA import Optimizer, ContrastCalculator
from continuity_moduli import ContinuityModuli
from sample_different_S import SignalGenerator
from SOBI import sobi_algo

from RICA import M_IplusE, get_rel_err

try:
    devices = jax.devices()
    print("Available devices:", devices)
    # Print more detailed device info
    print("Device details:")
    for i, d in enumerate(devices):
        print(f"  Device {i}: {d}")
        
    # Check if we're actually using Metal
    backend = jax.lib.xla_bridge.get_backend()
    print(f"Active backend: {backend.platform}")
except Exception as e:
    print(f"Error getting device information: {e}")
    print("Falling back to CPU")
    config.update("jax_platform_name", "cpu")
    print("Using CPU for JAX")
    

def main():
    np.random.seed(0)
    d, nr_of_mc_samples_for_moment_computation = 3, 1_000_000
    MC_SAM_LEN = 15
    nr_of_experiments = 3

    available_conf_types = ["two_groups", "common_corruptor", "talking_pairs", "multiplicative"]
    conf_strengths = [0.001, 0.005, 0.01, 0.02]

    rica_rel_err_hist = []
    fastica_rel_err_hist = []
    delta_hist = []
    eps0_hist = []
    rica_abs_err_hist = []
    fast_ica_abs_err_hist = []
    sobi_abs_err_hist = []
    sobi_rel_err_hist = []
    bound1_hist = []
    bound2_hist = []

    A = jnp.array(
         [[-0.37842,     0.91451844, -0.14301863],
        [-0.69043073, -0.3817858,  -0.61444691],
        [-0.61652552, -0.13377454,  0.77588701]])
    
    # A = np.array(
    #     [[1, 0.5, 0],
    #      [0, 1, 0.5],
    #      [0, 0.5, 1]])
    # A = np.random.rand(d, d)
    A_inv = np.linalg.inv(A)
    print(" A_inverse: \n", A_inv)
    print(" cond of A_inv: ", np.linalg.cond(A_inv))

    for con_type in available_conf_types:
        for conf_str in conf_strengths:
           
            for e_nr in range(nr_of_experiments):
                print(f"\n\n\nfor experiment nr = {e_nr + 1} with MC_LEN = {MC_SAM_LEN}, confounding type: {con_type}, confounding strength: {conf_str}")
                signal_generator = SignalGenerator()
                skip_experiment = False
                
                # --- generate sources and their mixtures -------------------------------
                if e_nr % 4 == 0:
                    S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='iid')
                elif e_nr % 4 == 1:
                    S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='gumbelMA') # easy to get under eps0
                elif e_nr % 4 == 2:
                    # EXAMPLE 3: OU process (requires the preprocessing step)
                    S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='OU')
                    length_of_S = S.shape[0]
                    burnin = int(0.1 * length_of_S)  # 10% burnin
                    S = S[burnin:]
                else:
                    # EXAMPLE 4: MA
                    S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='MA')
                    

                length_of_S = S.shape[0]
                S = signal_generator.confound_pure_signal(S, con_type, conf_str)
                print("confounding type: ", con_type, " with strength: ", conf_str)
                
                S = jnp.array(S)
                # S = sample_S(d, nr_of_mc_samples_for_moment_computation * MC_SAM_LEN)
                X = S @ A.T                                           # mix
                                                    
                # ------------------------------------------------------------------------
                contrastcalculator = ContrastCalculator(length_of_S, d, MC_SAM_LEN, check_identifiability_criteria=False)
                
                # Handle potential singular matrix error in delta computation
                try:
                    delta_for_this_MC_SAM_LEN = contrastcalculator.compute_delta(S)
                except np.linalg.LinAlgError as e:
                    warnings.warn(f"Experiment {e_nr + 1}: Singular matrix encountered in delta computation. Skipping this experiment. Error: {e}")
                    skip_experiment = True
                
                if skip_experiment:
                    continue


                ricaoptimizer = Optimizer(length_of_S, d, MC_SAM_LEN, verbose=False)
                
                # Handle potential errors in RICA optimization
                try:
                    I_x = ricaoptimizer.RICA(X)
                    M_rica, E_rica = M_IplusE(I_x, A)
                    rica_abs_err = jnp.linalg.norm(E_rica, 'fro')
                    rica_rel_err = get_rel_err(I_x, A_inv, M_rica, E_rica)
                except (np.linalg.LinAlgError, ValueError) as e:
                    warnings.warn(f"Experiment {e_nr + 1}: Error in RICA optimization. Skipping this experiment. Error: {e}")
                    skip_experiment = True
                
                if skip_experiment:
                    continue

                # Handle potential errors in continuity moduli computation
                try:
                    continuity_moduli = ContinuityModuli(true_source = S, mixed_signal=X, A = A, k_0=1.1, MC_SAM_LEN=MC_SAM_LEN)
                    eps0, c_1, c_2 = continuity_moduli.test_blind_inversion_thm(I_x)
                except (np.linalg.LinAlgError, ValueError) as e:
                    warnings.warn(f"Experiment {e_nr + 1}: Error in continuity moduli computation. Skipping this experiment. Error: {e}")
                    skip_experiment = True
                if delta_for_this_MC_SAM_LEN > 1.0:
                    warnings.warn(f"Experiment {e_nr + 1}: delta(S) = {delta_for_this_MC_SAM_LEN} is larger than 1.0, skipping this experiment.")
                    skip_experiment = True
                if skip_experiment:
                    continue
                
                # Store values for plotting
                bound1_value = c_1 * delta_for_this_MC_SAM_LEN/(1 - delta_for_this_MC_SAM_LEN)
                bound2_value = c_2 * delta_for_this_MC_SAM_LEN/(1 - delta_for_this_MC_SAM_LEN)

                print("delta(S) =", delta_for_this_MC_SAM_LEN, " should be smaller than eps0 =", eps0)
                print("relative error of RICA:", rica_rel_err)
                # print("I_x @ A:\n", I_x @ A)  
                print("first (abs) claim of theorem 4.3: ", jnp.linalg.norm(E_rica, 'fro'), " <= ", c_1 * delta_for_this_MC_SAM_LEN/(1 - delta_for_this_MC_SAM_LEN))
                print("second (rel) claim of theorem 4.3: ", rica_rel_err, " <= ", c_2 * delta_for_this_MC_SAM_LEN/(1 - delta_for_this_MC_SAM_LEN))
                
                if delta_for_this_MC_SAM_LEN < eps0 and jnp.linalg.norm(E_rica, 'fro') > c_1 * delta_for_this_MC_SAM_LEN/(1 - delta_for_this_MC_SAM_LEN):
                    warnings.warn("The first claim of Theorem 4.3 is violated!")
                if delta_for_this_MC_SAM_LEN < eps0 and rica_rel_err > c_2 * delta_for_this_MC_SAM_LEN/(1 - delta_for_this_MC_SAM_LEN):
                    warnings.warn("The second claim of Theorem 4.3 is violated!")
                

                # compare to FastICA
                try:
                    ica    = FastICA(n_components=d, max_iter=100_000, random_state=0)
                    S_fastica = ica.fit_transform(X)  
                    W_ica  = ica.components_       

                    M_fast_ica, E_fast_ica = M_IplusE(W_ica, A)
                    fast_ica_abs_err = jnp.linalg.norm(E_fast_ica, 'fro')
                    fastica_rel_err = get_rel_err(W_ica, A_inv, M_fast_ica, E_fast_ica)
                    print("relative error of FastICA:", fastica_rel_err)

                except (np.linalg.LinAlgError, ValueError) as e:
                    warnings.warn(f"Experiment {e_nr + 1}: Error in FastICA computation. Skipping this experiment. Error: {e}")
                    skip_experiment = True
                
                if skip_experiment:
                    continue

                # print("W_ica @ A:\n", W_ica @ A)

                # --- SOBI ---
                try:
                    _, _, W_sobi = sobi_algo(X.T, num_lags=MC_SAM_LEN, eps=1e-6, random_order=True)

                    print("SOBI results:")
                    # print("W_sobi: ", W_sobi)
                    
                    M_sobi, E_sobi = M_IplusE(W_sobi, A)
                    sobi_rel_err = get_rel_err(W_sobi, A_inv, *M_IplusE(W_sobi, A))
                    sobi_abs_err = jnp.linalg.norm(E_sobi, 'fro')
                except (np.linalg.LinAlgError, ValueError) as e:
                    warnings.warn(f"Experiment {e_nr + 1}: Error in SOBI computation. Skipping this experiment. Error: {e}")
                    skip_experiment = True
                
                if skip_experiment:
                    continue

                print("SOBI relative error:", sobi_rel_err)
                # print("W_sobi @ A:\n", W_sobi @ A)

                # print("SOBI absolute error:", sobi_abs_err)
                rica_rel_err_hist.append(rica_rel_err)
                rica_abs_err_hist.append(rica_abs_err)
                
                bound1_hist.append(bound1_value)
                bound2_hist.append(bound2_value)
                
                delta_hist.append(delta_for_this_MC_SAM_LEN)
                eps0_hist.append(eps0)

                fastica_rel_err_hist.append(fastica_rel_err)
                fast_ica_abs_err_hist.append(fast_ica_abs_err)

                sobi_rel_err_hist.append(sobi_rel_err)
                sobi_abs_err_hist.append(sobi_abs_err)
                print("\n\n")

    total_nr_of_experiments = len(delta_hist)
    

    #### Professional plotting configuration ###
    # Set up publication-ready plotting style
    plt.style.use('seaborn-v0_8-whitegrid')  # Professional base style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.9,
        'figure.figsize': [12, 8],
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linewidth': 0.8,
        'axes.linewidth': 1.2,
        'axes.edgecolor': 'black',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'text.usetex': False,  # Set to True if LaTeX is available
        'mathtext.fontset': 'cm'
    })
    
    # Define professional color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # First plot: Experiment number vs δ and ε₀
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    line1 = ax1.plot(range(1, total_nr_of_experiments + 1), delta_hist, 
                     label=r'$\delta(S)$', marker='o', linewidth=2.5, 
                     markersize=8, color=colors[0], markerfacecolor='white', 
                     markeredgewidth=2, markeredgecolor=colors[0])
    line2 = ax1.plot(range(1, total_nr_of_experiments + 1), eps0_hist, 
                     label=r'$\epsilon_0$', marker='s', linewidth=2.5, 
                     markersize=8, color=colors[1], markerfacecolor='white', 
                     markeredgewidth=2, markeredgecolor=colors[1])
    
    ax1.set_title(r'IC-defect: $\delta(S)$ and $\epsilon_0$ vs Experiment Number', 
                  fontweight='bold', pad=20)
    ax1.set_xlabel('Experiment Number', fontweight='bold')
    ax1.set_ylabel('Parameter Value', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Set integer ticks for x-axis
    ax1.set_xticks(range(1, total_nr_of_experiments + 1))
    
    plt.tight_layout()
    plt.savefig('ic-defect.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Second plot: Theorem 4.3 Claim 1 - Absolute Error Bounds
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Sort data by delta for smooth lines 
    sorted_indices = np.argsort(delta_hist)
    delta_sorted = np.array(delta_hist)[sorted_indices]
    bound1_sorted = np.array(bound1_hist)[sorted_indices]
    rica_abs_sorted = np.array(rica_abs_err_hist)[sorted_indices]
    fast_ica_abs_sorted = np.array(fast_ica_abs_err_hist)[sorted_indices]
    eps0_hist_sorted = np.array(eps0_hist)[sorted_indices]
    sobi_abs_sorted = np.array(sobi_abs_err_hist)[sorted_indices]
    sobi_rel_sorted = np.array(sobi_rel_err_hist)[sorted_indices]
    # Create filter mask for indices where delta < eps0
    filter_mask = delta_sorted < eps0_hist_sorted

    # Apply the filter to all arrays
    delta_sorted_masked = delta_sorted[filter_mask]
    bound1_sorted_masked = bound1_sorted[filter_mask]
    rica_abs_sorted_masked = rica_abs_sorted[filter_mask]
    fast_ica_abs_sorted_masked = fast_ica_abs_sorted[filter_mask]
    sobi_abs_sorted_masked = sobi_abs_sorted[filter_mask]
    sobi_rel_sorted_masked = sobi_rel_sorted[filter_mask]
    # Plot the theoretical bound and the errors

    ax2.plot(delta_sorted_masked, bound1_sorted_masked,
             label=r'Theoretical Bound: $c_1 \frac{\delta}{1-\delta}$',
             marker='o', linewidth=3, markersize=8, color=colors[0],
             markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[0])
    ax2.plot(delta_sorted_masked, rica_abs_sorted_masked, 
             label='RICA Absolute Error', marker='^', linewidth=2.5, 
             markersize=8, color=colors[1], markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor=colors[1])
    ax2.plot(delta_sorted_masked, fast_ica_abs_sorted_masked,
             label='FastICA Absolute Error', marker='s', linewidth=2.5,
             markersize=8, color=colors[2], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=colors[2])
    ax2.plot(delta_sorted_masked, sobi_abs_sorted_masked,
             label='SOBI Absolute Error', marker='d', linewidth=2.5,
             markersize=8, color=colors[3], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=colors[3])

    ax2.set_title(r'Theorem 4.3 Claim 1: Absolute Error vs $\delta(S)$',
                  fontweight='bold', pad=20)
    ax2.set_xlabel(r'$\delta(S)$', fontweight='bold')
    ax2.set_ylabel('Absolute Error', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization of errors
    
    plt.tight_layout()
    plt.savefig('theorem_4_3_claim_1.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Third plot: Theorem 4.3 Claim 2 - Relative Error Bounds
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    rica_rel_sorted_masked = np.array(rica_rel_err_hist)[sorted_indices][filter_mask]
    fastica_rel_sorted_masked = np.array(fastica_rel_err_hist)[sorted_indices][filter_mask]
    sobi_rel_sorted_masked = np.array(sobi_rel_err_hist)[sorted_indices][filter_mask]
    bound2_sorted_masked = np.array(bound2_hist)[sorted_indices][filter_mask]

    ax3.plot(delta_sorted_masked, bound2_sorted_masked,
             label=r'Theoretical Bound: $c_2 \frac{\delta}{1-\delta}$',
             marker='o', linewidth=3, markersize=8, color=colors[0],
             markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[0])
    ax3.plot(delta_sorted_masked, rica_rel_sorted_masked,
             label='RICA Relative Error', marker='^', linewidth=2.5,
             markersize=8, color=colors[1], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=colors[1])
    ax3.plot(delta_sorted_masked, fastica_rel_sorted_masked,
             label='FastICA Relative Error', marker='s', linewidth=2.5,
             markersize=8, color=colors[2], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=colors[2])
    ax3.plot(delta_sorted_masked, sobi_rel_sorted_masked,
             label='SOBI Relative Error', marker='d', linewidth=2.5,
             markersize=8, color=colors[3], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=colors[3])

    ax3.set_title(r'Theorem 4.3 Claim 2: Relative Error vs $\delta(S)$', 
                  fontweight='bold', pad=20)
    ax3.set_xlabel(r'$\delta(S)$', fontweight='bold')
    ax3.set_ylabel('Relative Error', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale for better visualization of errors
    
    plt.tight_layout()
    plt.savefig('theorem_4_3_claim_2.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    # ++++++ plot without the mask ++++++
    bound1_sorted = np.array(bound1_hist)[sorted_indices]
    bound2_sorted = np.array(bound2_hist)[sorted_indices]
    rica_abs_sorted = np.array(rica_abs_err_hist)[sorted_indices]
    fast_ica_abs_sorted = np.array(fast_ica_abs_err_hist)[sorted_indices]
    sobi_abs_sorted = np.array(sobi_abs_err_hist)[sorted_indices]
    rica_rel_sorted = np.array(rica_rel_err_hist)[sorted_indices]
    fastica_rel_sorted = np.array(fastica_rel_err_hist)[sorted_indices]
    sobi_rel_sorted = np.array(sobi_rel_err_hist)[sorted_indices]
    
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.plot(delta_sorted, bound2_sorted, 
             label=r'Theoretical Bound: $c_2 \frac{\delta}{1-\delta}$', 
             marker='o', linewidth=3, markersize=8, color=colors[0],
             markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[0])
    ax4.plot(delta_sorted, rica_rel_sorted, 
             label='RICA Relative Error', marker='^', linewidth=2.5, 
             markersize=8, color=colors[1], markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor=colors[1])
    ax4.plot(delta_sorted, fastica_rel_sorted,
                label='FastICA Relative Error', marker='s', linewidth=2.5, 
                markersize=8, color=colors[2], markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor=colors[2])
    ax4.plot(delta_sorted, sobi_rel_sorted,
                label='SOBI Relative Error', marker='d', linewidth=2.5,
                markersize=8, color=colors[3], markerfacecolor='white',
                markeredgewidth=2, markeredgecolor=colors[3])
    ax4.set_title(r' Relative Error vs $\delta(S)$, no guarantees', 
                  fontweight='bold', pad=20)
    ax4.set_xlabel(r'$\delta(S)$', fontweight='bold')
    ax4.set_ylabel('Relative Error', fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')  # Log scale for better visualization of errors
    plt.tight_layout()
    plt.savefig('theorem_4_3_claim_2_without_mask.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    # Plot the abs errors withoiut the mask
    ax5.plot(delta_sorted, bound1_sorted,
                label=r'Theoretical Bound: $c_1 \frac{\delta}{1-\delta}$',
                marker='o', linewidth=3, markersize=8, color=colors[0],
                markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[0])
    ax5.plot(delta_sorted, rica_abs_sorted,
                label='RICA Absolute Error', marker='^', linewidth=2.5,
                markersize=8, color=colors[1], markerfacecolor='white',
                markeredgewidth=2, markeredgecolor=colors[1])
    ax5.plot(delta_sorted, fast_ica_abs_sorted,
                label='FastICA Absolute Error', marker='s', linewidth=2.5,
                markersize=8, color=colors[2], markerfacecolor='white',
                markeredgewidth=2, markeredgecolor=colors[2])
    ax5.plot(delta_sorted, sobi_abs_sorted,
                label='SOBI Absolute Error', marker='d', linewidth=2.5,
                markersize=8, color=colors[3], markerfacecolor='white',
                markeredgewidth=2, markeredgecolor=colors[3])
    ax5.set_title(r' Absolute Error vs $\delta(S)$, no guarantees',
                  fontweight='bold', pad=20)
    ax5.set_xlabel(r'$\delta(S)$', fontweight='bold')
    ax5.set_ylabel('Absolute Error', fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')  # Log scale for better visualization of errors
    plt.tight_layout()
    plt.savefig('theorem_4_3_claim_1_without_mask.pdf', dpi=300, bbox_inches='tight')
    plt.show()


    # Summary statistics table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Metric':<25} {'RICA':<15} {'FastICA':<15} {'Improvement':<15}")
    print("-"*80)
    
    mean_rica_rel = np.mean(rica_rel_err_hist)
    mean_fastica_rel = np.mean(fastica_rel_err_hist)
    mean_rica_abs = np.mean(rica_abs_err_hist)
    mean_fastica_abs = np.mean(fast_ica_abs_err_hist)
    
    rel_improvement = (mean_fastica_rel - mean_rica_rel) / mean_fastica_rel * 100
    abs_improvement = (mean_fastica_abs - mean_rica_abs) / mean_fastica_abs * 100
    
    print(f"{'Mean Relative Error':<25} {mean_rica_rel:<15.6f} {mean_fastica_rel:<15.6f} {rel_improvement:<15.2f}%")
    print(f"{'Mean Absolute Error':<25} {mean_rica_abs:<15.6f} {mean_fastica_abs:<15.6f} {abs_improvement:<15.2f}%")
    print("="*80)
    
    print(" \n\n")
    print("RESULTS: \n\n")
    print("Delta values: ", delta_hist)

    print("RICA relative errors: ", rica_rel_err_hist)
    print("FastICA relative errors: ", fastica_rel_err_hist)
    print("bound1 values: ", bound1_hist)
    print("bound2 values: ", bound2_hist)
    print("eps0 values: ", eps0_hist)


if __name__ == "__main__":
    main()