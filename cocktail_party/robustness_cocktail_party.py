import numpy as np
from pydub import AudioSegment
import os
print(os.path.exists("gray.mp3"))
print(os.path.exists("goethe.mp3"))
print(os.path.exists("kordian.mp3"))

# Load the MP3 files
audio_gray = AudioSegment.from_mp3("gray.mp3")
audio_goethe = AudioSegment.from_mp3("goethe.mp3")
audio_kordian = AudioSegment.from_mp3("kordian.mp3")

# Convert each to a NumPy array of samples
data_gray = np.array(audio_gray.get_array_of_samples(), dtype=np.float32)
data_goethe = np.array(audio_goethe.get_array_of_samples(), dtype=np.float32)
data_kordian = np.array(audio_kordian.get_array_of_samples(), dtype=np.float32)

# Ensure they are all the same length
# If not, you might need to truncate or pad them.
min_length = min(len(data_gray), len(data_goethe), len(data_kordian))
data_gray = data_gray[:min_length]
data_goethe = data_goethe[:min_length]
data_kordian = data_kordian[:min_length]

n = len(data_gray)  # Number of samples
d = 3  # Number of signals (3 audio files)
MC_SAM_LENS = [10, 20, 50, 100]

available_conf_types = ["two_groups", "common_corruptor", "talking_pairs", "multiplicative"]
conf_strengths = [0.001, 0.02]

# Stack into a single S matrix of shape (N, 3)
S_pure = np.column_stack((data_gray, data_goethe, data_kordian))

# Define the matrix A
A = np.array([[1,  2, 0],
              [0, 1,  2],
              [0, 0, 1]])

A_inv = np.linalg.inv(A)


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from RICA import Optimizer, ContrastCalculator, get_rel_err, M_IplusE
from continuity_moduli import ContinuityModuli
from sample_different_S import SignalGenerator

from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


rica_rel_err_hist = []
fastica_rel_err_hist = []
delta_hist = []
eps0_hist = []
rica_abs_err_hist = []
fast_ica_abs_err_hist = []
# sobi_abs_err_hist = []
# sobi_rel_err_hist = []
bound1_hist = []
bound2_hist = []

for MC_SAM_LEN in MC_SAM_LENS:
    for conf_type in available_conf_types:
        for conf_strength in conf_strengths:
            print(f"Generating confounded signal with type: {conf_type}, strength: {conf_strength}")

            contrastcalc = ContrastCalculator(n,d,MC_SAM_LEN, False)
            # Generate the confounded signal
            signalgenerator = SignalGenerator()
            S = signalgenerator.confound_pure_signal(S_pure, conf_type, conf_strength)
            delta = contrastcalc.compute_delta(S)

        
            # Compute X = S * A^T
            X = S @ A.T  # Matrix multiplication
            
            
            optimizer = Optimizer(n, d,  MC_SAM_LEN, False)
            I_x = optimizer.RICA(X)
            M_rica, E_rica = M_IplusE(I_x, A)

            rica_abs_err = np.linalg.norm(E_rica, 'fro')
            rel_error_rica = get_rel_err(I_x , A_inv, *M_IplusE(I_x, A))

            
            # Compare with FastICA
            ica = FastICA(n_components=d, random_state=42)
            X_transformed = ica.fit_transform(X)
            W_fastica = ica.components_
            
            rel_error_fastica = get_rel_err(W_fastica, A_inv, *M_IplusE(W_fastica, A))
            M_fast_ica, E_fast_ica = M_IplusE(W_fastica, A)
            fast_ica_abs_err = np.linalg.norm(E_fast_ica, 'fro')

            continuity_moduli = ContinuityModuli(true_source = S, mixed_signal=X, A = A, k_0=4, MC_SAM_LEN=MC_SAM_LEN)
            eps0, c_1, c_2 = continuity_moduli.test_blind_inversion_thm(I_x)
            M_rica, E_rica = M_IplusE(I_x, A)
            
            bound2_value = c_2 * delta/(1 - delta)
            bound1_value = c_1 * delta/(1 - delta)


            # Compare contrasts for different methods using A at the end
            print("\n\n\n", "-"*10, "comparing RICA and FastICA ", "-"*10)
            print("I_X @ A : \n", I_x @ A)
            print("W_fastica @ A: \n", W_fastica @ A)

            print("Relative error for RICA: ", rel_error_rica)
            print("Relative error for FastICA: ", rel_error_fastica)

            rica_abs_err_hist.append(rica_abs_err)
            rica_rel_err_hist.append(rel_error_rica)
            bound1_hist.append(bound1_value)
            bound2_hist.append(bound2_value)
            delta_hist.append(delta)
            eps0_hist.append(eps0)

            fastica_rel_err_hist.append(rel_error_fastica)
            fast_ica_abs_err_hist.append(fast_ica_abs_err)
            

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
# sobi_abs_sorted = np.array(sobi_abs_err_hist)[sorted_indices]
# sobi_rel_sorted = np.array(sobi_rel_err_hist)[sorted_indices]
# Create filter mask for indices where delta < eps0
filter_mask = delta_sorted < eps0_hist_sorted

# Apply the filter to all arrays
delta_sorted_masked = delta_sorted[filter_mask]
bound1_sorted_masked = bound1_sorted[filter_mask]
rica_abs_sorted_masked = rica_abs_sorted[filter_mask]
fast_ica_abs_sorted_masked = fast_ica_abs_sorted[filter_mask]
# sobi_abs_sorted_masked = sobi_abs_sorted[filter_mask]
# sobi_rel_sorted_masked = sobi_rel_sorted[filter_mask]
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
# ax2.plot(delta_sorted_masked, sobi_abs_sorted_masked,
#             label='SOBI Absolute Error', marker='d', linewidth=2.5,
#             markersize=8, color=colors[3], markerfacecolor='white',
#             markeredgewidth=2, markeredgecolor=colors[3])

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
# sobi_rel_sorted_masked = np.array(sobi_rel_err_hist)[sorted_indices][filter_mask]
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
# ax3.plot(delta_sorted_masked, sobi_rel_sorted_masked,
#             label='SOBI Relative Error', marker='d', linewidth=2.5,
#             markersize=8, color=colors[3], markerfacecolor='white',
#             markeredgewidth=2, markeredgecolor=colors[3])

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
# sobi_abs_sorted = np.array(sobi_abs_err_hist)[sorted_indices]
rica_rel_sorted = np.array(rica_rel_err_hist)[sorted_indices]
fastica_rel_sorted = np.array(fastica_rel_err_hist)[sorted_indices]
# sobi_rel_sorted = np.array(sobi_rel_err_hist)[sorted_indices]

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
# ax4.plot(delta_sorted, sobi_rel_sorted,
#             label='SOBI Relative Error', marker='d', linewidth=2.5,
#             markersize=8, color=colors[3], markerfacecolor='white',
#             markeredgewidth=2, markeredgecolor=colors[3])
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
# ax5.plot(delta_sorted, sobi_abs_sorted,
#             label='SOBI Absolute Error', marker='d', linewidth=2.5,
#             markersize=8, color=colors[3], markerfacecolor='white',
#             markeredgewidth=2, markeredgecolor=colors[3])
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


