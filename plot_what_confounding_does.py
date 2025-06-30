from RICA import ContrastCalculator
from sample_different_S import SignalGenerator
import matplotlib.pyplot as plt
import numpy as np


def main():
    n = 15_000_000  # number of time points
    d = 3     # number of channels
    ts_type = 'ARMA'  # type of time series to sample: 'iid
    corruption_type = 'multiplicative'  # type of corruption to apply
    # ts_type = 'iid'
    # ts_type = 'gammaMA'
    index_to_plot = 1000  # Show first 1000 samples
    S_original = SignalGenerator.sample_s(d, n, ts_type)
    contrastcalculator = ContrastCalculator(n,d, 15, False)
    delta_original = contrastcalculator.compute_delta(S_original)
    print("S shape: ", S_original.shape)
    
    # Create signal generator instance for confounding
    signal_gen = SignalGenerator()
    
    # Create slightly confounded signal (low corruption)
    S_slight = S_original.copy()
    S_slight = signal_gen.confound_pure_signal(S_slight, corruption_type, 0.3)
    delta_slight = contrastcalculator.compute_delta(S_slight)
    # Create more confounded signal (higher corruption)
    S_heavy = S_original.copy()
    S_heavy = signal_gen.confound_pure_signal(S_heavy, corruption_type, 0.7)
    delta_heavy = contrastcalculator.compute_delta(S_heavy)

    print(" deltas: ", delta_original, delta_slight, delta_heavy)
    # Create the three subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original signal (left)
    for i in range(d):
        axes[0].plot(S_original[:index_to_plot, i], label=f'Channel {i+1}')
    axes[0].set_title(f'Original {ts_type} Signal')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Add delta text box
    axes[0].text(0.02, 0.98, f'δ = {delta_original:.4f}',
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           alpha=0.8))
    
    # Plot slightly confounded signal (middle)
    for i in range(d):
        axes[1].plot(S_slight[:index_to_plot, i], label=f'Channel {i+1}')
    axes[1].set_title(f'Slightly Confounded Signal (0.3, "{corruption_type}")')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # Add delta text box
    axes[1].text(0.02, 0.98, f'δ = {delta_slight:.4f}',
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           alpha=0.8))
    
    # Plot heavily confounded signal (right)
    for i in range(d):
        axes[2].plot(S_heavy[:index_to_plot, i], label=f'Channel {i+1}')
    axes[2].set_title(f'More Confounded Signal (0.7, "{corruption_type}")')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    # Add delta text box
    axes[2].text(0.02, 0.98, f'δ = {delta_heavy:.4f}',
                 transform=axes[2].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return


if __name__ == "__main__":
    main()
