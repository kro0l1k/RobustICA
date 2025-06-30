import numpy as np
# import iisignature
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import itertools
import random
from scipy.integrate import odeint  # (if you wish to use an ODE solver)
import warnings
from statsmodels.graphics.tsaplots import plot_acf

np.set_printoptions(precision=2)
# from signax_works_perf import Optimizer, ContrastCalculator  
class SignalGenerator:
    """
    Class for   generating synthetic signals.
    
    Also includes function for graduual mixing of channels. 
    """
    @staticmethod
    def sample_s(d, n, ts_type):
        if ts_type == 'iid':
            for k in range(d):
                if k == 0:
                    x = np.random.gamma(1, 1, n)
                elif k == 1:
                    x =  np.random.gamma(d, 3, n)
                elif k == 2:
                    x =  np.random.gumbel(0, 3, n)
                else:
                    x = np.random.gumbel(0, 3 * k, n)
                x = x - np.mean(x)
                X = x if k == 0 else np.vstack((X, x))
                
            S = X.T
        elif ts_type == 'OU':
            # get paremeters to generate OU process. theta= 1,2, ...,d
            theta = np.arange(1, d + 1) * 0.1
            mu = np.zeros(d)
            x0 = np.zeros(d)
            print(f" when sampling ou: theta: {theta}, mu: {mu}, x0: {x0}")
            S = SignalGenerator.sample_ou_vectorized(n, theta, mu, sigma=1.0, dt=1.0, x0=x0)
        elif ts_type == 'ARMA':
            if d != 3:
                raise ValueError("ARMA process is only implemented for d=3")
            
            p = 10
            a = np.random.rand(d, p) 
            b = np.random.rand(d, p)
            
            # clip the coefficients to be in [-0.1, 0.1]
            a = np.clip(a, -1/p, 1/p)
            b = np.clip(b, -1/p, 1/p)
            S = SignalGenerator.sample_ARMA(n, d, a, b)
        
        elif ts_type == 'MA':
            S_no_stack = SignalGenerator.sample_s(d, n, 'iid')
            # create a moving average process from iid noise
            S = np.zeros((n, d))
            for k in range(d):
                # create a moving average process with a window of 5
                S[:, k] = np.convolve(S_no_stack[:, k], np.ones(5)/5, mode='same')
                S[:, k] += np.random.normal(0, 0.1, n)
                S[:, k] -= np.mean(S[:, k])  # Center the signal around zero
        elif ts_type == 'gumbelMA':
            S_no_stack = np.random.gumbel(0, 1, (n, d))
            S_no_stack *= 10
            print(f" means and stds of S_no_stack: {np.mean(S_no_stack, axis=0)}, {np.std(S_no_stack, axis=0)}")

            # create a moving average process from Gumbel noise
            S = np.zeros((n, d))
            for k in range(d):
                # create a moving average process with a window of 5
                S[:, k] = np.convolve(S_no_stack[:, k], np.ones(5)/5, mode='same')
                S[:, k] += np.random.normal(0, 0.1, n)
                S[:, k] -= np.mean(S[:, k])

        else:
            raise NotImplementedError("time series type not implemented")
        return S # Return shape (n, d)

    def sample_ou_vectorized(n, theta, mu, sigma=1.0, dt=1.0, x0=None):
        """
        Samples a d-dimensional Ornstein–Uhlenbeck process using fully vectorized operations.
        This is the fastest implementation, especially for large n.

        Parameters:
            n (int): Number of time steps.
            theta (array-like): Array of mean-reversion rates (shape: (d,)).
            mu (array-like): Array of long-term means (shape: (d,)).
            sigma (scalar or array-like): Volatility parameter(s) (default=1.0).
            dt (float): Time increment (default=1.0).
            x0 (array-like): Initial state (if None, defaults to mu).
            
        Returns:
            np.ndarray: Array of shape (n, d) containing the sampled process.
        """
        theta = np.asarray(theta)
        mu = np.asarray(mu)
        
        if np.isscalar(sigma):
            sigma = sigma * np.ones_like(theta)
        else:
            sigma = np.asarray(sigma)

        d = theta.shape[0]
        
        # Pre-compute all constants
        alpha = np.exp(-theta * dt)  # Decay factor
        beta = mu * (1 - alpha)     # Mean adjustment
        var_factor = sigma**2 * (1 - alpha**2) / (2 * theta)  # Variance factor
        
        # Handle the case where theta might be very small (avoid division by zero)
        small_theta_mask = theta < 1e-10
        if np.any(small_theta_mask):
            var_factor[small_theta_mask] = sigma[small_theta_mask]**2 * dt
        
        # Generate all random increments at once
        innovations = (np.random.gumbel(1,1,(n-1, d)) - np.ones((n-1, d)) * np.sqrt(var_factor))

        # Initialize result
        S = np.zeros((n, d))
        if x0 is None:
            S[0] = np.zeros_like(mu)
        else:
            S[0] = np.asarray(x0)
        
        # Vectorized exact solution (when possible) or Euler-Maruyama
        for t in range(1, n):
            S[t] = alpha * S[t-1] + beta + innovations[t-1]
            
        return S
    
    def sample_ARMA(n, d, a, b):
        # generate ARMA(p,p) process with d dimensions
        # a and b are [d x p] matrices of coefficients
        p = a.shape[1]  # number of AR coefficients
        print(" p = ", p)
        scale = 1
        # Gaussian is not great for third-order expected signatures, so we use Gumbel
        e = np.random.gumbel(0, 1, (n + p, d))  
        # Center the noise around zero
        e -= np.mean(e, axis=0)  
        e *= scale  # can scale up if needed
        y = np.zeros((n + p, d))

        # generate ARMA process
        for i in range(p, n + p):
            past_y = y[i-p:i][::-1]  # shape (p, d)
            past_e = e[i-p:i][::-1]  # shape (p, d)
            
            ar_term = np.sum(past_y * a.T, axis=0)  # element-wise mult then sum over p
            ma_term = np.sum(past_e * b.T, axis=0)  # element-wise mult then sum over p
            y[i] = ar_term + ma_term + e[i]
        
        # remove the first p samples
        y = y[p:]
        return y

    def confound_pure_signal(self, S, conf_type, conf_strength):
        """
        Takes a (n,d) shaped signal S, with independent channels, and returns
        an (n,d) shaped signal S_conf in which the channels have been corrupted acc.
        to the conf_type. conf_strength [0,1] regulates the level of corruption.
        conf_strenght = 0 <- returns the original signal
        conf_strenght = 1 <- the channels are very corrupted. 
        conf_type can be one of the following:
        - "common_corruptor": a common 1-dim time series is sampled and mixed with every channel.
        - "talking_pairs": the first channel is mixed with the second, the third with the fourth, etc.
        - "two_groups": the channels are split into two groups. the first group is mixed with itself, the second group is mixed with itself.
        - "multiplicative": a random normal is sampled and multiplied with every channel.
        """
        n,d = S.shape
        
        if conf_strength > 1:
            print("conf strength should be <=1. using conf_strength = 1")
            conf_strength = 1
        if conf_strength < 0:
            print("conf_strength should be >=0. using conf_strength = 0")
            conf_strength = 0

        if conf_type == "common_corruptor":
            # sample a 1-dim time series of length n. then linearly mix every channel with it.
            conf_strength = 0.001 * conf_strength
            corruptor = np.random.gamma(n, 3, n)
            print("corruptor shape", corruptor.shape )
            
            for k in range(d):
                k_th_channel = S[:, k]
                k_th_channel = (conf_strength) * corruptor + (1 - conf_strength) * k_th_channel
                S[:, k] = k_th_channel

        elif conf_type == "talking_pairs":
            # we confound it with 0.001 * confstrenght because the mixing is so strong
            print("conf_type: talking_pairs")
            conf_strength = 0.4 * conf_strength
            if conf_strength == 0.5:
                print(" we shouldnt use conf_strenght = 0.5 for this corruption type as the neighbouring channels will look exactly the same")
                print("changing conf strenght tp 0.6")
                conf_strength = 0.6

            # confoud the first with the second, the third with the fourth, etc.
            for k in range(0,d - 1, 2): # can only do till d-1 so that we can use the k+1 channel 
                print(k, " in talking pairs") 
                
                k_th_channel = (conf_strength) * S[:, k+1] + (1 - conf_strength) * S[:, k]
                next_channel = (conf_strength) * S[:, k] + (1 - conf_strength) * S[:, k+1]
                
                S[:,k] = k_th_channel
                S[:,k+1] = next_channel
                
        elif conf_type == "two_groups":
            # conf_strength = 0.001 * conf_strength
            # split the channels into two groups. mix the channels within the groups.
            if conf_strength == 1:
                print("conf strenght should be <1. using conf_strength = 0.9")
                conf_strength = 0.9
            
            av1 = np.mean(S[:, :d//2], axis=1)
            av2 = np.mean(S[:, d//2:], axis=1)
            
            # plot the two averages            
            for k in range(d//2):
                S[:, k] = (conf_strength) * av1 + (1 - conf_strength) * S[:, k]
                
            for k in range(d//2, d):
                S[:, k] = (conf_strength) * av2 + (1 - conf_strength) * S[:, k]

        elif conf_type == "multiplicative":
            # generate a random normal and modulate the signal with it.
            noise = np.random.normal(1, 3 * conf_strength, n)
            ones = np.ones_like(noise)

            for k in range(d):
                S[:, k] = S[:, k] * (noise)
        else:
            # corruption not implemented.
            raise NotImplementedError("confounding type not implemented.")
        
        return S
    
def main():
    n = 10000
    d = 3
    ts_type = 'ARMA'  # type of time series to sample: 'iid'
    corruption_type = 'multiplicative'  # type of corruption to apply
    # ts_type = 'iid'
    # ts_type = 'gammaMA'
    
    index_to_plot = 1000  # Show first 1000 samples
    signal_gen = SignalGenerator()

    S_original = signal_gen.sample_s(d, n, ts_type)

    # Create slightly confounded signal (lower corruption)
    S_slight = S_original.copy()
    S_slight = signal_gen.confound_pure_signal(S_slight, corruption_type, 0.3)

    # Create more confounded signal (higher corruption)
    S_heavy = S_original.copy()
    S_heavy = signal_gen.confound_pure_signal(S_heavy, corruption_type, 0.7)

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
    
    # Plot slightly confounded signal (middle)
    for i in range(d):
        axes[1].plot(S_slight[:index_to_plot, i], label=f'Channel {i+1}')
    axes[1].set_title(f'Slightly Confounded Signal (0.3, "{corruption_type}")')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot heavily confounded signal (right)
    for i in range(d):
        axes[2].plot(S_heavy[:index_to_plot, i], label=f'Channel {i+1}')
    axes[2].set_title(f'More Confounded Signal (0.7, "{corruption_type}")')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return


if __name__ == "__main__":
    main()
