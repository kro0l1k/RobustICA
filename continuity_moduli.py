### implements a class to test theorems 4.3 and 4.6 from the paper. ### 

import itertools
import os
import jax
import jax.numpy as jnp
import numpy as np

# Set environment variable for device preference
PREFERRED_DEVICE = os.environ.get('JAX_PLATFORM_NAME', 'auto')

def detect_and_configure_device():
    """Detect and configure the best available JAX device"""
    
    if PREFERRED_DEVICE == 'auto':
        try:
            # Try to detect available devices
            devices = jax.devices()
            print(f"Available devices: {devices}")
            
            if len(devices) > 0:
                device = devices[0]
                platform = str(device.platform).upper()
                
                if 'METAL' in platform:
                    print(f"✓ Metal acceleration detected: {device}")
                    print("  Note: Metal support is experimental in JAX")
                    return "metal"
                elif 'GPU' in platform or 'CUDA' in platform:
                    print(f"✓ CUDA GPU acceleration detected: {device}")
                    return "gpu"
                elif 'TPU' in platform:
                    print(f"✓ TPU acceleration detected: {device}")
                    return "tpu"
                else:
                    print(f"Using CPU device: {device}")
                    return "cpu"
                    
        except Exception as e:
            print(f"Device detection warning: {e}")
            print("Falling back to CPU for stability")
            # Force CPU for compatibility with signax/equinox
            jax.config.update('jax_platform_name', 'cpu')
            return "cpu"
    else:
        print(f"Using forced device: {PREFERRED_DEVICE}")
        try:
            jax.config.update('jax_platform_name', PREFERRED_DEVICE)
            return PREFERRED_DEVICE
        except:
            print("Failed to set device, falling back to CPU")
            jax.config.update('jax_platform_name', 'cpu')
            return "cpu"

# Configure device - this will fall back to CPU if needed for compatibility
detected_device = detect_and_configure_device()

# Now import the rest after JAX is properly configured
import signax
from RICA import SignatureComputer, ContrastCalculator, Optimizer, get_rel_err, M_IplusE
from sample_different_S import SignalGenerator

from scipy.optimize import linear_sum_assignment
import warnings


class ContinuityModuli:
    def __init__(self, true_source, mixed_signal, A, k_0, MC_SAM_LEN):        
        # true_source jnp of shape (n, d)
        # mixed_signal jnp of shape (n, d)
        # A jnp of shape (d, d)
        # k_0 is a scalar

        self.true_source = true_source
        self.mixed_signal = mixed_signal
        self.A = A
        self.k_0 = k_0
        self.MC_SAM_LEN = MC_SAM_LEN
        self.n, self.d = true_source.shape
        self.k_d = jnp.sqrt(1/6 * (self.d - 1) * self.d * (2 * self.d - 1)) # k_d is a scalar
        self.gamma = 1 + jnp.sqrt(5)  # gamma is a scalar, as defined in the paper

        self.signature_computer = SignatureComputer(self.n, self.d, self.MC_SAM_LEN)
        self.contrast_calculator = ContrastCalculator(self.n, self.d, self.MC_SAM_LEN)
        self.optimizer = Optimizer(self.n, self.d, self.MC_SAM_LEN)
        
        
    def get_diagonal_moments(self, signal):
        # given a signal and MC_SAM_LEN, computes the diagonal signature moments
        # <S>_i,i for MC_SAM_LEN, and returns a vector of d elements and vector of d elements
        n, d = signal.shape
        M2and3 = self.signature_computer.compute_up_to_lvl3_expected_sig(signal, test_mean_stationarity=False)
        M2 = M2and3[0]  # M2 is the second moment matrix
        M3 = M2and3[1:]  # M3 are the third moment matrices (d,d,d)
        # Extract diagonals
        second_diag = np.diag(M2)
        third_diag = np.array([M3[i, i, i] for i in range(d)])

        return second_diag, third_diag

    def compute_xi(self, A, R, true_source): 
        # compute the moment matrices for the observed signal whitened by R
        X_obs = jnp.dot(true_source, A.T)
        # whitened signal
        X_whitened = jnp.dot(X_obs, R.T)

        M2and3 = self.signature_computer.compute_up_to_lvl3_expected_sig(X_whitened, test_mean_stationarity=False)
        M2 = M2and3[0]  # M2 is the second moment matrix
        M3 = M2and3[1:]  # M3 are the third moment matrices (d,d,d)
        
        # compute the sum of the Frobenius norms of the moment matrices
        sum2 = 0
        for i in range(self.d):
            sum2 += jnp.linalg.norm(M3[i, :, :], ord='fro') ** 2
        xi = jnp.sqrt(sum2)
        return xi
    
    def compute_s_and_s1(self, S):
        # s is the varsigma constant, s_1 is the varsigma_1 constant
        second_diagonal_S, third_diagonal_S = self.get_diagonal_moments(signal=S)

        print(" while computing s and s_1, second_diagonal_S: ", second_diagonal_S, " third_diagonal_S: ", third_diagonal_S)
        sigmaonesum = 0
        sigmasum = 0
        smallest_sigmaonesum = jnp.inf
        largest_sigmaonesum = -jnp.inf
        # consider all permutations of the indices 0, 1, ..., d-1
        for perm in itertools.permutations(range(self.d)):
            sigmaonesum = 0
            for i in range(self.d):
                sigmaonesum += (i**2) * (second_diagonal_S[perm[i]]**3) / (third_diagonal_S[perm[i]]**2)

            print("sigmaonesum for permutation ", perm, " is ", sigmaonesum)
            smallest_sigmaonesum = min(smallest_sigmaonesum, sigmaonesum)
            largest_sigmaonesum = max(largest_sigmaonesum, sigmaonesum)
        sigmaonesum = smallest_sigmaonesum
        
        for i in range(self.d):
            sigmasum +=  (third_diagonal_S[i]**2) / (second_diagonal_S[i]**3)
    
        print("using sigmaonesum ", sigmaonesum)
        s_1 = 1 / self.k_d * jnp.sqrt(smallest_sigmaonesum)
        s = jnp.sqrt(sigmasum)

        return s_1, s
    
    def get_r0(self, s_1, s, R, B):
        xi = self.compute_xi(self.A, R,  self.true_source)
        print("xi", xi)

        print("R in get_r0: \n", R)
        print(" kond of R is ", np.linalg.cond(R))
        print(" norm of R is ", np.linalg.norm(R, ord='fro'))

        norm_RA = np.linalg.norm(R @ self.A, ord='fro') # NOTE: what norm should we use here?
        kond_RA = np.linalg.cond(R @ self.A)
        
        A_np = np.array(self.A)
        kond_RA = np.linalg.cond(R @ A_np)

        # returns r_0
        print("norm_RA ", norm_RA, " kond_RA ", kond_RA, " xi ", xi)
        print(" while computing r_0: first compononent is ",
              self.k_0 * s_1 * norm_RA / np.sqrt(self.d), " second component is ",
              (1 + self.k_0 + (1 + xi * self.d) * self.k_0 * kond_RA), " then we add this to the third component: ",
              (1 + xi * self.d) * self.k_0 * kond_RA, " and finally we add the fourth component: ",
              kond_RA * s)
        
        r_0 = self.k_0 * s_1 * \
            ( norm_RA/np.sqrt(self.d) *
                (1 + self.k_0 +
                    (1 + xi * self.d) * self.k_0 * kond_RA) + kond_RA * s)

        return r_0

    def test_blind_inversion_thm(self, I_x):
        # theta_star is a jnp of shape (d, d)
        # returns the contunuity moduli c_1 (scalar), c_2(scalar).
        
        M2 = self.signature_computer.compute_lvl2_expected_sig(self.mixed_signal, test_mean_stationarity=False)
        R = self.optimizer.compute_R_fromM2(M2)
        # print("R in test blind inversion thm: \n", R)
        s, s_1 = self.compute_s_and_s1(self.true_source)

        A_np = np.array(self.A)  
        B = np.linalg.inv(A_np)  # B is A_inv
        kond_RA = np.linalg.cond(R @ A_np)
        
        # compute r_0
        r_0 = self.get_r0(s_1, s, R, B)
        print("r_0", r_0)
        
        q0 = (self.gamma * self.k_d * r_0) ** (-1)
        eps0 = q0 /(1 + q0)

        # Calculate the continuity moduli
        c_1 = 2 * self.d * self.k_d * r_0
        c_2 = jnp.sqrt(self.d) * kond_RA * c_1

        # assert they are positive
        assert c_1 > 0, "c_1 is not positive"
        assert c_2 > 0, "c_2 is not positive"

        return eps0, c_1, c_2


##################################################################
########## Robustness Moduli
class RobustnessModuli:
    def __init__(self, true_source, mixed_signal, A, k_0, MC_SAM_LEN):        
        # true_source jnp of shape (n, d)
        # mixed_signal jnp of shape (n, d)
        # A jnp of shape (d, d)
        # k_0 is a scalar

        self.true_source = true_source
        self.mixed_signal = mixed_signal
        self.A = A
        self.k_0 = k_0
        self.MC_SAM_LEN = MC_SAM_LEN
        self.n, self.d = true_source.shape
        self.k_d = jnp.sqrt(1/6 * (self.d - 1) * self.d * (2 * self.d - 1)) # k_d is a scalar
        self.gamma = 1 + jnp.sqrt(5)  # gamma is a scalar, as defined in the paper

        self.signature_computer = SignatureComputer(self.n, self.d, self.MC_SAM_LEN)
        self.contrast_calculator = ContrastCalculator(self.n, self.d, self.MC_SAM_LEN)
        self.optimizer = Optimizer(self.n, self.d, self.MC_SAM_LEN)
       
        #### constants we use for simplification 
        self.alpha = 1
        self.beta = 2
        self.p = 4/3
        self.c_p = np.sqrt(2) / (np.sqrt(2) - 1)
        self.C_p = 8 * self.c_p

    def compute_sigma_star(self, Mu_star_zero, A):
        # get the smallest eigenvalue of A @ Mu_star_zero @ A.T.
        a0sym = 0.5 * (Mu_star_zero + Mu_star_zero.T)  # symmetrise
        A_mu_zero_At = A @ a0sym @ A.T
        eigenvalues = jnp.linalg.eigvalsh(A_mu_zero_At)
        sigma_star = jnp.min(eigenvalues)
        print("sigma_star", sigma_star)
        return sigma_star
    
    def get_rho_0_and_rho_1(self, Mu_star_zero):
        eigenvalues = jnp.linalg.eigvalsh(Mu_star_zero)
        rho_0 = jnp.min(eigenvalues)
        rho_1 = jnp.max(eigenvalues)
        # print("rho0, rho1", rho_0, rho_1)
        return rho_0, rho_1

    def getr_0(self, rho_0, rho_1):
        return rho_0 / (rho_1 + 2 * self.C_p)

    def get_varrho_0(self, Mu_star_matrices):
        # varrho = min <mu_star>_iii for i>=2
        
        third_moments = np.array([Mu_star_matrices[i, i - 1, i - 1] for i in range(1, self.d + 1)])  # shape (d,)
        third_moments_no_first = third_moments[1:]  # shape (d-1,)
        varrho_0 = jnp.min(third_moments_no_first)
        print("varrho_0", varrho_0)
        return varrho_0

    def get_r_1(self, Mu_star_matrices, rho_0, varrho_0):
        mathfrak_mu_star = self.get_L_hat(Mu_star_matrices)
        r_1 =  min( rho_0, varrho_0) / mathfrak_mu_star
        return r_1
    


    def get_delta_distance(self, Mu_matrices, Mu_star_matrices): 
        # Mu is the moment matrix of the observed signal (d+1, d, d)
        # Mu_star is the moment matrix of the idealised orthogonal signal (d+1, d, d)

        difference = Mu_matrices - Mu_star_matrices

        N = self.contrast_calculator.compute_N_fromM2(Mu_matrices[0])
        Ninv = np.linalg.inv(N)
        
        denominator = [1.0]
        denominator.extend(np.diag(N)) 
        denominator = np.array(denominator)  # (d+1,)
        
        sum_sq = 0.0
        for k in range( self.d + 1):
            k_th_component = np.linalg.norm(Ninv @ (difference[k] / denominator[k]) @ Ninv, 'fro')**2
            sum_sq += k_th_component
        res = np.sqrt(sum_sq)
        # print("delta distance", res)
        return res

    def get_L_hat(self, Mu_star_matrices):
        # \hat{L} = \mathfrak{m}_{mu_star}, as defined in (122).
        # \mathfrak{m}_mu_star is defined around (95)
        # as max_i,j,k { sqrt( <mu_star>_ii <mu_star>_jj <mu_star>_kk) },
        # but that is just <mu>_ii **(3/2) since mu_star is diagonal.
        # Mu_star_matrices is of shape (d+1, d, d)
        max_eigenvalue = jnp.max(Mu_star_matrices[0, :, :])
        # print("we want to get L_hat from the max eigenvalue
        # of Mu_star_matrices[0, :, :]", Mu_star_matrices[0, :, :])
        # print("max eigenvalue", max_eigenvalue)
        L_hat = jnp.sqrt(max_eigenvalue ** 3)
        # print("L_hat", L_hat)
        return L_hat
  
    def get_sigmahatstar_at_tilde_c0(self, tilde_c_0, Mu_star_matrices,
                            *, lr=1e-2, n_steps=5_000, key=None):
        """
        Maximise  sqrt(λ_max(A a₀ Aᵀ))  over   a ∈ V
        subject to  Σ‖a_i - μ*_i‖_F ≤ c0.
        
        Returns
        -------
        sigma_hat_star : scalar
        a_star         : jnp.ndarray, shape (d+1, d, d)
        """
        @jax.jit
        def _func(a):
            """λ_max(A a₀ Aᵀ) for the first slice of `a`."""
            a0 = a[0]
            sym_a0 = 0.5 * (a0 + a0.T)  # symmetrise
            return jnp.sqrt(jnp.max(jnp.linalg.eigvalsh(self.A @ sym_a0 @ self.A.T) ))

        @jax.jit
        def _distance(a):
            """Σ‖a_i - μ*_i‖_F  (vectorised)."""
            frob_per_slice = jnp.linalg.norm(a - Mu_star_matrices, ord="fro", axis=(-2, -1))
            return jnp.sum(frob_per_slice)

        @jax.jit
        def _project(a):
            """Exact projection onto the -of-Frobenius ball of radius c0."""
            diff = a - Mu_star_matrices
            dist = _distance(a)
            scale = jnp.minimum(1.0, tilde_c_0 / (dist + 1e-12))
            return Mu_star_matrices + diff * scale

        value_and_grad = jax.value_and_grad(_func)

        @jax.jit
        def _step(a, _):
            val, g = value_and_grad(a)     # ascend on λ_max
            a_new = a + lr * g
            a_new = _project(a_new)
            return a_new, val

        if key is None:
            key = jax.random.PRNGKey(0)
        noise = 1e-3 * jax.random.normal(key, Mu_star_matrices.shape)
        a_init = _project(Mu_star_matrices + noise)

        # run loop
        a_star, val_history = jax.lax.scan(_step, a_init, None, length=n_steps)
        # val_history[-1] is λ_max(a_star)

        sigma_hat_star = jnp.sqrt(_func(a_star))
        return sigma_hat_star, a_star

    def get_rho0tilde(self, second_moments, third_moments, *, key=None):

        Mu_star_zero = jnp.diag(second_moments)  # shape (d,d)
        # print(" Mu_star_zero in get_rho0tilde", Mu_star_zero)
        sigmastar = self.compute_sigma_star(jnp.diag(second_moments), self.A)

        # print("sigmastar in get_rho0tilde", sigmastar)

        def get_minimal_sigma_in_a_ball_of_radius_s(s_radius, inner_key=None):
            # radius of the ball for the projection

            def _min_eigenvalue_of_C(a):
                sym_a = 0.5 * (a + a.T)  # symmetrise
                C = self.A @ sym_a @ self.A.T
                eigenvalues = jnp.linalg.eigvalsh(C)
                min_eigenvalue = jnp.min(eigenvalues)
                return min_eigenvalue

            def _project(x, s_radius):
                # Euclidean projection on the r-ball around second_moments
                dist = jnp.linalg.norm(x - second_moments, ord='fro')
                scale = jnp.minimum(1.0, s_radius / (dist + 1e-12))
                return second_moments + (x - second_moments) * scale

            def _step(x, _):
                g = jax.grad(_min_eigenvalue_of_C)(x)
                return _project(x - 1e-5 * g, s_radius), _min_eigenvalue_of_C(x)  # Changed + to - to minimize

            # Initialize at the centre + tiny noise
            if inner_key is None:
                inner_key = jax.random.PRNGKey(0)

            x0 = _project(Mu_star_zero + 1e-3 * jax.random.normal(inner_key, Mu_star_zero.shape), s_radius)  # shape (d,)
            v_star, min_eigenvalues_hist = jax.lax.scan(_step, x0, xs=None, length=5_000)
            minimal_eigenvalue_in_the_ball = min_eigenvalues_hist[-1]
            
            ## attempt nr 2
            nr_samples = 10000
            # draw nr of samples of noise around Mu_star_zero, with sum <= s_radius, 
            # take among them, take the one with the minimal eigenvalue of C
            noise_samples = jax.random.normal(inner_key, (nr_samples, Mu_star_zero.shape[0], Mu_star_zero.shape[1]))
            noise_samples = noise_samples / jnp.linalg.norm(noise_samples, ord='fro', axis=(-2, -1), keepdims=True) * s_radius
            test_matrices = Mu_star_zero + noise_samples
            min_eigenvalues = jax.vmap(_min_eigenvalue_of_C)(test_matrices)
            # disard negative eigenvalues -- they were not psd matrices
            # min_eigenvalues = jnp.where(min_eigenvalues < 0, jnp.inf, min_eigenvalues)
            # print("min_eigenvalues", min_eigenvalues
            # find the minimum eigenvalue and the corresponding matrix
            minimal_eigenvalue_in_the_ball_second_method = jnp.min(min_eigenvalues)
            v_star = test_matrices[jnp.argmin(min_eigenvalues)]
            print(" two methods for minimal_eigenvalue_in_the_ball: ", minimal_eigenvalue_in_the_ball, minimal_eigenvalue_in_the_ball_second_method)
            return v_star, minimal_eigenvalue_in_the_ball_second_method
       
        # since the function is monotone we will do a binsearch up to a given precision
        precision = 1e-6 # NOTE: can set to 1e-10 but takes a few seconds longer
        low, high = 0.0, 100.0
        while high - low > precision:
            mid = (low + high) / 2.0
            _, min_eigenvalue_mid = get_minimal_sigma_in_a_ball_of_radius_s(mid, key)
            # print(f"Checking mid={mid}, min_eigenvalue_mid={min_eigenvalue_mid}")

            if min_eigenvalue_mid > 0.5 * sigmastar:
                low = mid
            else:
                high = mid
        
        print(" \n\nwas rho_0_tilde radius correctly computed?  ", low)
        print("Final minimal_eigenvalue at this radius:", get_minimal_sigma_in_a_ball_of_radius_s(low, key)[1])
        print(" is it 2 * sigmastar?", 2* get_minimal_sigma_in_a_ball_of_radius_s(low, key)[1] , "should be larger than ", sigmastar)

        r_test = low * 0.9
        print(" what if we used 0.9 of the found radius? (for a smaller r_test 2*min sigma should be larger than sigmastar) ", r_test)
        print(" minimal eigenvalue at this radius: ", get_minimal_sigma_in_a_ball_of_radius_s(r_test, key)[1])
        print(" is it 2 * sigmastar?", 2* get_minimal_sigma_in_a_ball_of_radius_s(r_test, key)[1] , " should be larger than ", sigmastar)
        
        r_test = 1.1 * low
        print(" what if we used 1.1 * the found radius? (for a larger r_test 2*min sigma should be smaller than sigmastar) ", r_test)
        print(" minimal eigenvalue at this radius: ", get_minimal_sigma_in_a_ball_of_radius_s(r_test, key)[1])
        print(" is it 2 * sigmastar?", 2* get_minimal_sigma_in_a_ball_of_radius_s(r_test, key)[1] , " should be smaller than ", sigmastar)
        print("end of debug for rho_0_tilde\n\n\n")


        
        if low <= 0:
            raise ValueError("The rho_0_tilde radius is not positive, which is unexpected.")
        return low  # This is the rho_0 tilde radius
    
    def jnp_inverse_sqrt_psd_matrix(self, M: np.ndarray, eps=1e-16) -> np.ndarray:
        """Computes the inverse square root of a positive semi-definite matrix M.
        """
        M = (M + M.T)/2
        w, v = jnp.linalg.eigh(M)
        # 2. Protect against tiny negative round-off
        w = jnp.maximum(w, eps)
        sqrt_vals = jnp.sqrt(w)
        # 3. Reconstruct:  (V * sqrt_vals) multiplies each column by √λ_i
        result = (v * (1/sqrt_vals)) @ jnp.swapaxes(v, -1, -2)
       
        return result

    def get_mathfrakK_constants(self, r, Mu_star_matrices, *, lr=1e-2, n_steps=5_000, key=None):
        
        # r is a scalar over which we optimise
        # Mu_star_matrices is a jnp of shape (d+1, d, d)
        d   = self.d

        second_moments = jnp.diag(Mu_star_matrices[0]) # shape (d,)
        third_moments  = jnp.array([Mu_star_matrices[i, i - 1, i - 1] for i in range(1, d + 1)])  # shape (d,)
        print("third_moments at the start of get K1 tilde", third_moments)

       ### maximise the varsigma_1 function over the r-ball around second_moments to get the mathfrak{K}_1 constant.
        def _project(x):
            dist  = jnp.linalg.norm(x - second_moments)  
            scale = jnp.minimum(1.0, r / (dist + 1e-12))
            return second_moments + (x - second_moments) * scale
        
        def _varsigma_1(x):
            weighted_sum = 0
            for i in range(self.d):
                weighted_sum += (i *  ((second_moments[i] + x[i]) ** (3/2)) / third_moments[i] ) *2
            res = 1/self.k_d * jnp.sqrt(weighted_sum)
            return res        # scalar

        def _step(x, _):                                 # f(carry, _)
            g = jax.grad(_varsigma_1)(x)
            return _project(x + lr * g), _varsigma_1(x)  # (new_carry, y_out)

        if key is None:
            key = jax.random.PRNGKey(0)
        noise = 1e-3 * jax.random.normal(key, (d,))      # shape matches (d,)
        x0    = _project(second_moments + noise)
        v_star, varsigma1_hist = jax.lax.scan(_step, x0, xs=None, length=n_steps)

        varsigma1_max = varsigma1_hist[-1]
        mfK1 = varsigma1_max
        print("varsigma1_max", varsigma1_max)
        print(" history of the gradient ascent: ", varsigma1_hist)
        # ================================================================
        
        # GET matffrak{K}_2 : simiar approach but different inner function
        def _varsigma(x):
            weighted_sum = 0
            for i in range(self.d):
                weighted_sum += (third_moments[i] + x[i])** 2 / second_moments[i] ** 3
            res = jnp.sqrt(weighted_sum)
            return res        # scalar

        def _step_varsigma(x, _):                                 # f(carry, _)
            g = jax.grad(_varsigma)(x)
            return _project(x + lr * g), _varsigma(x)  # (new_carry, y_out)

        if key is None:
            key = jax.random.PRNGKey(0)
        noise = 1e-3 * jax.random.normal(key, (d,))      # shape matches (d,)
        x0    = _project(second_moments + noise)
        v_star_varsigma, varsigma_hist = jax.lax.scan(_step_varsigma, x0, xs=None, length=n_steps)
        mfK2 = varsigma_hist[-1] # the last value is the maximum

        ### get the rho_0 tilde radius
        rho_zero_tilde = self.get_rho0tilde(second_moments, third_moments)

        print(" while getting K1_tilde, rho_zero_tilde is equal to :  ", rho_zero_tilde)
        
        # get the mathfrak{K}_3 constant : the sup of \xi(a) over rho_0_tilde-ball.
        def _xi(nu): # there is a small error in the paper. when computing R, it should be computed from the mixed singal Anu not from nu.
            # nu is (d+1, d, d)
            # if nu are the second and third order statistics of the true source, we first get the whitening matrix R_nu,
            # then compute the whitened_Anu = jnp.dot(nu, (R@A).T ) 
            second_moments_of_nu = nu[0] # shape (d, d) # NOTE: i think the mistake is here! 

            second_moments_of_observed = jnp.dot(
                jnp.dot(self.A, second_moments_of_nu),
                self.A.T)  # shape (d, d)

            
            C = 0.5 * (second_moments_of_observed + second_moments_of_observed.T)  # symmetrise
            eps = 1e-10
            C_reg = C + eps * jnp.eye(C.shape[0]) # NOTE: I cant even use if else statements to check for invertibility due to JIT!
            R = self.jnp_inverse_sqrt_psd_matrix(M = C_reg)
            whitened_A = R @ self.A
            whitened_second_moments_of_observed = jnp.dot(
                jnp.dot(whitened_A, second_moments_of_nu),
                whitened_A.T)  # shape (d, d)

            third_moments = []
            for k in range(self.d):
                k_th_sum = jnp.zeros((self.d, self.d)) # shape (d, d)
                for l in range(self.d):
                    k_th_sum += whitened_A[k, l] * nu[l + 1, :, :] # shape (d, d)
                
                k_th_moment_of_observed = jnp.dot(
                        jnp.dot(whitened_A, k_th_sum),
                        whitened_A.T)  # shape (d, d)
                third_moments.append(k_th_moment_of_observed)
            third_moments = jnp.array(third_moments)  # shape (d, d, d)

            whitened_second_moments_of_observed = jnp.expand_dims(
                whitened_second_moments_of_observed, axis=0)  # shape (1, d, d)
            whitened_Anu = jnp.concatenate(
                [whitened_second_moments_of_observed, third_moments],
                axis=0)  # shape (d+1, d, d)
            
            # print("whitened_Anu (while computing xi for mathfrakK_3)", whitened_Anu)
            weighted_sum = 0
            for i in range(1, self.d+1):
                weighted_sum += jnp.linalg.norm(whitened_Anu[i], ord='fro') ** 2  # TODO: double check if it should really be frobenius norm
            res = jnp.sqrt(weighted_sum)
            return res
        
        # print(" example of computing xi(a) for the orthogonal signal mustar: ", _xi(Mu_star_matrices))
        
        def project_rho_zero_tilde_ball(nu):
            # nu is (d+1, d, d)
            # Euclidean projection on the rho_zero_tilde-ball around Mu_star_matrices
            sum_squared = jnp.sum(jnp.linalg.norm(nu - Mu_star_matrices, ord='fro', axis=(-2, -1)) ** 2)
            dist = jnp.sqrt(sum_squared)
            scale = jnp.minimum(1.0, rho_zero_tilde / (dist + 1e-12))
            return Mu_star_matrices + (nu - Mu_star_matrices) * scale

        # initlaise at Mu_star_matrices + noise and do projected gradient ascent over (d+1, d, d) matrices
        
        if key is None:
            key = jax.random.PRNGKey(0)
        noise = 1e-3 * jax.random.normal(key, Mu_star_matrices.shape)  # shape (d+1, d, d)
        nu_init = project_rho_zero_tilde_ball(Mu_star_matrices + noise)
        print("nu_init", nu_init)
        value_and_grad = jax.value_and_grad(_xi)
        
        @jax.jit
        def _step_nu(nu, _):                                 # f(carry, _)
            val, grad = value_and_grad(nu)
            return project_rho_zero_tilde_ball(nu + lr * grad), val
        nu_star, xi_hist = jax.lax.scan(_step_nu, nu_init, xs=None, length=n_steps)
        
        print(" the scan to compute mathfrakK_3:")
        print("nu_star", nu_star)
        print("xi_hist", xi_hist)
        mfK3 = xi_hist[-1]
        # print(" xi history while computing mathfrakK_3", xi_hist)
        
        # approach 2: draw 10000 samples of noise around Mu_star_matrices, with sum <= rho_zero_tilde,
        # take among them, take the one with the maximal xi value.
        noise_samples = jax.random.normal(key, (10000, Mu_star_matrices.shape[0], Mu_star_matrices.shape[1], Mu_star_matrices.shape[2]))
        noise_samples = noise_samples / jnp.linalg.norm(noise_samples, ord='fro', axis=(-2, -1), keepdims=True) * rho_zero_tilde
        test_matrices = Mu_star_matrices + noise_samples
        # compute the xi for each test matrix
        xi_values = jax.vmap(_xi)(test_matrices)
        # find the maximum xi value and the corresponding matrix
        mfK3_second_method = jnp.max(xi_values)

        print("mfK3 from the second method", mfK3_second_method)
        print("mfK3 from the first method", mfK3)
        
        print(" sanity check: value of xi for the orthogonal signal Mu_star_matrices: ", _xi(Mu_star_matrices))
        
        if jnp.isnan(mfK3) or jnp.isinf(mfK3):
            raise ValueError("mathfrakK_3 is NaN or Inf, which is unexpected.")
        # ensure all constants are positive
        if mfK1 <= 0 or mfK2 <= 0 or mfK3 <= 0:
                raise ValueError("One of the mathfrakK constants is not positive, which is unexpected.")
        print("mathfrakK_1, mathfrakK_2, mathfrakK_3", mfK1, mfK2, mfK3)
        return mfK1, mfK2, mfK3

    # --------------------------------------------------------------------------

    def get_K1_tilde_and_r3(self, r, tilde_r_2, Mu_star_matrices,
                            *, lr=1e-2, n_steps=5_000, key=None):

        hatK  = self.get_L_hat(Mu_star_matrices)  # using remark E1(ii) from the paper
        mfq_r = hatK * r
        mfK1, mfK2, mfK3 = self.get_mathfrakK_constants(
            r = mfq_r,
            Mu_star_matrices=Mu_star_matrices,
            lr=lr,
            n_steps=n_steps,
            key=key
        )
        print("mathfrakK_1, mathfrakK_2, mathfrakK_3", mfK1, mfK2, mfK3)
        # ------------------------------------------------------------------

        def _beta1_for_tau(a, A):
            # a is a jnp of shape (d+1, d, d)
            a0 = a[0]
            A_inv = jnp.linalg.inv(A)
            sym_a0 = 0.5 * (a0 + a0.T)  # symmetrise
            max_eigenvalue = jnp.max(jnp.linalg.eigvalsh(A @ sym_a0 @ A.T))
            fro_inv_norm = jnp.linalg.norm(A_inv , ord='fro')
            return fro_inv_norm * jnp.sqrt(max_eigenvalue)
            
        def _beta2hat_for_tau(a, A):
            a0 = a[0]
            sym_a0 = 0.5 * (a0 + a0.T)  # symmetrise
            prod_of_eigenvalues = jnp.prod(jnp.linalg.eigvalsh(A @ sym_a0 @ A.T))     
            numerator =  2 * (prod_of_eigenvalues ** 0.5) * (jnp.linalg.norm(A, 'fro') ** self.d)
            res = numerator
            return res
        
        def _betahat_r_for_tau(a, A, smallest_denominator):
            numerator = _beta2hat_for_tau(a, A)
            denominator = smallest_denominator
            return numerator / denominator
        # ------------------------------------------------------------

        def find_smallest_beta2_denominator(r_for_beta2_denominator, inner_key=None): 

            def _denominator(a0): 
                syma0 = 0.5 * (a0 + a0.T)  # symmetrise
                minimal_eigenvalue = jnp.min(
                        jnp.linalg.eigvalsh(self.A @ syma0 @ self.A.T))
                denominator = jnp.linalg.det(self.A) * \
                    (jnp.sqrt(self.d * minimal_eigenvalue) ** self.d)
                return denominator

            def _project_for_denominator(a0):
                # Euclidean projection on the qr-ball around Mu_star_matrices[0]
                dist = jnp.linalg.norm(a0 - Mu_star_matrices[0], ord='fro')
                scale = jnp.minimum(1.0, r_for_beta2_denominator / (dist + 1e-12))
                return Mu_star_matrices[0] + (a0 - Mu_star_matrices[0]) * scale

            def _step(a0, _):
                g = jax.grad(_denominator)(a0)
                return _project_for_denominator(a0 - lr * g), _denominator(a0)
            if inner_key is None:
                inner_key = jax.random.PRNGKey(0)
            noise = 1e-3 * jax.random.normal(inner_key,
                                             Mu_star_matrices[0].shape)
            a0_init = _project_for_denominator(Mu_star_matrices[0] + noise)
            a0_star, denominator_hist = jax.lax.scan(_step,
                                                     a0_init,
                                                     xs=None,
                                                     length=n_steps)
            smallest_denominator = denominator_hist[-1]
            
            # second method: draw 10000 samples of noise around Mu_star_matrices[0], with sum <= r_for_beta2_denominator,
            # take among them, take the one with the minimal denominator.
            noise_samples = jax.random.normal(inner_key, (10000, Mu_star_matrices[0].shape[0], Mu_star_matrices[0].shape[1]))
            noise_samples = noise_samples / jnp.linalg.norm(noise_samples, ord='fro', axis=(-2, -1), keepdims=True) * r_for_beta2_denominator
            test_matrices = Mu_star_matrices[0] + noise_samples
            # compute the denominator for each test matrix
            denominators = jax.vmap(_denominator)(test_matrices)
            # find the minimum denominator and the corresponding matrix
            smallest_denominator_second_method = jnp.min(denominators)
            print("for r = ", r_for_beta2_denominator, " find_smallest_beta2_denominator", smallest_denominator, " history: ", denominator_hist)
            print("smallest_denominator_second_method", smallest_denominator_second_method)

            return smallest_denominator

        # ------------------------------------------------------------
        def _Q(u):
            return u / (1 + u)
        
        def _tau(r_tau, inner_key=None): 
            
            gamma = 1 + jnp.sqrt(5)
            # tau is the minimum of the t function:
            mfq_r_tau = hatK * tilde_r_2
            
            ##### find the smallest denominator here, for r = mfq_r_tau
            smallest_denominator_for_tau = find_smallest_beta2_denominator(mfq_r_tau, inner_key)
            print(" for r = ", r_tau, " and mfq_r_tau = ", mfq_r_tau)
            print(" the smallest denominator for tau: ", smallest_denominator_for_tau)
            def _psi_for_tau(a):
                """a is (d+1, d, d)"""
                # evaluate \psi_r(a)
                evaluated_hatbeta_r = _betahat_r_for_tau(a, self.A, smallest_denominator_for_tau)
                psi = self.k_0 * mfK1 * ( _beta1_for_tau(a, self.A)/jnp.sqrt(self.d) *  (1 + self.k_0 + (1 + mfK3 * self.d) * self.k_0 * evaluated_hatbeta_r) + evaluated_hatbeta_r * mfK2)
                return psi
            
            def t(a):
                reci = gamma * self.k_d * _psi_for_tau(a) + 1e-12
                return _Q(reci ** (-1))

            # find the minimum of t over the mfq_r-ball around Mu_star_matrices
            def _project_for_tau(a):
                # Euclidean projection on the mfq_r-ball around Mu_star_matrices
                sum_squared = jnp.sum(jnp.linalg.norm(a - Mu_star_matrices, ord='fro', axis=(-2, -1)) ** 2)
                dist = jnp.sqrt(sum_squared)
                scale = jnp.minimum(1.0, mfq_r_tau / (dist + 1e-12))
                return Mu_star_matrices + (a - Mu_star_matrices) * scale
            lr_to_learn_tau = 1e-2
            def _step_tau(a, _):
                g = jax.grad(t)(a)
                return _project_for_tau(a - lr_to_learn_tau * g), t(a)

            # ------------------------------------------------------------

            if inner_key is None:
                inner_key = jax.random.PRNGKey(0)
            noise = 1e-2 * jax.random.normal(inner_key, Mu_star_matrices.shape)
            
            a_init = _project_for_tau(Mu_star_matrices + noise)
            a_star, t_hist = jax.lax.scan(_step_tau, a_init, xs=None, length=n_steps)
            t_of_r = t_hist[-1]  # the last value is the minimum
            print(" t_history to find tau: ", t_hist)
            return t_of_r  # scalar
        
        # ------------------------------------------------------------
        # get the target for tau - we want to find the supremum of this function. I think the bug is here:
        # ===============================
        def _get_sup_of_IC_defect(Mu_star_matrices, r_mid_for_IC_defect, verbose = False, inner_key=None):
            
            # PGD over \mathfrak{B}_{r_mid_for_IC_defect}(Mu_star_matrices)
            EPS = 1e-8                              
            def safe_sqrt(x, eps: float = EPS):
                return jnp.sqrt(jnp.clip(x, a_min=eps))

            def safe_inv(x, eps: float = EPS):
                return 1.0 / jnp.clip(x, a_min=eps)
            
            second_moments_diag_star = jnp.diag(Mu_star_matrices[0])  # shape (d,)
            sqrt_second_moments_star = safe_sqrt(second_moments_diag_star)
            Ninv_star = jnp.diag(safe_inv(sqrt_second_moments_star))
            denominator_star = jnp.array([1.0] + [sqrt_second_moments_star[i] for i in range(self.d)])

            def target_delta_orthogonal(mu):
                # this is the way to compute the IC-defect of a mean-stationary signal
                # mu is (d+1, d, d)
                
                second_moments_diag = jnp.diag(mu[0])  # shape (d,)
                
                sqrt_second_moments = safe_sqrt(second_moments_diag)
                Ninv = jnp.diag(safe_inv(sqrt_second_moments))
                # Create a proper denominator array with shape (d+1,d)
                denominator = jnp.array([1.0] + [sqrt_second_moments[i] for i in range(self.d)])
                print("denominator shape in target delta", denominator.shape)                
                norm_values = jnp.array([
                    jnp.linalg.norm(Ninv @ ((mu[i, :, :] - Mu_star_matrices[i, :, :]) / denominator[i, None]) @ Ninv, 'fro')**2
                    for i in range(self.d + 1)
                ])
                delta_sq = jnp.sum(norm_values)
                delta = jnp.sqrt(delta_sq)
                return delta
            
            # ------------------------------------------------------------
            def _get_mathd_distance(mu_matrices):
                # mu_matrices is (d+1, d, d)
                # first we get the delta(Mu_star_matrices, mu_matrices)
                # then bar_d = d(selfA, A_test)

                def get_delta_distance(Mu_star_matrices, mu_matrices):
                    # this is just delta(mu1, mu2) in the paper
                    # print(" what is the diff : \n",  mu_matrices - Mu_star_matrices )
                    norm_values = jnp.array([
                        jnp.linalg.norm(
                            Ninv_star @ ((mu_matrices[i, :, :] - Mu_star_matrices[i, :, :]) / denominator_star[i]) @ Ninv_star,
                            'fro') ** 2
                        for i in range(self.d + 1)
                    ])
                    delta_sq = jnp.sum(norm_values)
                    deltamu1mu2 = jnp.sqrt(delta_sq)
                    return deltamu1mu2

                
                delta_mu1mu2 = get_delta_distance(Mu_star_matrices, mu_matrices)
                mathd = delta_mu1mu2 
                return mathd

            def _project_using_math_d_distance(mu_matrices):
                # projection on the r_mid_for_IC_defect-ball around Mu_star_matrices
                dist = _get_mathd_distance(mu_matrices)
                scale = jnp.minimum(1.0, r_mid_for_IC_defect / (dist + 1e-12))
                res = Mu_star_matrices * (1 - scale) + mu_matrices * scale
               
                return res
            # ------------------------------------------------------------
            lr_for_sup_of_ICdefect = 1e-2  # learning rate for the projected gradient ascent
            def _step_for_delta_orthogonal(carry, _):
                # Unpack the tuple
                mu_matrices = carry
                g = jax.grad(target_delta_orthogonal)(mu_matrices)
                # print("gradient in _step_for_delta_orthogonal", g)
                new_mu_matrices = _project_using_math_d_distance(mu_matrices + lr_for_sup_of_ICdefect * g)
                # Return the new carry state (tuple) and the output value
                return new_mu_matrices, _get_mathd_distance(new_mu_matrices)
            # ------------------------------------------------------------
            # initial point (centre of the ball + tiny noise)
            if inner_key is None:
                inner_key = jax.random.PRNGKey(0)
            
            # second way: sample 10000 matrices around Mu_star_matrices, with sum <= r_mid_for_IC_defect,
            # compute the IC defect for each of them, and take the maximum
            noise_samples = jax.random.normal(inner_key, (10_000, Mu_star_matrices.shape[0], Mu_star_matrices.shape[1], Mu_star_matrices.shape[2]))
            test_matrices = Mu_star_matrices + noise_samples
             # for each of the 10000 matrices get the mathd distance to Mu_star_matrices
            dist_for_test_matrices = jax.vmap(_get_mathd_distance)(test_matrices)  # shape (10000,)
            noise_samples = noise_samples / jnp.expand_dims(dist_for_test_matrices, axis=(-3, -2, -1)) * r_mid_for_IC_defect
            test_matrices = Mu_star_matrices + noise_samples
            # compute the IC defect for each test matrix
            IC_defect_values = jax.vmap(target_delta_orthogonal)(test_matrices)
            # find the maximum IC defect value and the corresponding matrix
            max_IC_defect = jnp.max(IC_defect_values)
            print("max_IC_defect", max_IC_defect, "for r_mid_for_IC_defect = ", r_mid_for_IC_defect)
            return max_IC_defect  
        # sup_{(mu,f) \in B(mu_star, r)} delta_orthogonal(mu) 
 
        if key is None:
                key = jax.random.PRNGKey(0)
        # ------------------------------------------------------------

        def get_r_3(tilde_r_2):
            # r_3 is defined as the inf for which \tau(r_3) - sup(r_3) <=0
            print("#" * 50)
            print("Computing r3_tilde")
            print("#" * 50)
            
            
            # we want to find the inf between 0 and 0.95 * r2 (what is r2? i guess we can upper bound it since we are looking for an inf anyway)
            tolerance = 1e-6
            low = 0.0
            high = tilde_r_2
            it = 0 

            # return 0
            # for now we still test the sup function
            while high - low > tolerance:
                r_mid = (low + high) / 2.0
                it += 1
                if it > 100:
                    raise ValueError("Too many iterations in get_r3_tilde")

                tau_rmid = _tau(r_mid, key)
                sup_rmid = _get_sup_of_IC_defect(Mu_star_matrices, r_mid)
                diff_for_rmid = tau_rmid - sup_rmid
                print("Iteration ", it, " r_mid: ", r_mid, " tau_rmid: ", tau_rmid, " sup_rmid: ", sup_rmid, " diff_for_rmid: ", diff_for_rmid)
                # print(" mid: ", r_mid, " tau_mid ", tau_rmid)

                if diff_for_rmid < 0:
                    low = r_mid
                else:
                    high = r_mid

            r_3 = low
            print(f" \n\n\n####### THIS IS WHAT WE WERE LOOKING FOR: r3= {r_3} #######")
            return r_3

        print("getting r_3")
        r_3 = get_r_3(tilde_r_2)  # r_3 is the infimum of the radius for which tau(r_3) - sup(r_3) <= 0
        # ------------------------------------------------------------
        # PGD in V = (d+1, d, d) matrices, radius mathfrak{q}_r3 
        mfqr3 = hatK * r_3  # \mathfrak{q}_r3 = \hat{K} * r_3
        
        print(" final gradient ascent to get K1_tilde")
        print("mfqr3 : ", mfqr3)
        ## find smallest denominator for K1_tilde
        smallest_denominator_for_K1_tilde = find_smallest_beta2_denominator(mfqr3, key)
        print("smallest_denominator_for_K1_tilde", smallest_denominator_for_K1_tilde)
        def _psi(a):
            """a is (d+1, d, d)"""
            # evaluate \psi_r(a)
            evaluated_hatbeta_r = _betahat_r_for_tau(a, self.A, smallest_denominator_for_K1_tilde)

            psi = self.k_0 * mfK1 * (_beta1_for_tau(a, self.A)/jnp.sqrt(self.d) *  (1 + self.k_0 + (1 + mfK3 * self.d) * self.k_0 * evaluated_hatbeta_r) + evaluated_hatbeta_r * mfK2)
            return psi
        
        def _project_for_K1_tilde(a):
            # Euclidean projection on the r-ball around Mu_star_matrices
            sum_squared = jnp.sum(jnp.linalg.norm(a - Mu_star_matrices, ord='fro', axis=(-2, -1)) ** 2)
            dist = jnp.sqrt(sum_squared)
            scale = jnp.minimum(1.0, mfqr3 / (dist + 1e-12))
            return Mu_star_matrices + (a - Mu_star_matrices) * scale
        # ------------------------------------------------------------
        def _step_K1_tilde(a, _):
            g = jax.grad(_psi)(a)
            return _project_for_K1_tilde(a + lr * g), _psi(a)
        if key is None:
            key = jax.random.PRNGKey(0)
        noise = 1e-3 * jax.random.normal(key, Mu_star_matrices.shape)
        a_init = _project_for_K1_tilde(Mu_star_matrices + noise)
        a_star, psi_hist = jax.lax.scan(_step_K1_tilde, a_init, xs=None, length=n_steps)
        max_psi = psi_hist[-1]  # the last value is the maximum
        print("max_psi in get_K1_tilde", max_psi, " history of the gradient ascent: ", psi_hist)

        # second approach: sample 10000 matrices around Mu_star_matrices, with sum <= mfqr3,
        # compute the psi for each of them, and take the maximum
        noise_samples = jax.random.normal(key, (10000, Mu_star_matrices.shape[0], Mu_star_matrices.shape[1], Mu_star_matrices.shape[2]))
        noise_samples = noise_samples / jnp.linalg.norm(noise_samples, ord='fro', axis=(-2, -1), keepdims=True) * mfqr3
        test_matrices = Mu_star_matrices + noise_samples
        # compute the psi for each test matrix
        psi_values = jax.vmap(_psi)(test_matrices)
        # find the maximum psi value and the corresponding matrix
        max_psi_second_method = jnp.max(psi_values)
        print("max_psi from the second method", max_psi_second_method)
        
        
        K1_tilde = 2 * self.d * self.k_d * max_psi
        return K1_tilde, r_3
        # ------------------------------------------------------------

    def test_robustness_thm(self, theta_star):

        # compute R_X
        M_2and3_of_X = self.signature_computer.compute_up_to_lvl3_expected_sig(self.mixed_signal, test_mean_stationarity=False)  # shape (d+1, d, d)
        M2, M3 = M_2and3_of_X[0], M_2and3_of_X[1:] 
        R = self.optimizer.compute_R_fromM2(M2)
        # NOTE: Is R the whitening matrix of the true source or the mixed signal?
        B = np.linalg.inv(self.A)

        
        kond_RA = np.linalg.cond(R @ self.A)
        print("condition number of R @ A", kond_RA)
       

        # get the constant Lr: 
        Mu_matrices = self.signature_computer.compute_up_to_lvl3_expected_sig(self.true_source, test_mean_stationarity=False)  # shape (d+1, d, d)
        Mu_star_matrices = self.contrast_calculator.get_Mu_star_matrices(Mu_matrices)  # (d+1, d, d)
        Mu_star_zero = Mu_star_matrices[0]
        sigma_star = self.compute_sigma_star(Mu_star_zero, self.A)
        print("sigma_star", sigma_star)

        rho_0, rho_1 = self.get_rho_0_and_rho_1(Mu_star_zero)
        print("rho0, rho1", rho_0, rho_1)
        print("")
        
        # get r_0, r_1, var_rho_0, r_star, \tilde_rho_0, \tilde{r}_1
        r_0 = self.getr_0(rho_0, rho_1)
        var_rho_0 = self.get_varrho_0(Mu_star_matrices)
        r_1 = self.get_r_1(Mu_star_matrices, rho_0, var_rho_0)
        
        second_moments = np.diag(Mu_star_zero)  # shape (d,)
        third_moments = np.array([Mu_star_matrices[i, i - 1, i - 1] for i in range(1, self.d + 1)])  # shape (d,)
        tilde_rho_0 = self.get_rho0tilde(second_moments,third_moments)
        mathfrak_m = self.get_L_hat(Mu_star_matrices)  
        r_star = 10000. # NOTE: placeholder for now, it's fine since we are doing a binary search for r_3, but could play a role in \psi
        print("mathfrak_m", mathfrak_m)
        tilde_r_2 = 0.95 * min(r_0, r_1, tilde_rho_0/mathfrak_m , r_star/ mathfrak_m) 
        print("tilde_r2", tilde_r_2)

        # get the delta distance
        delta_distance = self.get_delta_distance(Mu_matrices, Mu_star_matrices)
        print("delta distance", delta_distance, " rho_0/rho_1", rho_0/rho_1, "tilde_r_2", tilde_r_2)

        assert rho_0/rho_1 < delta_distance, "rho0/rho1 is not less than delta distance"
        r = 0.5 * ( rho_0 / rho_1 + delta_distance)
        print("r", r)
         
        Lr = jnp.sqrt( 2 * self.d / sigma_star) # right before 190
        print("Lr ", Lr)
        L_hat = self.get_L_hat(Mu_star_matrices)
        print("L_hat", L_hat)
        # gamma_star_tilde = self.get_gamma_star_tilde(Mu_star_matrices)
        # print("gamma_star_tilde", gamma_star_tilde)
        
        Kprime = Lr  # After Remark E1(ii)
        
        # NOTE: if it is true that Kprime >= \tilde{K}_1 and the inequality is strict, is in the right direction then we use
        Ktilde_1, r_3 = self.get_K1_tilde_and_r3(
            r=r,
            tilde_r_2=tilde_r_2,
            Mu_star_matrices=Mu_star_matrices,
        )
    
        tilde_c_0 = r_3 

        # recall \mathfrak{q}_r = \hat{K}*r^\alpha = \hat{T}*r
        sigma_hat_star_tilde_c_0, a_maximal_ = self.get_sigmahatstar_at_tilde_c0(
            tilde_c_0=tilde_c_0,
            Mu_star_matrices=Mu_star_matrices,
            lr=1e-2,
            n_steps=5_000,
        )
        print("sigma_hat_star_tilde_c0", sigma_hat_star_tilde_c_0)

        print("tilde_c_0", tilde_c_0)

        tilde_c_1 = 1/Kprime
        print("tilde_c_1", tilde_c_1)
        tilde_c_2 = Ktilde_1
        tilde_c_3 = jnp.sqrt(self.d) * Lr * np.linalg.norm(self.A, ord=2) 
        tilde_c_4 = jnp.sqrt(self.d) * Lr * sigma_hat_star_tilde_c_0 * Lr * np.linalg.cond(self.A)
        return tilde_c_0, tilde_c_1, tilde_c_2, tilde_c_3, tilde_c_4


def main():
    
    print("JAX devices:", jax.devices())
    d, nr_of_mc_samples_for_moment_computation = 3, 1_000_000
    MC_SAM_LEN = 15
    A = jnp.array(
         [[-0.37842,     0.91451844, -0.14301863],
        [-0.69043073, -0.3817858,  -0.61444691],
        [-0.61652552, -0.13377454,  0.77588701]])

    A_inv = jnp.linalg.inv(A)
    signalgenerator = SignalGenerator()
    S = signalgenerator.sample_s(d = d, n = MC_SAM_LEN * nr_of_mc_samples_for_moment_computation, ts_type= "iid")
    print("shape of S", S.shape)

    contrastcalc = ContrastCalculator(n = MC_SAM_LEN * nr_of_mc_samples_for_moment_computation, d = d, MC_SAM_LEN= MC_SAM_LEN, check_identifiability_criteria=False)
    print(" delta of S: ", contrastcalc.compute_delta(S))
    X = S @ A.T
    optimizer = Optimizer(MC_SAM_LEN * nr_of_mc_samples_for_moment_computation, d , MC_SAM_LEN=MC_SAM_LEN)
    I_x = optimizer.RICA(X)
    rica_rel_err = get_rel_err(I_x, A_inv, *M_IplusE(I_x , A))
    print("relative error of RICA:", rica_rel_err)
    print("I_x @ A:", I_x @ A)

    eps0, c1, c2 = ContinuityModuli(S, X, A, 1, MC_SAM_LEN).test_blind_inversion_thm(I_x)
    print(f"Continuity moduli: c1 = {c1}, c2 = {c2}")

    print(" computing the constants for theorem 4.6")
    ctilde = RobustnessModuli(S, X, A, 1,
                            MC_SAM_LEN).test_robustness_thm(I_x)
    thm46c0, thm46c1, thm46c2, thm46c3, thm46c4 = ctilde

    print(f"Continuity moduli for theorem 4.6: "
        f"c0 = {thm46c0}, c1 = {thm46c1}, c2 = {thm46c2}, "
        f"c3 = {thm46c3}, c4 = {thm46c4}")


    epsilon = 0.001
    print("for epsilon = " , epsilon, "  tilde_delta:  we take the min of", thm46c0, " second part ", (thm46c1 * epsilon / (thm46c2 + epsilon)) **2)


    epsilon = 0.05
    print("for epsilon = " , epsilon, "  tilde_delta:  we take the min of", thm46c0, " second part ", (thm46c1 * epsilon / (thm46c2 + epsilon)) **2)

    epsilon = 0.1
    print("for epsilon = " , epsilon, "  tilde_delta:  we take the min of", thm46c0, " second part ", (thm46c1 * epsilon / (thm46c2 + epsilon)) **2)

    epsilon = 0.2
    print("for epsilon = " , epsilon, "  tilde_delta:  we take the min of", thm46c0, " second part ", (thm46c1 * epsilon / (thm46c2 + epsilon)) **2)
    
    epsilon = 0.5
    print("for epsilon = " , epsilon, "  tilde_delta:  we take the min of", thm46c0, " second part ", (thm46c1 * epsilon / (thm46c2 + epsilon)) **2)

    epsilon = 1.0
    print("for epsilon = " , epsilon, "  tilde_delta:  we take the min of", thm46c0, " second part ", (thm46c1 * epsilon / (thm46c2 + epsilon)) **2)

    epsilon = 10.0
    print("for epsilon = " , epsilon, "  tilde_delta:  we take the min of", thm46c0, " second part ", (thm46c1 * epsilon / (thm46c2 + epsilon)) **2)

    epsilon = 50.0
    print("for epsilon = " , epsilon, "  tilde_delta:  we take the min of", thm46c0, " second part ", (thm46c1 * epsilon / (thm46c2 + epsilon)) **2)

if __name__ == "__main__":
    main()