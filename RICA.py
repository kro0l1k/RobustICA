import jax
import jax.numpy as jnp
from jax import config
import numpy as np
import signax
import scipy
import warnings
import os
from sample_different_S import SignalGenerator
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from sklearn.decomposition import FastICA
from SOBI import sobi_algo
from statsmodels.tsa.stattools import adfuller

from numpy.linalg import svd, eigh, qr, norm, cond


print("\n\nAvailable devices:", jax.devices())
device_found = False  # Initialize the variable

try:
    metal_devices = jax.devices("METAL")
    if metal_devices:
        config.update("jax_default_device", metal_devices[0])
        print("Using Metal/Apple GPU for computations")
        device_found = True
except Exception as e:
    print(f"Metal device detection error: {e}")

# If Metal wasn't found/set, try standard CUDA GPU
if not device_found:
    try:
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            config.update("jax_default_device", gpu_devices[0])
            print("Using CUDA GPU for computations")
            device_found = True
    except Exception as e:
        print(f"CUDA GPU detection error: {e}")

# Try TPU if GPU and Metal both failed
if not device_found:
    try:
        tpu_devices = jax.devices("tpu")
        if tpu_devices:
            config.update("jax_default_device", tpu_devices[0])
            print("Using TPU for computations")
            device_found = True
    except Exception as e:
        print(f"TPU detection error: {e}")

# Fall back to CPU if no accelerators were found
if not device_found:
    print("No accelerators found, using CPU for computations")


def M_IplusE(W, A):
    d = A.shape[0]
    WA = W @ A
    Id= np.eye(d)
    if WA.shape[0] != d or WA.shape[1] != d:
        raise ValueError("W @ A must be a square matrix of shape (d, d)")
    cost = -np.abs(WA)
    r, c = linear_sum_assignment(cost)
    M = np.zeros_like(WA)
    for i, j in zip(r, c):
        M[i, j] = WA[i, j]
    # Convert JAX arrays to NumPy arrays for better Metal compatibility
    M_np = np.array(M)
    # Use NumPy's matrix inversion instead of JAX's
    if np.linalg.cond(M_np) > 1e3:
        warnings.warn("M_inv nearly singular")
    M_inv = np.linalg.inv(M_np)
    WA_np = np.array(WA)
    E = M_inv @ WA_np - Id
    
    return M, E

def get_rel_err(I_x, A_inv, M, E):
    abs_err = np.linalg.norm(I_x - M @ A_inv, 'fro')
    rel_err = abs_err / np.linalg.norm(M @ A_inv, 'fro')
    return rel_err


# ===== FLEX Joint Diagonaliser for Real Matrices =====
class FlexJDReal:
    """
    Joint diagonaliser restricted to real orthogonal matrices whose columns
    have unit l2-norm.  The ``fit`` method returns (W, success_bool, info_dict).
    """
    def __init__(self, max_iter=10_000_000, tol_angle=1e-12,
                 offdiag_tol=1e-6, norm_tol=1e-2):
        self.max_iter   = max_iter       # angular convergence test
        self.tol_angle  = tol_angle
        self.offdiag_tol = offdiag_tol   # diag success test
        self.norm_tol   = norm_tol       # ‖col‖≃1 test

    def fit(self, mats):
        """
        Parameters
        ----------
        mats : list[ndarray]      # each (d,d), typically symmetric
        Returns
        -------
        W         : (d,d) real orthogonal matrix
        success   : bool
        info      : dict   {'iters':…, 'offdiag':…, 'colnorms':…}
        """
        d = mats[0].shape[0]
        # -- real random orthogonal initialisation --------------------
        W, _ = qr(np.random.randn(d, d))            # real    instead of complex
        prev_W = np.empty_like(W)

        # -- main loop ------------------------------------------------
        logging_freq = 10000
        for it in range(self.max_iter):
            prev_W[:] = W
            for i in range(d):
                W_bar = np.delete(W, i, axis=1)
                Q_i   = self._Q_i(mats, W, W_bar)
                C_i   = self._null_space(W_bar.T)    # transpose    instead of conjugate transpose
                C_t   = C_i @ C_i.T
                W[:, i] = self._moqo(C_t, Q_i)       # already unit-normed
            
            # -- check convergence --------------------------------------
            if it % logging_freq == 0:
                offdiag = self._contrast(mats, W)
                col_norms = norm(W, axis=0)
                print(f"Iter {it}: offdiag={offdiag:.2e}, col_norms={col_norms}")
            if np.allclose(W, prev_W, rtol=self.tol_angle):
                break

        # ------------- SUCCESS DIAGNOSTICS ---------------------------
        col_norms  = norm(W, axis=0) 
        offdiag    = self._contrast(mats, W)

        success = (np.all(np.abs(col_norms - 1) < self.norm_tol)
                   and offdiag < self.offdiag_tol)

        info = dict(iters=it+1, offdiag=offdiag, colnorms=col_norms)

        return W, success, info

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _Q_i(mats, W, W_bar):
        """Equation (2.4) in the FLEX paper – but *purely real*."""
        Q = np.zeros((W.shape[0], W.shape[0]))
        for R in mats:
            t1 = R @ W_bar @ W_bar.T @ R.T
            t2 = R.T @ W_bar @ W_bar.T @ R
            Q += t1 + t2
        return Q

    @staticmethod
    def _null_space(A, rcond=None):
        """Real null-space (columns form an orthonormal basis)."""
        U, s, Vt = svd(A, full_matrices=True)
        if rcond is None:
            rcond = np.finfo(s.dtype).eps * max(A.shape)
        rank = (s > rcond * s[0]).sum()
        return Vt[rank:].T

    def _moqo(self, C_tilde, Q):
        """
        Maximise the quartic objective on the *real* Stiefel-1 manifold.
        All inputs are real symmetric, so eigenvectors are real.
        """
        eigvals, eigvecs = eigh(Q)
        thresh = self.tol_angle * np.max(np.abs(eigvals))
        zero   = np.abs(eigvals) < thresh

        if not np.any(zero):                     # Q invertible
            #_, v = eigh(C_tilde, Q)
            _, v = scipy.linalg.eigh(C_tilde, Q)
            w = v[:, -1]
        else:                                    # Q singular  -> cases 2–3
            U0, U1 = eigvecs[:, zero], eigvecs[:, ~zero]
            L1     = eigvals[~zero]
            if U0.size > 0 and not np.allclose(U0.T @ C_tilde @ U0, 0):
                _, v = eigh(U0.T @ C_tilde @ U0)
                w = U0 @ v[:, -1]
            else:
                #_, v = eigh(U1.T @ C_tilde @ U1, np.diag(L1))
                _, v = scipy.linalg.eigh(U1.T @ C_tilde @ U1, np.diag(L1))
                w = U1 @ v[:, -1]

        # enforce unit ℓ₂-norm
        return w / norm(w)

    @staticmethod
    def _contrast(mats, W):
        """Sum of off-diagonal Frobenius norms ‖diag-removed‖²."""
        acc = 0.0
        for R in mats:
            M = W.T @ R @ W
            acc += norm(M - np.diag(np.diag(M)), 'fro')**2
        return acc


class SignatureComputer:
    def __init__(self, n, d, MC_SAM_LEN):
        self.n = n
        self.d = d
        self.MC_SAM_LEN = MC_SAM_LEN
    
    def _lev_2_mat(self, sig2: jnp.ndarray) -> np.ndarray:
        # (d*d,) → (d, d)
        return sig2.reshape(self.d, self.d)

    def _lev_3_mat(self, sig3: jnp.ndarray) -> np.ndarray:
        # (d*d*d,) → (d, d, d)
        return sig3.reshape(self.d, self.d, self.d)
    
    def get_jpaths(self, S: np.ndarray) -> jnp.ndarray:
        """
        Converts the signal S into a JAX array of paths.
        :param S: Signal of shape (n, d)
        :return: JAX array of paths of shape (n_sub, MC_SAM_LEN, d)
        """
        n_sub = self.n // self.MC_SAM_LEN
        S = S[: n_sub * self.MC_SAM_LEN]
        paths = S.reshape(n_sub, self.MC_SAM_LEN, self.d)
        
        # NOTE: another sensible way to prepare for computing signature moments is basepointing. uncomment to try
        # paths = paths - paths[:, 0:1, :]  # Use 0:1 instead of 0 to keep the dimension
        
        # substract the mean of each path
        if self.MC_SAM_LEN > 1:
            paths = paths - np.mean(paths, axis=1, keepdims=True)  # Center each path around zero
        
        # Add a zero vector at the beginning of each path # need it for iid and MA
        zeros = np.zeros((n_sub, 1, self.d))
        paths = np.concatenate([zeros, paths], axis=1)  # (batch, T, d)

         # INSPECT IF THE paths we are cutting look sensible
        # for i in range(5):
        #     plt.plot(paths[i, :, 0], label=f'dim 1')
        #     plt.plot(paths[i, :, 1], label=f'dim 2')
        #     plt.plot(paths[i, :, 2], label=f'dim 3')
            
        # for i in range(550, 555):
        #     plt.plot(paths[i, :, 0], label=f'dim 1')
        #     plt.plot(paths[i, :, 1], label=f'dim 2')
        #     plt.plot(paths[i, :, 2], label=f'dim 3')
            
        # plt.title('Paths')
        # plt.xlabel('Time')
        # plt.ylabel('amplitude')
        # plt.legend()
        # plt.show()

        jpaths = jnp.array(paths)
        return jpaths

    def compute_lvl2_expected_sig(self, S: np.ndarray, test_mean_stationarity: bool) -> np.ndarray:
        """
        Computes the expected level-2 signature for a given signal S.
        :param S: Signal of shape (n, d)
        :return: Expected level-2 signature of shape (d, d)
        """        
        jpaths = self.get_jpaths(S)

        @jax.vmap
        def one_sig(path):
            # print("  shape of path in one_sig (should be MC_SAM_LEN (+ 1 if == 1),d): ", path.shape) # (MC_SAM_LEN, d)
            whole_signature = signax.signature(path, 2)
            return whole_signature

        batched = one_sig(jpaths)
        avg = jax.tree.map(lambda x: x.mean(axis=0), batched)
        avg_lvl2 = avg[self.d : self.d + self.d**2]
        return self._lev_2_mat(avg_lvl2.ravel())

    def compute_up_to_lvl3_expected_sig(self, S: np.ndarray, test_mean_stationarity: bool) -> np.ndarray:
        """
        Computes the expected signature for a given signal S of levels 2 and 3.
        :param S: Signal of shape (n, d)
        :return: Expected  signature of shape (d+1, d, d)
        """
        jpaths = self.get_jpaths(S)

        @jax.vmap
        def one_sig(path):
            # print("  shape of path in one_sig (should be MC_SAM_LEN (+ 1 if == 1),d): ", path.shape) # (MC_SAM_LEN, d)
            whole_signature = signax.signature(path, 3)
            return whole_signature

        batched = one_sig(jpaths)

        if test_mean_stationarity:
            # for each dimension, test if the mean is stationary
            for i in range(batched.shape[1]):
                series = batched[:, i]
                is_stationary, adf_stat, adf_p = test_mean_stationarity_adf(series)
                if not is_stationary:
                    warnings.warn(f"####Series {i} is not stationary: ADF stat={adf_stat}, p-value={adf_p}#####")
                else:
                    if adf_p > 0.001:
                        print(f"Series {i} is stationary, but we had a non-zero p value: ADF stat={adf_stat}, p-value={adf_p}")

        avg = jax.tree.map(lambda x: x.mean(axis=0), batched)

        # print("batched size: ", batched.shape) # (B, D + D**2 + D**3)
        avg_lvl1 = avg[:self.d]
        avg_lvl2 = avg[self.d : self.d + self.d**2]
        avg_lvl3 = avg[self.d + self.d**2 :]
        
        M_only_lvl2 = self._lev_2_mat(avg_lvl2.ravel())
        M_only_lvl2 = np.expand_dims(M_only_lvl2, axis=0)  # (1, d, d)
        M_only_lvl3 = self._lev_3_mat(avg_lvl3.ravel())  # (d, d, d)
        M_2and3 = np.concatenate([M_only_lvl2, M_only_lvl3], axis=0)
        
        return M_2and3

# === ADF test for mean stationarity ===
def test_mean_stationarity_adf(series,  alpha=0.05):
    """Test for mean stationarity using ADF and KPSS tests.

    Args:
        series (np.ndarray): Time series data.
        alpha (float, optional): Significance level for the tests. Defaults to 0.05.
    Returns:
        tuple: A tuple containing:
            - bool: True if the series is stationary, False otherwise.
            - float: ADF test statistic.
            - float: ADF p-value.
    """
    
    def run_adf(series):
        stat, pvalue, *_ = adfuller(series, autolag="AIC")
        return stat, pvalue

    adf_stat, adf_p = run_adf(series)
        
    return (adf_stat < 0 and adf_p < alpha), adf_stat, adf_p

def check_identifiability(S, kappa_thresh=0.01, mc_sam_len=None):
    # this can take a while for large n
    
    n, d = S.shape
    # Use the provided MC_SAM_LEN from the caller or default to the one passed in
    actual_mc_sam_len = mc_sam_len if mc_sam_len is not None else 5
    to_check = SignatureComputer(n, d, actual_mc_sam_len)
    M2and3 = to_check.compute_up_to_lvl3_expected_sig(S, test_mean_stationarity=False)
    Monly2 = M2and3[0, :, :]  # (d, d)
    Monly3 = M2and3[1:, :, :]  # (d,d,d)
    zeros = sum(abs(Monly3[k, k, k]) < kappa_thresh for k in range(d))
    if zeros > 1:
        warnings.warn("Identifiability violation: >1 zero third-moments")
        print("The diagonal third-moments were: ", [Monly3[k, k, k] for k in range(d)])
        print("The number of zero third-moments is: ", zeros)

    return



class ContrastCalculator:
    def __init__(self, n, d, MC_SAM_LEN, check_identifiability_criteria=True, verbose=False):
        self.n = n
        self.d = d
        self.MC_SAM_LEN = MC_SAM_LEN
        self.signaturecomputer = SignatureComputer(n, d, MC_SAM_LEN)
        self.check_identifiability_criteria = check_identifiability_criteria
        self.verbose = verbose

    def compute_N_fromM2(self, M2: np.ndarray) -> np.ndarray:
        """takes the signal nu and computes the N matrix, which is diagonal with entries sqrt<mu>_ii

        Args:
            M2 (np.ndarray): signal of shape (d, d)

        Returns:
            np.ndarray: N matrix of shape (d, d)
        """
        diag = np.sqrt(np.diag(M2))
        res = np.diag(diag)
        if np.linalg.cond(res) > 1e3:
            warnings.warn("N nearly singular")
            print("N matrix is: \n", res)
        return res

    def get_Mu_star_matrices(self, Mu_matrices: np.ndarray) -> np.ndarray:
        """returns the Mu_star matrices, which are the Mu matrices in the perfectly IC case.

        Args:
            Mu_matrices (np.ndarray): shape (d+1, d, d)

        Returns:
            np.ndarray: shape (d+1, d, d)
        """
        Mu_only2 = Mu_matrices[0]  # (d, d)
        Mu_only3 = Mu_matrices[1:]  # (d, d, d)
        Mu_star = np.zeros_like(Mu_matrices)
        Mu_star[0] = np.diag(np.diag(Mu_only2))  # D[0] is the diagonal of M2
        # for level 3
        for k in range(0, self.d):
            D = np.zeros_like(Mu_only3[k])
            D[k, k] = Mu_only3[k, k, k]
            Mu_star[k + 1] = D
        return Mu_star

    def compute_delta(self, S:np.ndarray) -> float:
        """computes the IC-defect delta for the sources S.

        Args:
            S (np.ndarray): shape(n, d)

        Returns:
            float: delta (IC-defect)
        """
        
        if self.check_identifiability_criteria:
            check_identifiability(S, mc_sam_len=self.MC_SAM_LEN)
        Mu_matrices = self.signaturecomputer.compute_up_to_lvl3_expected_sig(S, test_mean_stationarity=False)

        N = self.compute_N_fromM2(Mu_matrices[0])
        Ninv = np.linalg.inv(N)
        
        denominator = [1.0]
        denominator.extend(np.diag(N)) 
        denominator = np.array(denominator)  # (d+1,)

        Mu_star_matrices = self.get_Mu_star_matrices(Mu_matrices)

        difference = Mu_matrices - Mu_star_matrices
        
        sum_sq = 0.0
        for k in range( self.d + 1):
            k_th_component = np.linalg.norm(Ninv @ (difference[k] / denominator[k]) @ Ninv, 'fro')**2
            sum_sq += k_th_component
        res = np.sqrt(sum_sq)
        if self.verbose:
            print(" DEBUG FOR IC-defECT CALCULATOR")
            print("Mu matrices while computing delta (should be diagonal after 3.12.): ", Mu_matrices)
            print(" delta: ", res)
            print(" Mu matrices for the sources S are: (check visually if they are close to diagonal) \n", Mu_matrices)
            print(" END DEBUG FOR IC-defECT CALCULATOR")
            
        return res
    


class Optimizer:
    def __init__(self, n, d, MC_SAM_LEN, verbose=False):
        self.n = n
        self.d = d
        self.MC_SAM_LEN = MC_SAM_LEN
        self.signature_computer = SignatureComputer(n, d, MC_SAM_LEN)
        self.contrast_calculator = ContrastCalculator(n, d, MC_SAM_LEN)
        self.verbose = verbose

    def inverse_sqrt_psd_matrix(self, M: np.ndarray, eps=1e-16) -> np.ndarray:
        """Computes the inverse square root of a positive semi-definite matrix M.
        """
        M_np = np.array(M)
        M_np = (M_np + M_np.T)/2
        w, v = np.linalg.eigh(M_np)
        w = np.maximum(w, eps)
        result = (v * (1/np.sqrt(w))) @ v.T
        if not np.allclose(result @ result @ M_np, np.eye(M_np.shape[0]), atol=1e-3):
            warnings.warn("Inverse square root did not find the right matrix.")
        return result
    
    def compute_R_fromM2(self, M2: np.ndarray) -> np.ndarray:
        """takes the second signature moments and returns the whitenting matrix R.
        Args:
            M2 (np.ndarray): second order coredinates of shape (d, d)

        Returns:
            np.ndarray: R matrix of shape (d, d)
        """
        C = 0.5*(M2 + M2.T)
        C_np = np.array(C)
        if np.linalg.cond(C_np) > 1e3:
            warnings.warn("C nearly singular")
        R = self.inverse_sqrt_psd_matrix(C_np)
        
        return np.array(R)
    
    def contrast_from_Mu(self, Mu_matrices: np.ndarray) -> float:
        """Computes the contrast from the Mu matrices.

        Args:
            Mu_matrices (np.ndarray): Mu matrices of shape (d+1, d, d)

        Returns:
            float: Contrast value
        """
        sum_sq = 0.0
        for k in range(0, self.d + 1):
            M_no_diag = Mu_matrices[k] - np.diag(np.diag(Mu_matrices[k]))
            k_th_component = np.linalg.norm(M_no_diag, 'fro')**2
            sum_sq += k_th_component
                
        return np.sqrt(sum_sq)
    
    def phi_from_signal(self, theta, R, X) -> float:
        
        thetaR = theta @ R  # (d, d)
        thetaR_X = X @ thetaR.T  # (n, d)
        Mu_thetaRX = self.signature_computer.compute_up_to_lvl3_expected_sig(thetaR_X, test_mean_stationarity=False)
        if self.verbose:
            print("\n debugging phi_from_signal")
            print(" Mu_thetaRX: ", Mu_thetaRX)
        Mu2 = Mu_thetaRX[0]  # (d, d)
        Mu3 = Mu_thetaRX[1:]  # (d, d, d)
        N = self.contrast_calculator.compute_N_fromM2(Mu2)
        Ninv = np.linalg.inv(N)
        denominator = np.diag(N)
        x_stats = np.zeros((self.d + 1, self.d, self.d))
        x_stats[0] = Ninv @ (Mu2 - np.diag(np.diag(Mu2))) @ Ninv  # level 2
        for k in range(0, self.d):
            x_stats[k + 1] = Ninv @ ((Mu3[k] - np.diag(np.diag(Mu3[k]))) / denominator[k]) @ Ninv
        
        phi2 = 0.0
        for k in range(0, self.d + 1):
            phi2 += np.linalg.norm(x_stats[k], 'fro')**2
            if self.verbose:
                print(f"  k = {k}, norm of x_stats[k]: {np.linalg.norm(x_stats[k], 'fro')}")
        phi = np.sqrt(phi2)
        
        return phi
    
    def compute_x_statistics(self, thetaX: np.ndarray) -> np.ndarray:
        """Computes the x-statistics for the whitened signal X.
        Args:
            X (np.ndarray): shape (n, d)

        Returns:
            np.ndarray: x-statistics of shape (d+1, d, d)
        """
        M2uand3 = self.signature_computer.compute_up_to_lvl3_expected_sig(thetaX, test_mean_stationarity=False)
        Mu_only2 = M2uand3[0]  # (d, d)
        Mu_only3 = M2uand3[1:]  # (d, d, d)
        

        N = self.contrast_calculator.compute_N_fromM2(Mu_only2)
        Ninv = np.linalg.inv(N)

        denominator = np.diag(N)
        x_stats = np.zeros((self.d + 1, self.d, self.d))
        x_stats[0] = Ninv @ Mu_only2 @ Ninv  # level 2
        for k in range(0, self.d):
            x_stats[k + 1] = Ninv @ (Mu_only3[k] / denominator[k]) @ Ninv
        if self.verbose:
            print(" DEBUG FOR OPTIMIZER")
            print(" the x_statistics are: ", x_stats)
            print(" N matrix is: \n", N)
            print(" M2uand3: \n", M2uand3)
            print(" for the above matrices, the value of the contrast is: ", self.contrast_from_Mu(M2uand3))
            print(" END DEBUG FOR OPTIMIZER")
        return x_stats
        
    def RICA(self, X: np.ndarray) -> np.ndarray:
        """Performs RICA on the signal X.
        Args:
            X (np.ndarray): shape (n, d)
        Returns:
            np.ndarray: RICA unmixing matrix W_x of shape (d, d)
        """

        M2 = self.signature_computer.compute_lvl2_expected_sig(X, test_mean_stationarity=False)
        R = self.compute_R_fromM2(M2)
        if self.verbose:
            print(" R matrix: \n", R)

        X_whitened = np.dot(X, R.T)
                
        x_statistics = self.compute_x_statistics(X_whitened)

        if self.verbose:
            print(" x statistics: \n", x_statistics)

        jd = FlexJDReal(max_iter=1000_000, tol_angle=1e-12, offdiag_tol=1e-4, norm_tol=1e-4)
        V, ok, info = jd.fit(list(x_statistics))

        theta_hat = V.T  # we should transpose, because FlexJD returns V s.t. V.T [M] V = D and we want theta [M] theta.T = D
        print(" condition number of theta_hat: ", cond(theta_hat))
        print(" norms of rows of theta_hat: ", norm(theta_hat, axis=1))
        W_x = theta_hat @ R
        if not ok:
            warnings.warn(
                f"FlexJD failed: off-diag={info['offdiag']:.2e}, "
                f"row norms of theta_hat={info['colnorms']}"
            )
        if self.verbose:
            print("FlexJD converged in", info['iters'], "iterations")
            print("Off-diagonal norm:", info['offdiag'])
            print("Row norms of theta_hat:", info['colnorms'])
            print("Condition number of theta_hat:", cond(theta_hat))
        return W_x




def main():
    np.random.seed(0)
    d, nr_of_mc_samples_for_moment_computation = 3, 1_000_000 # 5_000_000 gives even better results. the more the better.
    MC_SAM_LENS = [15, 30, 60] 

    rica_rel_err_vs_len = []
    fastica_rel_err_vs_len = []
    delta_vs_len = []

    A = jnp.array(
         [[-0.37842,     0.91451844, -0.14301863],
        [-0.69043073, -0.3817858,  -0.61444691],
        [-0.61652552, -0.13377454,  0.77588701]])

    
    A_inv = np.linalg.inv(A)
    
    for MC_SAM_LEN in MC_SAM_LENS:
        print(f"\n\n\nfor MC_LEN = {MC_SAM_LEN}")
        signal_generator = SignalGenerator()
        # --- generate sources and their mixtures -------------------------------
        # EXAMPLE 1: IID
        S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='iid') #$ ( generates diagonal M2 and M3)
        
        # EXAMPLE 2: ARMA process
        # S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='ARMA')

        # EXAMPLE3: OU process (requires the preprocessing step)
        # S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='OU')

        # length_of_S = S.shape[0]
        # burnin = int(0.1 * length_of_S)  # 10% burnin
        # S = S[burnin:]
        # ---------------------------------

        # EXAMPLE 4: gumbelMA signal
        # S = signal_generator.sample_s(d, n=nr_of_mc_samples_for_moment_computation * MC_SAM_LEN, ts_type='gumbelMA')
        # ---------------------------------
 
                                       

        # ------------------------------------------------------------------------
        length_of_S = S.shape[0] 
        contrastcalculator = ContrastCalculator(length_of_S, d, MC_SAM_LEN, False, verbose=False)
        delta_for_this_MC_SAM_LEN = contrastcalculator.compute_delta(S)
        print("delta(S) for MC_SAM_LEN =", MC_SAM_LEN, "is", delta_for_this_MC_SAM_LEN)
        delta_vs_len.append(delta_for_this_MC_SAM_LEN)

        X = S @ A.T  # Mix the sources S with the mixing matrix A
        # ---------------------------------
        ricaoptimizer = Optimizer(length_of_S, d, MC_SAM_LEN, verbose=False)
        I_x = ricaoptimizer.RICA(X)
        rica_rel_err = get_rel_err(I_x, A_inv, *M_IplusE(I_x, A))
        
        print(" $$$ results for MC_SAM_LEN =", MC_SAM_LEN, " $$$")
        print("delta(S) =", delta_for_this_MC_SAM_LEN)
        print("relative error of RICA:",
              rica_rel_err)
        print("I_x @ A: \n", I_x @ A)
        
        # sobi: 
        _, _, W_sobi = sobi_algo(X.T, num_lags=MC_SAM_LEN, eps=1e-6)
        
        M_sobi, E_sobi = M_IplusE(W_sobi, A)
        sobi_rel_err = get_rel_err(W_sobi, A_inv, *M_IplusE(W_sobi, A))
        sobi_abs_err = jnp.linalg.norm(E_sobi, 'fro')
        
        rica_rel_err_vs_len.append(rica_rel_err)

        # compare to FastICA
        ica    = FastICA(n_components=d, max_iter=100_000, random_state=0)
        W_ica  = ica.fit_transform(X.T)
        fastica_rel_err = get_rel_err(W_ica, A_inv, *M_IplusE(W_ica, A))
    
        print("rel_err of FastICA:",
              fastica_rel_err)
        print("W_ica @ A: \n", W_ica @ A)
        print("W_sobi @ A: \n", W_sobi @ A)
        print("rel_err of SOBI: ", sobi_rel_err)
        
        print(" $$$$$$$$$$$$$$$$$$$")
        fastica_rel_err_vs_len.append(fastica_rel_err)

        print("\n\n")

    print(" \n\n")
    print("RESULTS: \n\n")
    print("RICA relative errors: ", rica_rel_err_vs_len)
    print("FastICA relative errors: ", fastica_rel_err_vs_len)
    print("Delta values: ", delta_vs_len)
    

if __name__ == "__main__":
    main()