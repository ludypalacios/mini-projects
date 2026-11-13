"""mp3lib.py - helper functions for ECE 3715 Mini-Project 3."""

import numpy as np

# Note: we use np.linalg.solve instead of np.linalg.inv everywhere.
# solve is faster and more accurate, especially when the matrix is
# close to singular (like Sigma_xx in Step 8).


def sample_mean_cov(X):
    """Return the sample mean and unbiased covariance of X with shape (N, d)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    N = X.shape[0]

    mean_vector = X.mean(axis=0)
    Xc = X - mean_vector                  # center the data
    cov_matrix = (Xc.T @ Xc) / (N - 1)    # divide by N-1 for unbiased estimate
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)  # force exact symmetry
    return mean_vector, cov_matrix


def is_valid_covariance(S, tol=1e-10):
    """Return True if S is symmetric and positive semidefinite."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        return False

    scale = max(1.0, np.max(np.abs(S)))

    # Symmetric check
    if np.max(np.abs(S - S.T)) > tol * scale:
        return False

    # PSD check: eigenvalues must be >= 0 (allow tiny rounding error)
    eigvals = np.linalg.eigvalsh(S)
    return bool(eigvals.min() >= -tol * scale)


def gaussian_pdf(x, mu, Sigma):
    """Multivariate normal density at x (one point (d,) or many points (N, d))."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    d = mu.shape[0]

    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(-1, 1) if d == 1 else x.reshape(1, d)

    if not np.allclose(Sigma, Sigma.T, rtol=1e-12, atol=0.0):
        raise np.linalg.LinAlgError("Sigma must be symmetric")
    eigvals = np.linalg.eigvalsh(Sigma)
    # A rank-deficient covariance can have a small positive computed determinant.
    tol = d * np.finfo(float).eps * np.max(np.abs(eigvals))
    if eigvals[0] <= tol:
        raise np.linalg.LinAlgError("Sigma must be positive definite at working precision")
    det = np.linalg.det(Sigma)

    diff = x - mu
    # Mahalanobis distance: (x-mu)^T Sigma^-1 (x-mu), using solve instead of inv
    Z = np.linalg.solve(Sigma, diff.T)
    maha = np.sum(diff * Z.T, axis=1)

    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det)
    return norm_const * np.exp(-0.5 * maha)


def whitening_matrix(Sigma):
    """Return W = Lambda^(-1/2) V^T so that W Sigma W^T = I."""
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    eigvals, V = np.linalg.eigh(Sigma)    # eigh for symmetric matrices

    if eigvals.min() <= 1e-12:
        raise np.linalg.LinAlgError("Sigma is singular, cannot whiten")

    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    return Lambda_inv_sqrt @ V.T


def lmmse_weights(Sigma_xx, Sigma_sx):
    """Return the LMMSE weights w = Sigma_xx^-1 Sigma_sx."""
    Sigma_xx = np.atleast_2d(np.asarray(Sigma_xx, dtype=float))
    Sigma_sx = np.asarray(Sigma_sx, dtype=float)
    # solve instead of inv: more accurate when Sigma_xx is nearly singular
    return np.linalg.solve(Sigma_xx, Sigma_sx)
