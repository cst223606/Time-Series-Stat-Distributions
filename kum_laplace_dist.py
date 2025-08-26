import numpy as np
from scipy.optimize import minimize
from scipy.stats import laplace
import warnings

def kumaraswamy_laplace_pdf(x, mu, sigma, a, b):
    # PDF 
    z = (x - mu) / sigma
    abs_z = np.abs(z)
    
    # avoid numerical underflow
    with np.errstate(over='ignore', under='ignore'):
        base = 1.0 - np.exp(-abs_z)
        term1 = base**(a - 1)
        term2 = (1.0 - base**a)**(b - 1)

    pdf = (a * b / sigma) * term1 * term2
    return pdf

def neg_log_likelihood(params, data):
    # negative log likelihood function
    mu, sigma, a, b = params
    if sigma <= 0 or a <= 0 or b <= 0:
        return np.inf  # constraint
    
    pdf_values = kumaraswamy_laplace_pdf(data, mu, sigma, a, b)
    log_likelihood = np.log(pdf_values + 1e-12).sum()  # prevent log(0)
    return -log_likelihood

def fit_kumaraswamy_laplace(data, initial_guess=None):
    """
    fit Kumaraswamy-Laplace distribution using MLE
    return params: mu, sigma, a, b
    """
    # initialize guess
    if initial_guess is None:
        mu_init = np.median(data)
        sigma_init = laplace.scale(data)
        a_init = 1.0
        b_init = 1.0
        initial_guess = [mu_init, sigma_init, a_init, b_init]
    
    bounds = [
        (-np.inf, np.inf),   # mu
        (1e-6, np.inf),      # sigma
        (1e-6, np.inf),      # a
        (1e-6, np.inf)       # b
    ]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            fun=neg_log_likelihood,
            x0=initial_guess,
            args=(data,),
            bounds=bounds,
            method='L-BFGS-B'
        )
    
    if not result.success:
        raise RuntimeError(f"MLE fitting failed: {result.message}")
    
    mu_opt, sigma_opt, a_opt, b_opt = result.x
    return mu_opt, sigma_opt, a_opt, b_opt
