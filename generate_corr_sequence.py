import numpy as np
from numpy import ndarray
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot_2samples, qqplot
import scipy
from scipy.signal import lfilter
from scipy.stats import norm, uniform, triang, probplot
from scipy.special import erfinv, eval_hermitenorm
from scipy.integrate import quad
from scipy.optimize import minimize, basinhopping, shgo
import matplotlib.pyplot as plt
import warnings


def findCoeff(dist_obj=uniform):
    """
    This function will find the d coefficients.

    Input:
    dist_obj - A scipy.stats target distribution we want to get d coefficients of.

    Output:
    d        - A list of d coefficients, np.shape = (numberOfHermitePoly, ),
               representing the chosen distribution.
    """
    firstDegreeOfHermitePoly = 1
    threshold_for_small_d_coeff = 10 ** -6
    max_amount_of_poly = 8

    amount_of_consecutive_zeroes = 2
    integration_min = dist_obj.ppf(1e-15)  # The minimum value of x for which Fy(x) is defined
    integration_max = dist_obj.ppf(1 - 1e-15)  # The maximum value of x for which Fy(x) is defined
    # patch for lognormal distribution
    try:  # not all distributions have dist attribute
        if dist_obj.dist.name == 'lognorm':
            integration_max_10 = dist_obj.mean() + dist_obj.std()*10
            if integration_max > integration_max_10:
                integration_max = integration_max_10
            integration_min_10 = dist_obj.mean() - dist_obj.std()*10
            if integration_min < integration_min_10:
                integration_min = integration_min_10
    except:  # nothing to do here
        pass

    d = []
    num_zeros = 0  # count zeroes in polynomials
    numberOfHermitePoly = firstDegreeOfHermitePoly
    while True:
        I = quad(integration_function, integration_min, integration_max, args=(numberOfHermitePoly, dist_obj))
        current_d = I[0] ** 2 / np.math.factorial(numberOfHermitePoly)
        if numberOfHermitePoly > firstDegreeOfHermitePoly:
            d_check = current_d / sum(d)
            if d_check < threshold_for_small_d_coeff:
                num_zeros += 1
            else:
                num_zeros = 0  # reset zeros counter
            if num_zeros == amount_of_consecutive_zeroes or numberOfHermitePoly == max_amount_of_poly:
                break
        d.append(current_d)
        numberOfHermitePoly += 1
    d = [x / sum(d) for x in d]
    return d


def integration_function(y, numberOfHermitePoly, dist_obj):
    """
    This function will find the integration function that helps in finding
    the d coefficients.

    Input:
    y                   - Integration variable (comes from quad() function).
    numberOfHermitePoly - Number of the order of the hermite polynomial.
    dist_obj            - A scipy.stats target distribution.

    Integration function that results with the d coefficients.
    """
    fx = lambda x: norm.pdf(x)  # normal distribution PDF
    fy = lambda x: dist_obj.pdf(x)  # desired distribution PDF
    Fy = lambda x: dist_obj.cdf(x)  # desired distribution CDF
    h = lambda x: np.sqrt(2) * erfinv(2 * x - 1)
    hdot = lambda x, y: np.sqrt(2 * np.pi) * x * np.exp(erfinv(2 * y - 1) ** 2)
    hermiteProb = lambda n, x: eval_hermitenorm(n, x)  # hermite polynom
    return y * hermiteProb(numberOfHermitePoly, h(Fy(y))) * fx(h(Fy(y))) * hdot(fy(y),
                                                                                Fy(y))  # The integration function


def find_ro_x(d, target_acf):
    """
    This function will find the approximation of the target ACF.

    Input:
    d          - d coefficients.
    target_acf - The target ACF.

    Output:
    ro_x       - Approximation of the target ACF with the usage of the
                 d coefficients, size = (length(target_acf), ).
    """
    ro_x = []
    ### calculation of ro x
    for Roy in target_acf:
        coeff = d[::-1]
        coeff.append(-Roy)
        Rox = np.poly1d(coeff)
        roots = np.roots(Rox)[np.iscomplex(np.roots(Rox)) == False]
        roots = np.real(roots)[np.real(roots) >= -1.05]
        roots = roots[roots <= 1.05]
        ro_x.extend(np.real(roots))
    return ro_x


# %% MSE cost function for ARMA optimization
def mse(result: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean Squared Error
    input:
        result - theoretical ACF
        target - target ACF
    output:
        score - clipped MSE score"""
    score = np.mean(np.square(np.clip(result, -1e5, 1e5) - target))
    return score


# %% ar, ma <-> vector
def ar_ma_to_vector(ar: np.ndarray, ma: np.ndarray) -> np.ndarray:
    """Convert ARMA(p,q) coefficients to vector"""
    return np.concatenate((ar[1:], ma[1:]))


def vector_to_ar_ma(x: np.ndarray) -> (np.ndarray, np.ndarray):
    """Convert vector to ARMA"""
    ar = np.zeros(4)
    ar[0] = 1
    ar[1:] = x[:3]
    ma = np.zeros(4)
    ma[0] = 1
    ma[1:] = x[3:]
    return ar, ma


# %% theoretical ACF
def my_arma_acf(x, lags: int) -> np.ndarray:
    """ARMA(3,3) model
    output: theoretical ACF
    """
    ar, ma = vector_to_ar_ma(x)
    return arma_acf(ar, ma, lags)


# %% Optimization cost function
def cost_function(x: np.ndarray, target_acf: np.ndarray) -> np.ndarray:
    """Cost function for ARMA optimization
    input:
    x - vector of ARMA coefficients
    """
    lags = len(target_acf)
    try:
        R = my_arma_acf(x, lags)  # theoretical ACF
    except:  # if ARMA coefficients are not stable - replace by optimization constraints in the future
        return np.array([1e10])
    return mse(R, target_acf)


# %% ARMA model
def get_arma_filter(target_acf: np.ndarray, debug: bool = False) -> (np.ndarray, np.ndarray):
    lags = len(target_acf)
    # x0 = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    while True:
        x0 = np.random.rand(6)
        bounds = [(-10, 10)] * 6
        # res = shgo(cost_function, bounds=bounds,
        #            iters=3,
        #            args=(lags, target_acf),
        #            options={'disp': False})
        res = basinhopping(cost_function, x0,
                           niter=300,
                           minimizer_kwargs={'args': target_acf})
        # res = minimize(cost_function, x0, args=(lags, target_acf),
        #                method='nelder-mead',
        #                options={'adaptive': True,
        #                         'fatol': 1e-5,
        #                         'xatol': 1e-5,
        #                         'maxfev': 5000,
        #                         'maxiter': 5000,
        #                         'disp': debug})
        # if debug:
        #     print(res.success, res.fun, res.message, res.nit)
        if res.fun != 1e10:
            if res.fun > 0.01:
                # warnings.simplefilter('module',UserWarning)
                warnings.warn('The optimization did not converge to the target ACF.')
            break

    ar, ma = vector_to_ar_ma(res.x)
    return ar, ma


def get_ranked_sequence(x, z):
    """
      This function will apply the rank matching operation.

      Input:
      x - Gaussian sequence with the desired ACF.
      z - Target distribution sequence without the desired ACF.

      Output:
      y - Target distribution sequence with the desired ACF, np.shape = (length(x), ).
    """
    I = np.argsort(x)
    y = np.sort(z)
    y[I] = y.copy()
    return y


def debugPlots(dist_obj, target_acf: np.ndarray, y: np.ndarray, fileName: str = None):
    """
    This Function will draw plots of the achieved PDF and ACF.

    Input:
    dist_obj - The target distribution object
    target_acf - The target ACF
    y - The final sequence with matching ACF and distribution to the target

    """
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, ec="k")
    yCorr = sm.tsa.acf(y, nlags=len(target_acf) - 1, fft=True)
    plt.rcParams.update({
        "figure.facecolor": (1.0, 1.0, 1.0, 1),
        "axes.facecolor": (1.0, 1.0, 1.0, 1),
        "savefig.facecolor": (1.0, 1.0, 1.0, 1),
        "figure.dpi": 300,
        "figure.figsize": (4, 3),
        "font.size": 10,
    })

    pdfxAxis = np.linspace(max(dist_obj.ppf(1e-15), dist_obj.mean() - dist_obj.std()*5),
                           min(dist_obj.ppf(1 - 1e-15), dist_obj.mean() + dist_obj.std()*5),
                           1000)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Resulting PDF')
    plt.plot(pdfxAxis, dist_obj.pdf(pdfxAxis), label="Required PDF")
    plt.hist(y, bins='auto', label="Simulated PDF", **kwargs)
    plt.xlim(
        (np.min(pdfxAxis), np.max(pdfxAxis)))  # handle long-tail distributions
    plt.xlabel('x')
    plt.ylabel('$f_x(x)$')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Resulting ACF')
    plt.plot(target_acf, '-', alpha=0.5, label='Required ACF')
    plt.plot(yCorr, '--', label='Simulated ACF')
    plt.xlabel('Lags')
    plt.ylabel('ACF')
    plt.xlim([0, len(target_acf) - 1])
    plt.grid()
    plt.legend(loc='upper right')
    if fileName is not None:
        plt.tight_layout()
        plt.savefig(fileName)
    plt.show()


def gen_corr_sequence(dist_obj=uniform,
                      target_acf=1 - np.minimum(np.arange(0, 100), 100) / 100,
                      L: int = 2 ** 20,
                      seed=None,
                      debug: bool = False,
                      plot_figures_name: str = None):
    """
    This Function will create a vector (sequence) of samples with the desired
    AutoCorrelation Function and distribution.

    Input:
    dist_obj   - The desired distribution, default is uniform.
    target_acf - Vector of the target ACF function.
    L          - Number of samples in the output sequence, default is 2^20.
    seed       - Seed for the random number generator.
    debug      - Whether to plot debugging graphs or not, default is False.

    Output:
    y          - An np.ndarray of samples with desired ACF and PDF, np.shape = (L, ).

    Example:
        import numpy as np
        from scipy.stats import nakagami
        from scipy.special import j0
        from generate_corr_sequence import gen_corr_sequence

        dist_obj = nakagami(nu=1)
        m = np.arange(0, 100)
        target_acf = np.array(j0(0.1 * np.pi * abs(m)))
        signal = gen_corr_sequence(
            dist_obj=dist_obj,
            target_acf=target_acf,
            debug=True)
    """
    if len(target_acf) > L -1:
        raise ValueError('The length of the target ACF vector is too long for the required sequence length.')

    if plot_figures_name is not None and debug is False:
        warnings.warn('plot_figures_name is not used when debug is False.')

    if seed is not None:
        np.random.seed(seed)

    d = findCoeff(dist_obj)

    ro_x = find_ro_x(d, target_acf)

    Xn = np.random.normal(size=L)  # normal sequence

    ar, ma = get_arma_filter(ro_x, debug)  # finding the appropriate filter to get the target ACF

    x = lfilter(ma, ar, Xn)  # applying the filter to the normal sequence

    z = dist_obj.rvs(size=L)  # Desired distribution sequence

    y = get_ranked_sequence(x, z)  # rank matching the sequence

    if debug:
        debugPlots(dist_obj, target_acf, y, plot_figures_name)

    return y


# %% Example
if __name__ == "__main__":
    from scipy.stats import nakagami
    from scipy.special import j0

    m = np.arange(0, 100)
    target_acf = np.array(j0(0.1 * np.pi * abs(m)))
    signal = gen_corr_sequence(
        dist_obj=nakagami(nu=1),
        target_acf=target_acf,
        debug=True)

# %%
