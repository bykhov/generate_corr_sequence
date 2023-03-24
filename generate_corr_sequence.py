import numpy as np
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import norm, uniform
from scipy.special import erfinv, eval_hermitenorm
from scipy.integrate import quad
import scipy.io


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

    integrationMin = dist_obj.ppf(1e-15)  # The minimum value of x for which Fy(x) is defined
    integrationMax = dist_obj.ppf(1 - 1e-15)  # The maximum value of x for which Fy(x) is defined
    d = []
    num_zeros = 0  # count zeroes in polynomials
    numberOfHermitePoly = firstDegreeOfHermitePoly
    while True:
        I = quad(integration_function, integrationMin, integrationMax, args=(numberOfHermitePoly, dist_obj))
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
    return y * hermiteProb(numberOfHermitePoly,
                           h(Fy(y))) * fx(h(Fy(y))) * hdot(fy(y),
                            Fy(y))  # The integration function


def find_ro_x(d, desiredACF):
    """
    This function will find the approximation of the target ACF.

    Input:
    d          - d coefficients.
    desiredACF - The target ACF.

    Output:
    ro_x       - Approximation of the target ACF with the usage of the
                 d coefficients, size = (length(desiredACF), ).
    """
    ro_x = []
    ### calculation of ro x
    for Roy in desiredACF:
        coeff = d[::-1]
        coeff.append(-Roy)
        Rox = np.poly1d(coeff)
        roots = np.roots(Rox)[np.iscomplex(np.roots(Rox)) == False]
        roots = np.real(roots)[np.real(roots) >= -1.05]
        roots = roots[roots <= 1.05]
        ro_x.extend(np.real(roots))
    return ro_x


def findFilter(ro_x):
    """
    This function finds the appropriate AR filter to adjust the target ACF to a Gaussian sequence.

    Parameters:
        ro_x (ndarray): Approximation of the target ACF

    Returns:
        ndarray: The filter to adjust the target ACF of a Gaussian sequence.
    """

    H = np.array(ro_x)
    ry = -H[1:]
    ry = np.append(ry, 0)
    a = scipy.linalg.solve_toeplitz(H, ry)
    a = np.append(1, a.T)
    return a


def get_ranked_sequence(x, z):
    """
    Applies the rank matching operation to transform a sequence with a Gaussian autocorrelation function to a target distribution
    sequence with the desired autocorrelation function.

    Args:
    - x: NumPy array representing a Gaussian sequence with the desired autocorrelation function.
    - z: NumPy array representing a target distribution sequence without the desired autocorrelation function.

    Returns:
    - y: NumPy array of shape (len(x),) representing the target distribution sequence with the desired autocorrelation function.
    """
    I = np.argsort(x)
    y = np.sort(z)
    y[I] = y.copy()
    return y


def drawDebugPlots(Xn, x, z, y, desiredACF):
    """
    This Function will draw plots of the achieved PDF and ACF.

    Input:
    Xn  - Vector of normal distributed samples.
    x   - Vector of the target ACF function.
    z   - Number of samples in the output sequence, default is 2^20.
    y   - A seed for the random number generator.

    """
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, ec="k")
    XnCorr = sm.tsa.acf(Xn.reshape(-1), nlags=len(desiredACF) - 1, fft=True)  # normal sequence acf
    xCorr = sm.tsa.acf(x, nlags=len(desiredACF) - 1, fft=True)
    zCorr = sm.tsa.acf(z, nlags=len(desiredACF) - 1, fft=True)
    yCorr = sm.tsa.acf(y, nlags=len(desiredACF) - 1, fft=True)
    plt.rcParams.update({
        "figure.facecolor": (1.0, 1.0, 1.0, 1),
        "axes.facecolor": (1.0, 1.0, 1.0, 1),
        "savefig.facecolor": (1.0, 1.0, 1.0, 1),
    })

    plt.title('Pre and Post filter PDF')
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams['figure.dpi'] = 100
    plt.hist(Xn[0], bins='auto', label="Pre-filtered PDF", **kwargs)
    plt.hist(x, bins='auto', label="Post-filtered PDF", **kwargs)
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.title('Pre and Post filter ACF')
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams['figure.dpi'] = 100
    plt.plot(XnCorr, label="Pre-filtered Gaussian ACF")
    plt.plot(xCorr, label="Post-filtered Gaussian ACF")
    plt.xlabel('Lags')
    plt.ylabel('ACC')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.title('resulting PDF')
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams['figure.dpi'] = 100
    plt.hist(z, bins='auto', label="Pre-ranked target PDF", **kwargs)
    plt.hist(y, bins='auto', label="Post-ranked target PDF", **kwargs)
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.title('resulting ACF')
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams['figure.dpi'] = 100
    plt.plot(zCorr, label='Pre-ranked target ACF')
    plt.plot(yCorr, '-.', label='Achieved ACF')
    plt.plot(desiredACF, '--', alpha=0.5, label='Target ACF')
    plt.xlabel('Lags')
    plt.ylabel('ACC')
    plt.legend()
    plt.grid(True)
    plt.show()


def gen_corr_sequence(dist_obj=uniform,
                           desiredACF=1 - np.minimum(np.arange(0, 100), 100) / 100,
                           L=2 ** 20,
                           seed=42,
                           debug=False):
    """
    Generate a sequence of samples with a specified autocorrelation function and probability distribution.

    Args:
    - dist_obj: scipy object that represents the desired probability distribution. Default is uniform.
    - desiredACF: vector of values representing the target autocorrelation function. Default is None, which implies
                  a white noise process with no correlation.
    - L: the desired length of the output sequence. Default is 2**20.
    - seed: seed for the random number generator.
    - debug: boolean flag indicating whether to produce debugging plots or not. Default is False.

    Returns:
    - y: a NumPy array of shape (L,) containing the generated sequence of samples with the desired autocorrelation
         function and probability distribution.
    """

    # initialize the random number generator
    np.random.seed(seed)

    Xn = np.random.randn(1, L)  # normal sequence

    d = findCoeff(dist_obj)

    ro_x = find_ro_x(d, desiredACF)

    a = findFilter(ro_x)  # finding the appropriate filter to get the target ACF

    x = signal.lfilter([1.0], a, Xn).reshape(-1)

    z = dist_obj.rvs(size=L)  # Desired distribution sequence

    y = get_ranked_sequence(x, z)  # rank matching the sequence

    if debug:
        drawDebugPlots(Xn, x, z, y, desiredACF)

    return y


if __name__ == "__main__":
    signal = gen_corr_sequence(debug=True)
