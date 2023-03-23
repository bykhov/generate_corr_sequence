import scipy
from scipy.stats import rayleigh, triang, laplace, uniform, expon
from scipy.special import j0
from scipy import signal
import statsmodels.api as sm
import matplotlib.pyplot as plt
from gen_corr_sequence import findCoeff, find_ro_x, findFilter, get_ranked_sequence

def test_d():
    rtol = 1e-4  # relative tolerance

    dRayleighCode = np.round(findCoeff(dist_obj=rayleigh)[:7], 7)
    dRayleighArticle = np.array([0.9724690, 0.0264400, 0.0010467, 0.0000113, 0.0000310, 0.0000017, 0.0000007])
    np.testing.assert_allclose(dRayleighArticle, dRayleighCode, rtol=rtol, atol=0,
                               err_msg="Rayleigh coeffs don't match")

    dLaplaceCode = np.round(findCoeff(dist_obj=laplace)[:7], 7)
    dLaplaceArticle = np.array([0.9632098, 0, 0.0352063, 0, 0.0013255, 0, 0.0002592])
    np.testing.assert_allclose(dLaplaceArticle, dLaplaceCode, rtol=rtol, atol=0, err_msg="Laplace coeffs don't match")

    dUniformCode = np.round(findCoeff(dist_obj=uniform)[:7], 7)
    dUniformArticle = np.array([0.9550627, 0, 0.0397943, 0, 0.0044768, 0, 0.0006662])
    np.testing.assert_allclose(dUniformArticle, dUniformCode, rtol=rtol, atol=0, err_msg="Uniform coeffs don't match")

    dExponCode = np.round(findCoeff(dist_obj=expon)[:7], 7)
    dExponArticle = np.array([0.8157660, 0.1773910, 0.0066847, 0.0001343, 0.0000169, 0.0000073, 0.0])
    np.testing.assert_allclose(dExponArticle, dExponCode, rtol=rtol, atol=0, err_msg="Exponential coeffs don't match")

    dTriangleCode = np.round(findCoeff(dist_obj=triang(c=0.5, loc=-np.sqrt(6), scale=2 * np.sqrt(6)))[:7], 7)
    dTriangleArticle = np.array([0.9927701, 0.0, 0.0071243, 0.0, 0.0000010, 0.0, 0.0000100])
    np.testing.assert_allclose(dTriangleArticle, dTriangleCode, rtol=rtol, atol=0,
                               err_msg="Triangle coeffs don't match")


def test_ACF(dist_obj=rayleigh):
    rtol = 0.33  # relative tolerance

    d = list(np.round(findCoeff(dist_obj=dist_obj), 7))
    m = np.arange(0, 100)

    LinearTargetACF = 1 - np.minimum(m, 100) / 100
    LinearCalcACF = np.array(find_ro_x(d, LinearTargetACF))
    np.testing.assert_allclose(LinearTargetACF, LinearCalcACF, rtol=rtol, atol=0, err_msg="Linear ACF doesn't match")

    ExponTargetACF = np.exp(-0.05 * np.abs(m))
    ExponCalcACF = np.array(find_ro_x(d, ExponTargetACF))
    np.testing.assert_allclose(ExponTargetACF, ExponCalcACF, rtol=rtol, atol=0, err_msg="Exponential ACF doesn't match")

    ExponCosTargetACF = np.exp(-0.05 * np.abs(m)) * np.cos(0.25 * np.abs(m))
    ExponCosCalcACF = np.array(find_ro_x(d, ExponCosTargetACF))
    np.testing.assert_allclose(ExponCosTargetACF, ExponCosCalcACF, rtol=rtol, atol=0,
                               err_msg="exp*cos ACF doesn't match")

    BesselTargetACF = np.array(j0(0.1 * np.pi * abs(m)))
    BesselCalcACF = np.array(find_ro_x(d, BesselTargetACF))
    np.testing.assert_allclose(BesselTargetACF, BesselCalcACF, rtol=rtol, atol=0, err_msg="Bessel ACF doesn't match")


def DrawTestPlots(dist_obj=rayleigh, dist_name='rayleigh'):
    L = 2 ** 20
    m = np.arange(0, 100)
    d = findCoeff(dist_obj)
    Xn = np.random.randn(1, L)  # normal sequence
    z = dist_obj.rvs(size=L)  # Desired distribution sequence
    names = ['linear', 'exp', 'exp*cos', 'bessel']
    targetACFs = [1 - np.minimum(m, 100) / 100, np.exp(-0.05 * np.abs(m)),
                  np.exp(-0.05 * np.abs(m)) * np.cos(0.25 * np.abs(m)), np.array(j0(0.1 * np.pi * abs(m)))]

    for i, desiredACF in enumerate(targetACFs):
        ro_x = find_ro_x(d, desiredACF)  # find appropriate ro_x
        a = findFilter(ro_x)  # finding the appropriate filter to get the target ACF
        x = signal.lfilter([1.0], a, Xn).reshape(-1)  # adjust the ACF to the Gaussian sequence
        y = get_ranked_sequence(x, z)  # rank matching the sequence
        yCorr = sm.tsa.acf(y, nlags=len(desiredACF) - 1, fft=True)  # the achieved ACF of target distribution
        plt.plot(m, yCorr, 'o', mfc='none', markersize=5, label=f'achieved {names[i]} ACF')

    plt.gca().set_prop_cycle(None)
    for i, acf in enumerate(targetACFs):
        plt.plot(acf, label=f'target {names[i]} ACF')
    plt.title(f'ACF tests on {dist_name} distribution')
    plt.xlabel('lags')
    plt.ylabel('ACF')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # example:
    import numpy as np

    ACFnames = ['ryleigh', 'uniform', 'triangular', 'exponential', 'laplace']
    for i, acf in enumerate([rayleigh, uniform, triang(c=0.5, loc=-np.sqrt(6), scale=2 * np.sqrt(6)), expon, laplace]):
        DrawTestPlots(acf, ACFnames[i])
