import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import rayleigh, triang, laplace, uniform, expon
from scipy.special import j0

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import generate_corr_sequence.generate_corr_sequence as gcs

#change to pytest in the future?

def test_d():
    rtol = 1e-4  # relative tolerance

    dRayleighCode = np.round(gcs.findCoeff(dist_obj=rayleigh)[:7], 7)
    dRayleighArticle = np.array([0.9724690, 0.0264400, 0.0010467, 0.0000113, 0.0000310, 0.0000017, 0.0000007])
    try:
        np.testing.assert_allclose(dRayleighArticle, dRayleighCode, rtol=rtol, atol=0)
        print("Rayleigh coeffs successfuly match")
    except:
        print("Rayleigh coeffs don't match")
        print(f"\tTarget coeffs:\n{dRayleighArticle}")
        print(f"\tCalculated coeffs:\n{dRayleighCode}")

    dLaplaceCode = np.round(gcs.findCoeff(dist_obj=laplace)[:7], 7)
    dLaplaceArticle = np.array([0.9632098, 0, 0.0352063, 0, 0.0013255, 0, 0.0002592])
    try:
        np.testing.assert_allclose(dLaplaceArticle, dLaplaceCode, rtol=rtol, atol=0)
        print("Laplce coeffs successfuly match")
    except:
        print("Laplace coeffs don't match")
        print(f"Target coeffs:\n{dLaplaceArticle}")
        print(f"Calculated coeffs:\n{dLaplaceCode}")

    dUniformCode = np.round(gcs.findCoeff(dist_obj=uniform)[:7], 7)
    dUniformArticle = np.array([0.9550627, 0, 0.0397943, 0, 0.0044768, 0, 0.0006662])
    try:
        np.testing.assert_allclose(dUniformArticle, dUniformCode, rtol=rtol, atol=0)
        print("Uniform coeffs successfuly match")
    except:
        print("Uniform coeffs don't match")
        print(f"Target coeffs:\n{dUniformArticle}")
        print(f"Calculated coeffs:\n{dUniformCode}")

    dExponCode = np.round(gcs.findCoeff(dist_obj=expon)[:7], 7)
    dExponArticle = np.array([0.8157660, 0.1773910, 0.0066847, 0.0001343, 0.0000169, 0.0000073, 0.0])
    try:
        np.testing.assert_allclose(dExponArticle, dExponCode, rtol=rtol, atol=0,
                                   err_msg="Exponential coeffs don't match")
        print("Exponential coeffs successfuly match")
    except:
        print("Exponential coeffs don't match")
        print(f"Target coeffs:\n{dExponArticle}")
        print(f"Calculated coeffs:\n{dExponCode}")

    dTriangleCode = np.round(gcs.findCoeff(dist_obj=triang(c=0.5, loc=-np.sqrt(6), scale=2 * np.sqrt(6)))[:7], 7)
    dTriangleArticle = np.array([0.9927701, 0.0, 0.0071243, 0.0, 0.0000010, 0.0, 0.0000100])
    try:
        np.testing.assert_allclose(dTriangleArticle, dTriangleCode, rtol=rtol, atol=0,
                                   err_msg="Triangle coeffs don't match")
        print("Triangle coeffs successfuly match")
    except:
        print("Triangle coeffs don't match")
        print(f"\tTarget coeffs:\n\t{dTriangleArticle}")
        print(f"\tCalculated coeffs:\n\t{dTriangleCode}")


def test_ACF(dist_obj=rayleigh):
    rtol = 0.1  # relative tolerance

    d = list(np.round(gcs.findCoeff(dist_obj=dist_obj), 7))
    m = np.arange(0, 100)

    LinearTargetACF = 1 - np.minimum(m, 100) / 100
    LinearCalcACF = np.array(gcs.find_ro_x(d, LinearTargetACF))
    try:
        np.testing.assert_allclose(LinearTargetACF, LinearCalcACF, rtol=rtol, atol=0)
        print("Linear target and calculated ACFs match successfuly")
    except:
        print("Linear target and calculated ACFs don't match")
        plt.plot(m, LinearTargetACF, label='target ACF')
        plt.plot(m, LinearCalcACF, label='calculated ACF with d coeffs')
        plt.title('ACF target vs calculated')
        plt.xlabel('lags')
        plt.ylabel('ACF')
        plt.legend()
        plt.show()

    ExponTargetACF = np.exp(-0.05 * np.abs(m))
    ExponCalcACF = np.array(gcs.find_ro_x(d, ExponTargetACF))
    try:
        np.testing.assert_allclose(ExponTargetACF, ExponCalcACF, rtol=rtol, atol=0)
        print("exponential target and calculated ACFs match successfuly")
    except:
        print("exponential target and calculated ACFs don't match")
        plt.plot(m, ExponTargetACF, label='target ACF')
        plt.plot(m, ExponCalcACF, label='calculated ACF with d coeffs')
        plt.title('ACF target vs calculated')
        plt.xlabel('lags')
        plt.ylabel('ACF')
        plt.legend()
        plt.show()

    ExponCosTargetACF = np.exp(-0.05 * np.abs(m)) * np.cos(0.25 * np.abs(m))
    ExponCosCalcACF = np.array(gcs.find_ro_x(d, ExponCosTargetACF))
    try:
        np.testing.assert_allclose(ExponCosTargetACF, ExponCosCalcACF, rtol=rtol, atol=0)
        print("exponent*cos target and calculated ACFs match successfuly")
    except:
        print("exponent*cos target and calculated ACFs don't match")
        plt.plot(m, ExponCosTargetACF, label='target ACF')
        plt.plot(m, ExponCosCalcACF, label='calculated ACF with d coeffs')
        plt.title('ACF target vs calculated')
        plt.xlabel('lags')
        plt.ylabel('ACF')
        plt.legend()
        plt.show()

    BesselTargetACF = np.array(j0(0.1 * np.pi * abs(m)))
    BesselCalcACF = np.array(gcs.find_ro_x(d, BesselTargetACF))
    try:
        np.testing.assert_allclose(BesselTargetACF, BesselCalcACF, rtol=rtol, atol=0)
        print("Bessel target and calculated ACFs match successfuly")
    except:
        print("Bessel target and calculated ACFs don't match")
        plt.plot(m, BesselTargetACF, label='target ACF')
        plt.plot(m, BesselCalcACF, label='calculated ACF with d coeffs')
        plt.title('ACF target vs calculated')
        plt.xlabel('lags')
        plt.ylabel('ACF')
        plt.legend()
        plt.show()


def DrawTestPlots(dist_obj=rayleigh, dist_name='rayleigh'):
    L = 2 ** 20
    m = np.arange(0, 100)
    d = gcs.findCoeff(dist_obj)
    Xn = np.random.randn(1, L)  # normal sequence
    z = dist_obj.rvs(size=L)  # Desired distribution sequence
    names = ['linear', 'exp', 'exp*cos', 'bessel']
    targetACFs = [1 - np.minimum(m, 100) / 100, np.exp(-0.05 * np.abs(m)),
                  np.exp(-0.05 * np.abs(m)) * np.cos(0.25 * np.abs(m)), np.array(j0(0.1 * np.pi * abs(m)))]

    for i, desiredACF in enumerate(targetACFs):
        ro_x = gcs.find_ro_x(d, desiredACF)  # find appropriate ro_x
        a = gcs.findFilter(ro_x)  # finding the appropriate filter to get the target ACF
        x = signal.lfilter([1.0], a, Xn).reshape(-1)  # adjust the ACF to the Gaussian sequence
        y = gcs.get_ranked_sequence(x, z)  # rank matching the sequence
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
    # example of d coeffs test:
    test_d()

    #example of ACF target vs calculated test
    test_ACF(rayleigh) # test that passes
    test_ACF(expon) # test that fails

    # example of generation of autoccorelated sequence test:
    ACFnames = ['ryleigh', 'uniform', 'triangular', 'exponential', 'laplace']
    for i, acf in enumerate([rayleigh, uniform, triang(c=0.5, loc=-np.sqrt(6), scale=2 * np.sqrt(6)), expon, laplace]):
        DrawTestPlots(acf, ACFnames[i])

