import sys
sys.path.append("..")
from generate_corr_sequence import gen_corr_sequence
import numpy as np
from scipy.stats import nakagami
from scipy.special import j0

if __name__ == "__main__":
    m = np.arange(0, 100)
    signal = gen_corr_sequence(dist_obj=nakagami(nu=1), desiredACF=np.array(j0(0.1 * np.pi * abs(m))), debug=True)