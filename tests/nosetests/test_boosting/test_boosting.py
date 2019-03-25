"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt

def test_blackbody_v_ck2004(verbose=False):
    b = phoebe.Bundle.default_binary()

    # compute mesh at quarter phase to maximize z velocities
    b.add_dataset('lc', times=[0.25])
    b.add_dataset('mesh', times=[0.25], columns=['boost_factors@*'])

    # fails with logarithmic [0.5, 0.5] and linear [0]
    b.set_value_all('ld_func', 'linear')
    b.set_value_all('ld_coeffs', [0.0])

    b.set_value('boosting_method', value='linear')

    # vary over velocities and temperatures
    for q in [0.5, 1]:
        b.set_value('q', value=q)

        for teff in [5000, 10000]:
            b.set_value('teff', component='primary', value=teff)

            b.set_value_all('atm', value='blackbody')
            b.run_compute(irrad_method='none')
            bb_factors = b.get_value('boost_factors', component='primary', time=0.25)

            b.set_value_all('atm', value='ck2004')
            b.run_compute(irrad_method='none')
            ck2004_factors = b.get_value('boost_factors', component='primary', time=0.25)

            if verbose:
                print "q={}, teff={} max abs difference={} max rel difference={}".format(q, teff, abs(bb_factors-ck2004_factors).max(), max(abs(bb_factors-ck2004_factors)/ck2004_factors))

        assert(np.allclose(bb_factors, ck2004_factors, rtol=1e-5, atol=0.))

    return b

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')

    b = test_blackbody_v_ck2004(verbose=True)
