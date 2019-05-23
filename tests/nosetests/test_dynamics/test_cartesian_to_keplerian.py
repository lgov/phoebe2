"""
"""

import phoebe
from phoebe import u
import numpy as np


def _compare_orbit(b, orbit='binary', verbose=False):
    """
    """
    def _angle_wrap(diff):
        if diff == 0.0:
            return diff
        else:
            return 2*np.pi % diff

    primary, secondary = b.hierarchy.get_children_of(orbit)

    orbit_s = b.filter(context='model', component=secondary)
    orbit_p = b.filter(context='model', component=primary)

    # get positions and velocities of secondary component wrt primary from the model
    # (these were computed using the dynamics.keplerian module)
    us = orbit_s.get_value('us', unit='solRad') - orbit_p.get_value('us', unit='solRad')
    vs = orbit_s.get_value('vs', unit='solRad') - orbit_p.get_value('vs', unit='solRad')
    ws = orbit_s.get_value('ws', unit='solRad') - orbit_p.get_value('ws', unit='solRad')

    vus = orbit_s.get_value('vus', unit='solRad/d') - orbit_p.get_value('vus', unit='solRad/d')
    vvs = orbit_s.get_value('vvs', unit='solRad/d') - orbit_p.get_value('vvs', unit='solRad/d')
    vws = orbit_s.get_value('vws', unit='solRad/d') - orbit_p.get_value('vws', unit='solRad/d')

    r = np.array([us[0], vs[0], ws[0]])
    v = np.array([vus[0], vvs[0], vws[0]])

    mass_tot = b.get_value('mass', component=primary, context='component', unit='solMass') + b.get_value('mass', component=secondary, context='component', unit='solMass')

    # convert the positions and velocites back to keplerian elements
    sma, ecc, period, incl, per0, long_an, true_anom, mean_anom, t0_perpass = phoebe.dynamics.nbody.keplerian_from_cartesian(r, v, mass_tot, t0=b.get_value('t0', context='system', unit=u.d))

    orbit_ps = b.filter(component=orbit, context='component')

    # assert that we recovered the original orbital parameters
    if verbose:
        print("sma diff {}".format(abs(sma - orbit_ps.get_value('sma', unit=u.solRad))))
        print("ecc diff {}".format(abs(ecc - orbit_ps.get_value('ecc'))))
        print("period diff {}".format(abs(period - orbit_ps.get_value('period', unit=u.d))))
        print("incl diff {}".format(abs(incl - orbit_ps.get_value('incl', unit=u.rad))))
        print("long_an diff {}".format(_angle_wrap(abs(long_an - orbit_ps.get_value('long_an', unit=u.rad)))))
        if orbit_ps.get_value('ecc') > 0:
            # if the recovered ecc wasn't exactly 0, per0 is still computed, but is meaningless
            print("per0 diff {}".format(_angle_wrap(abs(per0 - orbit_ps.get_value('per0', unit=u.rad)))))
        print("t0_perpass/period diff {}".format(abs(t0_perpass - orbit_ps.get_value('t0_perpass', unit=u.d))/orbit_ps.get_value('period', unit=u.d)))
        print("\n\n\n")

    assert(abs(sma - orbit_ps.get_value('sma', unit=u.solRad)) < 1e-6)
    assert(abs(ecc - orbit_ps.get_value('ecc')) < 1e-6)
    assert(abs(period - orbit_ps.get_value('period', unit=u.d)) < 1e-6)
    assert(abs(incl - orbit_ps.get_value('incl', unit=u.rad)) < 1e-6)
    assert(_angle_wrap(abs(long_an - orbit_ps.get_value('long_an', unit=u.rad))) < 1e-6)
    if orbit_ps.get_value('ecc') > 0:
        # if the recovered ecc wasn't exactly 0, per0 is still computed, but is meaningless
        assert(_angle_wrap(abs(per0 - orbit_ps.get_value('per0', unit=u.rad))) < 1e-6)
    # assert(abs(t0_perpass - orbit_ps.get_value('t0_perpass', unit=u.d))/orbit_ps.get_value('period', unit=u.d) < 1e-6)

def test_binary(verbose=False):
    """
    """
    b = phoebe.default_binary()

    b.add_dataset('orb', times=[0])

    # b.set_value('period', component='binary', context='component', value=3.123)

    for t0 in [0, 3.67]:
        b.set_value('t0', context='system', value=t0)
        b.set_value_all('times', context='dataset', value=[b.get_value('t0', context='system')])
        for ecc in [0.0, 0.2]:
            b.set_value('ecc', value=ecc)
            for p in [3]:
                b.set_value('period', component='binary', context='component', value=p)
                for per0 in [0, 15]:
                    b.set_value('per0', value=per0)
                    for incl in [90, 45, 0]:
                        b.set_value('incl', component='binary', value=incl)

                        if verbose:
                            print("testing with t0={} ecc={}, period={}, per0={}, incl={}".format(t0, ecc, p, per0, incl))

                        # compute POS positions and velocities from keplerian orbit
                        b.run_compute()

                        _compare_orbit(b, 'binary', verbose=verbose)


if __name__ == '__main__':
    # logger = phoebe.logger(clevel='info')

    test_binary(verbose=True)
