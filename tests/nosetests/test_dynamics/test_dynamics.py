"""
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


def _keplerian_v_nbody(b, verbose=False, plot=False):
    """
    test a single bundle for the phoebe backend's kepler vs nbody dynamics methods
    """

    # TODO: loop over ltte=True,False (once keplerian dynamics supports the switch)

    # b.add_compute(dynamics_method='bs')
    # b.set_value('dynamics_method', 'bs')

    times = np.linspace(0, 100, 10000)
    rb_ts, rb_us, rb_vs, rb_ws, rb_vus, rb_vvs, rb_vws = phoebe.dynamics.nbody.dynamics_from_bundle_rebound(b, times, ltte=False)
    bs_ts, bs_us, bs_vs, bs_ws, bs_vus, bs_vvs, bs_vws = phoebe.dynamics.nbody.dynamics_from_bundle_bs(b, times, ltte=False)
    k_ts, k_correctedts, k_us, k_vs, k_ws, k_vus, k_vvs, k_vws = phoebe.dynamics.keplerian.dynamics_from_bundle(b, times, ltte=False)

    assert(np.allclose(rb_ts, k_ts, 1e-12))
    assert(np.allclose(bs_ts, k_ts, 1e-12))

    for ci in range(len(b.hierarchy.get_stars())):
        # TODO: make atol lower (currently 1e-5 solRad which is awfully big, but 1e-6 currently fails!)
        if verbose:
            print("rb max atol ts:", np.max(abs(rb_ts - k_ts)))
            print("rb max atol us:", np.max(abs(rb_us[ci] - k_us[ci])))
            print("rb max atol vs:", np.max(abs(rb_vs[ci] - k_vs[ci])))
            print("rb max atol ws:", np.max(abs(rb_ws[ci] - k_ws[ci])))
            print("rb max atol vus:", np.max(abs(rb_vus[ci] - k_vus[ci])))
            print("rb max atol vvs:", np.max(abs(rb_vvs[ci] - k_vvs[ci])))
            print("rb max atol vws:", np.max(abs(rb_vws[ci] - k_vws[ci])))

            print("bs max atol ts:", np.max(abs(bs_ts - k_ts)))
            print("bs max atol us:", np.max(abs(bs_us[ci] - k_us[ci])))
            print("bs max atol vs:", np.max(abs(bs_vs[ci] - k_vs[ci])))
            print("bs max atol ws:", np.max(abs(bs_ws[ci] - k_ws[ci])))
            print("bs max atol vus:", np.max(abs(bs_vus[ci] - k_vus[ci])))
            print("bs max atol vvs:", np.max(abs(bs_vvs[ci] - k_vvs[ci])))
            print("bs max atol vws:", np.max(abs(bs_vws[ci] - k_vws[ci])))

        if plot:
            plt.plot(rb_ts, rb_us[ci]-k_us[ci], 'k-', label='u resids')
            plt.plot(rb_ts, rb_vs[ci]-k_vs[ci], 'b-', label='v resids')
            plt.plot(rb_ts, rb_ws[ci]-k_ws[ci], 'r-', label='w resids')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('residuals')
            plt.show()

            plt.plot(rb_ts, bs_us[ci]-k_us[ci], 'k-', label='u resids')
            plt.plot(rb_ts, bs_vs[ci]-k_vs[ci], 'b-', label='v resids')
            plt.plot(rb_ts, bs_ws[ci]-k_ws[ci], 'r-', label='w resids')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('residuals')
            plt.show()

        assert(np.allclose(rb_us[ci], k_us[ci], atol=4e-5))
        assert(np.allclose(rb_vs[ci], k_vs[ci], atol=1e-12))
        assert(np.allclose(rb_ws[ci], k_ws[ci], atol=4e-5))
        assert(np.allclose(rb_vus[ci], k_vus[ci], atol=3e-4))
        assert(np.allclose(rb_vvs[ci], k_vvs[ci], atol=1e-12))
        assert(np.allclose(rb_vws[ci], k_vws[ci], atol=3e-4))

        assert(np.allclose(bs_us[ci], k_us[ci], atol=6e-6))
        assert(np.allclose(bs_vs[ci], k_vs[ci], atol=1e-12))
        assert(np.allclose(bs_ws[ci], k_ws[ci], atol=6e-6))
        assert(np.allclose(bs_vus[ci], k_vus[ci], atol=4e-5))
        assert(np.allclose(bs_vvs[ci], k_vvs[ci], atol=1e-12))
        assert(np.allclose(bs_vws[ci], k_vws[ci], atol=4e-5))

def _phoebe_v_photodynam(b, verbose=False, plot=False):
    """
    test a single bundle for phoebe's nbody vs photodynam via the frontend
    """

    times = np.linspace(0, 100, 1000)

    b.add_dataset('orb', times=times, dataset='orb01', component=b.hierarchy.get_stars())
    # photodynam and phoebe should have the same nbody defaults... if for some reason that changes,
    # then this will probably fail
    b.add_compute('photodynam', compute='pdcompute')
    # photodynam backend ONLY works with ltte=True, so we will run the phoebe backend with that as well
    # TODO: remove distortion_method='nbody' once that is supported
    # NOTE: bs is the exact same as that used in photodynam.  Nbody and rebound are slightly different.
    b.add_compute('phoebe', dynamics_method='bs', ltte=True, compute='phoebecompute')

    b.run_compute('pdcompute', model='pdresults')
    b.run_compute('phoebecompute', model='phoeberesults')

    for comp in b.hierarchy.get_stars():
        # TODO: check to see how low we can make atol (or change to rtol?)
        # TODO: look into justification of flipping x and y for both dynamics (photodynam & phoebe)
        # TODO: why the small discrepancy (visible especially in y, still <1e-11) - possibly a difference in time0 or just a precision limit in the photodynam backend since loading from a file??


        if plot or verbose:
            for k in ['us', 'vs', 'ws', 'vus', 'vvs', 'vws']:
                diff = abs(b.get_value(k, model='phoeberesults', component=comp) - b.get_value(k, model='pdresults', component=comp))

                if verbose:
                    print("*** max abs ({}): {}".format(k, max(diff)))

                if plot:
                    plt.cla()
                    plt.plot(b.get_value('times', model='phoeberesults', component=comp, unit=u.d), b.get_value(k, model='phoeberesults', component=comp), 'r-')
                    plt.plot(b.get_value('times', model='phoeberesults', component=comp, unit=u.d), b.get_value(k, model='pdresults', component=comp), 'b-')

                    plt.xlabel('t')
                    plt.ylabel(k)
                    plt.show()

        assert(np.allclose(b.get_value('times', dataset='orb01', model='phoeberesults', component=comp, unit=u.d), b.get_value('times', dataset='orb01', model='pdresults', component=comp, unit=u.d), atol=1e-6))
        assert(np.allclose(b.get_value('us', dataset='orb01', model='phoeberesults', component=comp, unit=u.AU), b.get_value('us', dataset='orb01', model='pdresults', component=comp, unit=u.AU), atol=1e-6))
        assert(np.allclose(b.get_value('vs', dataset='orb01', model='phoeberesults', component=comp, unit=u.AU), b.get_value('vs', dataset='orb01', model='pdresults', component=comp, unit=u.AU), atol=1e-6))
        assert(np.allclose(b.get_value('ws', dataset='orb01', model='phoeberesults', component=comp, unit=u.AU), b.get_value('ws', dataset='orb01', model='pdresults', component=comp, unit=u.AU), atol=1e-6))
        assert(np.allclose(b.get_value('vus', dataset='orb01', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vus', dataset='orb01', model='pdresults', component=comp, unit=u.solRad/u.d), atol=1e-6))
        assert(np.allclose(b.get_value('vvs', dataset='orb01', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vvs', dataset='orb01', model='pdresults', component=comp, unit=u.solRad/u.d), atol=1e-6))
        assert(np.allclose(b.get_value('vws', dataset='orb01', model='phoeberesults', component=comp, unit=u.solRad/u.d), b.get_value('vws', dataset='orb01', model='pdresults', component=comp, unit=u.solRad/u.d), atol=1e-6))

def _frontend_v_backend(b, verbose=False, plot=False):
    """
    test a single bundle for the frontend vs backend access to both kepler and nbody dynamics
    """

    # TODO: loop over ltte=True,False

    times = np.linspace(0, 100, 21)
    b.add_dataset('orb', times=times, dataset='orb01', component=b.hierarchy.get_stars())
    b.add_compute('phoebe', dynamics_method='keplerian', ltte=False, compute='keplerian')
    b.add_compute('phoebe', dynamics_method='nbody', ltte=False, compute='nbody')


    # NBODY
    # do backend Nbody
    b_ts, b_us, b_vs, b_ws, b_vus, b_vvs, b_vws = phoebe.dynamics.nbody.dynamics_from_bundle(b, times, ltte=False, compute='nbody')

    # do frontend Nbody
    b.run_compute('nbody', model='nbodyresults')



    for ci,comp in enumerate(b.hierarchy.get_stars()):
        # TODO: can we lower tolerances?
        for qualifier, backend_array, unit, atol in zip(['times', 'us', 'vs', 'ws', 'vus', 'vvs', 'vws'], [b_ts, b_us[ci], b_vs[ci], b_ws[ci], b_vus[ci], b_vvs[ci], b_vws[ci]], [u.d, u.solRad, u.solRad, u.solRad, u.solRad/u.d, u.solRad/u.d, u.solRad/u.d], [1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20]):
            frontend_array = b.get_value(qualifier, dataset='orb01', model='nbodyresults', component=comp, unit=unit)
            if verbose:
                print("*** max abs diff ({}): {}, max abs value ({}): {}".format(qualifier, np.max(abs(frontend_array-backend_array)), qualifier, max(abs(backend_array))))
                # print(frontend_array)

            assert(np.allclose(frontend_array, backend_array, atol=atol))


    # KEPLERIAN
    # do backend keplerian
    b_ts, b_correctedts, b_us, b_vs, b_ws, b_vus, b_vvs, b_vws = phoebe.dynamics.keplerian.dynamics_from_bundle(b, times, ltte=False, compute='keplerian')

    # do frontend keplerian
    b.run_compute('keplerian', model='keplerianresults')

    for ci,comp in enumerate(b.hierarchy.get_stars()):
        # TODO: can we lower tolerances?
        for qualifier, backend_array, unit, atol in zip(['times', 'us', 'vs', 'ws', 'vus', 'vvs', 'vws'], [b_ts, b_us[ci], b_vs[ci], b_ws[ci], b_vus[ci], b_vvs[ci], b_vws[ci]], [u.d, u.solRad, u.solRad, u.solRad, u.solRad/u.d, u.solRad/u.d, u.solRad/u.d], [1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20]):
            frontend_array = b.get_value(qualifier, dataset='orb01', model='keplerianresults', component=comp, unit=unit)
            if verbose:
                print("*** max abs diff ({}): {}, max abs value ({}): {}".format(qualifier, np.max(abs(frontend_array-backend_array)), qualifier, max(abs(backend_array))))
                # print(frontend_array)

            assert(np.allclose(frontend_array, backend_array, atol=atol))


def test_binary(verbose=False, plot=False):
    """
    """
    phoebe.devel_on() # required for dynamics_method 'bs' or 'rebound' (but not 'nbody' which uses rebound by default)

    # TODO: once ps.copy is implemented, just send b.copy() to each of these

    b = phoebe.default_binary()
    _keplerian_v_nbody(b, verbose=verbose, plot=plot)

    b = phoebe.default_binary()
    _phoebe_v_photodynam(b, verbose=verbose, plot=plot)

    b = phoebe.default_binary()
    _frontend_v_backend(b, verbose=verbose, plot=plot)

    phoebe.devel_off() # reset for future tests


if __name__ == '__main__':
    # logger = phoebe.logger(clevel='debug')


    test_binary(verbose=True, plot=False)

    # TODO: create tests for both triple configurations (A--B-C, A-B--C) - these should first be default bundles
