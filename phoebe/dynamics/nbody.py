"""
"""

import numpy as np
from scipy.optimize import newton


from phoebe import u, c
from phoebe.dynamics import keplerian
from phoebe import conf

from distutils.version import LooseVersion

try:
    import photodynam
except ImportError:
    _can_bs = False
else:
    _can_bs = True

try:
    import rebound
except ImportError:
    _can_rebound = False
else:
    _can_rebound = LooseVersion(rebound.__version__) >= LooseVersion('3.4.0')

if _can_rebound:
    try:
        import reboundx
    except ImportError:
        _can_reboundx = False
    else:
        _can_reboundx = True
else:
    _can_reboundx = False

import logging
logger = logging.getLogger("DYNAMICS.NBODY")
logger.addHandler(logging.NullHandler())


au_to_solrad = (1*u.AU).to(u.solRad).value

def _ensure_tuple(item):
    """
    Simply ensure that the passed item is a tuple.  If it is not, then
    convert it if possible, or raise a NotImplementedError

    Args:
        item: the item that needs to become a tuple

    Returns:
        the item casted as a tuple

    Raises:
        NotImplementedError: if converting the given item to a tuple
            is not implemented.
    """
    if isinstance(item, tuple):
        return item
    elif isinstance(item, list):
        return tuple(item)
    elif isinstance(item, np.ndarray):
        return tuple(item.tolist())
    else:
        raise NotImplementedError

def dynamics_from_bundle(b, times, compute=None, return_roche_euler=False, use_kepcart=False, **kwargs):
    """
    Parse parameters in the bundle and call :func:`dynamics`.

    See :func:`dynamics` for more detailed information.

    NOTE: you must either provide compute (the label) OR all relevant options
    as kwargs (ltte, stepsize, gr, integrator)

    Args:
        b: (Bundle) the bundle with a set hierarchy
        times: (list or array) times at which to run the dynamics
        stepsize: (float, optional) stepsize for the integration
            [default: 0.01]
        orbiterror: (float, optional) orbiterror for the integration
            [default: 1e-16]
        ltte: (bool, default False) whether to account for light travel time effects.
        gr: (bool, default False) whether to account for general relativity effects.

    Returns:
        t, xs, ys, zs, vxs, vys, vzs.  t is a numpy array of all times,
        the remaining are a list of numpy arrays (a numpy array per
        star - in order given by b.hierarchy.get_stars()) for the cartesian
        positions and velocities of each star at those same times.

    """

    # TODO: need to make a non-bundle version of this

    b.run_delayed_constraints()

    hier = b.hierarchy

    computeps = b.get_compute(compute, check_visible=False, force_ps=True)
    stepsize = computeps.get_value('stepsize', check_visible=False, **kwargs)
    ltte = computeps.get_value('ltte', check_visible=False, **kwargs)
    gr = computeps.get_value('gr', check_visible=False, **kwargs)
    integrator = computeps.get_value('integrator', check_visible=False, **kwargs)

    starrefs = hier.get_stars()
    orbitrefs = hier.get_orbits() if use_kepcart else [hier.get_parent_of(star) for star in starrefs]

    # vgamma = b.get_value('vgamma', context='system', unit=u.AU/u.d) # should be extracted by keplerian.dynamics_from_bundle
    t0 = b.get_value('t0', context='system', unit=u.d)

    if not _can_rebound and not conf.devel:
        # NOTE: the conf.devel only needs to be in place until rebound releases v 3.3.2
        raise ImportError("rebound 3.3.2+ is not installed")

    if gr and not _can_reboundx:
        raise ImportError("reboundx is not installed (required for gr effects)")

    dump, dump, xs, ys, zs, vxs, vys, vzs = keplerian.dynamics_from_bundle(b, [t0], compute=compute)

    def mean_anom(t0, t0_perpass, period):
        # TODO: somehow make this into a constraint where t0 and mean anom
        # are both in the compute options if dynamic_method==nbody
        # (one is constrained from the other and the orbit.... nvm, this gets ugly)
        return 2 * np.pi * (t0 - t0_perpass) / period

    def particle_ltte(sim, particle_N, t_obs):
        c_AU_d = c.c.to(u.AU/u.d).value

        def residual(t):
            # print "*** ltte trying t:", t
            if sim.t != t:
                sim.integrate(t, exact_finish_time=True)
            ltte_dt = sim.particles[particle_N].z / c_AU_d
            t_barycenter = t - ltte_dt
            # print "***** ", t_barycenter-t_obs

            return t_barycenter - t_obs

        t_barycenter = newton(residual, t_obs)

        if sim.t != t_barycenter:
            sim.integrate(t_barycenter)

        return sim.particles[particle_N]

    def com_phantom_particle(particles):
        m = sum([p.m for p in particles])
        x = 1./m * sum([p.m*p.x for p in particles])
        y = 1./m * sum([p.m*p.y for p in particles])
        z = 1./m * sum([p.m*p.z for p in particles])
        vx = 1./m * sum([p.m*p.vx for p in particles])
        vy = 1./m * sum([p.m*p.vy for p in particles])
        vz = 1./m * sum([p.m*p.vz for p in particles])

        return rebound.Particle(m=m, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

    def calculate_euler(sim, j):
        # NOTE: THIS WILL FAIL FOR j==0 FOR REBOUND < 3.3.2
        particle = sim.particles[j]

        sibling_particles = [sim.particles[k] for k in sibling_Ns[j]]
        # NOTE: this isn't the com of THIS system, but rather the COM
        # of the sibling (if it consists of more than 1 star)
        com_particle = com_phantom_particle(sibling_particles)
        # com_particle = com_phantom_particle([particle]+sibling_particles)

        # get the orbit based on this com_particle as the primary component
        orbit = particle.calculate_orbit(primary=com_particle)

        # print "*** m1 m2: {} {}".format(particle.m, com_particle.m)
        # print "*** (x1, y1, z1) (x2, y2, z2): ({}, {}, {}) ({}, {}, {})".format(particle.x, particle.y, particle.z, com_particle.x, com_particle.y, com_particle.z)
        # print "*** (vx1, vy1, vz1) (vx2, vy2, vz2): ({}, {}, {}) ({}, {}, {})".format(particle.vx, particle.vy, particle.vz, com_particle.vx, com_particle.vy, com_particle.vz)
        # print "**** P, a, d, d/a, F, inc", orbit.P, orbit.a, orbit.d, orbit.d/orbit.a, orbit.P/rotperiods[j], orbit.inc

        # for instantaneous separation, we need the current separation
        # from the sibling component in units of its instantaneous (?) sma
        d = orbit.d / orbit.a
        # for syncpar (F), assume that the rotational FREQUENCY will
        # remain fixed - so we simply need to updated syncpar based
        # on the INSTANTANEOUS orbital PERIOD.
        F = orbit.P / rotperiods[j]

        # TODO: need to add np.pi for secondary component?
        etheta = orbit.f + orbit.omega + np.pi # true anomaly + periastron

        elongan = orbit.Omega - np.pi # TODO: need to check if this is correct, this set to orbit.Omega seemed to be causing the offset issue

        eincl = orbit.inc

        # print "*** calculate_euler ind: {}, etheta: {} [{}] ({}, {}), sibling_inds: {}, self: ({}, {}, {}), sibling: ({}, {}, {})".format(j, etheta, etheta+np.pi if etheta<np.pi else etheta-np.pi, orbit.f, orbit.omega, sibling_Ns[j], particle.x, particle.y, particle.z, com_particle.x, com_particle.y, com_particle.z)

        period = orbit.P
        sma = orbit.a
        ecc = orbit.e
        per0 = orbit.omega
        long_an = orbit.Omega - np.pi
        incl = orbit.inc
        t0_perpass = orbit.T

        # TODO: all after eincl are only required if requesting to store in the
        # mesh... but shouldn't be taking too much time to calculate and store
        return d, F, etheta, elongan, eincl, period, sma, ecc, per0, long_an, incl, t0_perpass

    times = np.asarray(times) - t0

    sim = rebound.Simulation()

    if gr:
        logger.info("enabling 'gr_full' in reboundx")
        rebx = reboundx.Extras(sim)
        # TODO: switch between different GR setups based on masses/hierarchy
        # http://reboundx.readthedocs.io/en/latest/effects.html#general-relativity
        params = rebx.add_gr_full()

    sim.integrator = integrator
    # NOTE: according to rebound docs: "stepsize will change for adaptive
    # integrators such as IAS15"
    sim.dt = stepsize

    sibling_Ns = []
    for i,starref in enumerate(starrefs):
        # print "***", i, starref

        # TODO: do this in rsol instead of AU so we don't have to convert the particles back and forth below?
        mass = b.get_value('mass', u.solMass, component=starref, context='component') * c.G.to('AU3 / (Msun d2)').value

        if return_roche_euler:
            # rotperiods are only needed to compute instantaneous syncpars
            rotperiods = [b.get_value('period', u.d, component=component, context='component') for component in starrefs]
        else:
            rotperiod = None


        # xs, ys, zs are in solRad; vxs, vys, vzs are in solRad/d
        # pass to rebound in AU and AU/d
        # print "***", starref, mass, xs[i][0]/au_to_solrad, ys[i][0]/au_to_solrad, zs[i][0]/au_to_solrad, vxs[i][0]/au_to_solrad, vys[i][0]/au_to_solrad, vzs[i][0]/au_to_solrad
        sim.add(m=mass, x=xs[i][0]/au_to_solrad, y=ys[i][0]/au_to_solrad, z=zs[i][0]/au_to_solrad, vx=vxs[i][0]/au_to_solrad, vy=vys[i][0]/au_to_solrad, vz=vzs[i][0]/au_to_solrad)

        # also track the index of all particles that need to be included as
        # the sibling of this particle (via their COM)
        # TODO: only do this if return_roche_euler?
        sibling_starrefs = hier.get_stars_of_sibling_of(starref)
        sibling_Ns.append([starrefs.index(s) for s in sibling_starrefs])


    #### TESTING ###
    # print "*** TESTING EULER BEFORE INTEGRATION"
    # for j,starref in enumerate(starrefs):
    #     print "*** {} original: (x, y, z) (vx, vy, vz): ({}, {}, {}) ({}, {}, {})".format(starref, xs[j][0]/au_to_solrad, ys[j][0]/au_to_solrad, zs[j][0]/au_to_solrad, vxs[j][0]/au_to_solrad, vys[j][0]/au_to_solrad, vzs[j][0]/au_to_solrad)
    #     calculate_euler(sim, j)
    # exit()

    ################



    xs = [np.zeros(times.shape) for j in range(sim.N)]
    ys = [np.zeros(times.shape) for j in range(sim.N)]
    zs = [np.zeros(times.shape) for j in range(sim.N)]
    vxs = [np.zeros(times.shape) for j in range(sim.N)]
    vys = [np.zeros(times.shape) for j in range(sim.N)]
    vzs = [np.zeros(times.shape) for j in range(sim.N)]

    if return_roche_euler:
        # from instantaneous Keplerian dynamics for Roche meshing
        ds = [np.zeros(times.shape) for j in range(sim.N)]
        Fs = [np.zeros(times.shape) for j in range(sim.N)]

        ethetas = [np.zeros(times.shape) for j in range(sim.N)]
        elongans = [np.zeros(times.shape) for j in range(sim.N)]
        eincls = [np.zeros(times.shape) for j in range(sim.N)]

        periods = [np.zeros(times.shape) for j in range(sim.N)]
        smas = [np.zeros(times.shape) for j in range(sim.N)]
        eccs = [np.zeros(times.shape) for j in range(sim.N)]
        per0s = [np.zeros(times.shape) for j in range(sim.N)]
        long_ans = [np.zeros(times.shape) for j in range(sim.N)]
        incls = [np.zeros(times.shape) for j in range(sim.N)]
        t0_perpasses = [np.zeros(times.shape) for j in range(sim.N)]

    for i,time in enumerate(times):

        # print "*** integrating to t=", time

        sim.integrate(time, exact_finish_time=True)

        # if return_roche:
            # TODO: do we need to do this after handling LTTE???
            # orbits = sim.calculate_orbits()

        for j in range(sim.N):

            # get roche/euler information BEFORE making LTTE adjustments
            if return_roche_euler:


                d, F, etheta, elongan, eincl, period, sma, ecc, per0, long_an, incl, t0_perpass = calculate_euler(sim, j)

                ds[j][i] = d
                Fs[j][i] = F
                ethetas[j][i] = etheta
                elongans[j][i] = elongan
                eincls[j][i] = eincl

                periods[j][i] = period
                smas[j][i] = sma
                eccs[j][i] = ecc
                per0s[j][i] = per0
                long_ans[j][i] = long_an
                incls[j][i] = incl
                t0_perpasses[j][i] = t0_perpass

            # if necessary, make adjustments to particle for LTTE and then
            # store position/velocity of the particle
            if ltte:
                # then we need to integrate to different times per object
                particle = particle_ltte(sim, j, time)
            else:
                # print "***", time, j, sim.N
                particle = sim.particles[j]

            xs[j][i] = particle.x * au_to_solrad # solRad
            ys[j][i] = particle.y * au_to_solrad # solRad
            zs[j][i] = particle.z * au_to_solrad  # solRad
            vxs[j][i] = particle.vx * au_to_solrad # solRad/d
            vys[j][i] = particle.vy * au_to_solrad # solRad/d
            vzs[j][i] = particle.vz * au_to_solrad # solRad/d


    if return_roche_euler:
        # d, solRad, solRad/d, rad, unitless (sma), unitless, rad, rad, rad
        return times, xs, ys, zs, vxs, vys, vzs, ds, Fs, ethetas, elongans, eincls, periods, smas, eccs, per0s, long_ans, incls, t0_perpasses

    else:
        # d, solRad, solRad/d, rad
        return times, xs, ys, zs, vxs, vys, vzs


def dynamics_from_bundle_bs(b, times, compute=None, return_roche_euler=False, **kwargs):
    """
    Parse parameters in the bundle and call :func:`dynamics`.

    See :func:`dynamics` for more detailed information.

    NOTE: you must either provide compute (the label) OR all relevant options
    as kwargs (ltte)

    Args:
        b: (Bundle) the bundle with a set hierarchy
        times: (list or array) times at which to run the dynamics
        stepsize: (float, optional) stepsize for the integration
            [default: 0.01]
        orbiterror: (float, optional) orbiterror for the integration
            [default: 1e-16]
        ltte: (bool, default False) whether to account for light travel time effects.

    Returns:
        t, xs, ys, zs, vxs, vys, vzs.  t is a numpy array of all times,
        the remaining are a list of numpy arrays (a numpy array per
        star - in order given by b.hierarchy.get_stars()) for the cartesian
        positions and velocities of each star at those same times.

    """

    stepsize = 0.01
    orbiterror = 1e-16
    computeps = b.get_compute(compute, check_visible=False, force_ps=True)
    ltte = computeps.get_value('ltte', check_visible=False, **kwargs)


    hier = b.hierarchy

    starrefs = hier.get_stars()
    orbitrefs = hier.get_orbits()

    def mean_anom(t0, t0_perpass, period):
        # TODO: somehow make this into a constraint where t0 and mean anom
        # are both in the compute options if dynamic_method==nbody
        # (one is constrained from the other and the orbit.... nvm, this gets ugly)
        return 2 * np.pi * (t0 - t0_perpass) / period

    masses = [b.get_value('mass', u.solMass, component=component, context='component') * c.G.to('AU3 / (Msun d2)').value for component in starrefs]  # GM
    smas = [b.get_value('sma', u.AU, component=component, context='component') for component in orbitrefs]
    eccs = [b.get_value('ecc', component=component, context='component') for component in orbitrefs]
    incls = [b.get_value('incl', u.rad, component=component, context='component') for component in orbitrefs]
    per0s = [b.get_value('per0', u.rad, component=component, context='component') for component in orbitrefs]
    long_ans = [b.get_value('long_an', u.rad, component=component, context='component') for component in orbitrefs]
    t0_perpasses = [b.get_value('t0_perpass', u.d, component=component, context='component') for component in orbitrefs]
    periods = [b.get_value('period', u.d, component=component, context='component') for component in orbitrefs]

    vgamma = b.get_value('vgamma', context='system', unit=u.solRad/u.d)
    t0 = b.get_value('t0', context='system', unit=u.d)

    # mean_anoms = [mean_anom(t0, t0_perpass, period) for t0_perpass, period in zip(t0_perpasses, periods)]
    mean_anoms = [b.get_value('mean_anom', u.rad, component=component, context='component') for component in orbitrefs]

    return dynamics_bs(times, masses, smas, eccs, incls, per0s, long_ans, \
                    mean_anoms, t0, vgamma, stepsize, orbiterror, ltte,
                    return_roche_euler=return_roche_euler)




def dynamics_bs(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms,
        t0=0.0, vgamma=0.0, stepsize=0.01, orbiterror=1e-16, ltte=False,
        return_roche_euler=False):
    """
    Burlisch-Stoer integration of orbits to give positions and velocities
    of any given number of stars in hierarchical orbits.  This code
    currently uses the NBody code in Josh Carter's photodynam code
    available here:

    [[TODO: include link]]

    If using the Nbody mode in PHOEBE, please cite him as well:

    [[TODO: include citation]]

    See :func:`dynamics_from_bundle` for a wrapper around this function
    which automatically handles passing everything in the correct order
    and in the correct units.

    For each iterable input, stars and orbits should be listed in order
    from primary -> secondary for each nested hierarchy level.  Each
    iterable for orbits should have length one less than those for each
    star (ie if 3 masses are provided, then 2 smas, eccs, etc need to
    be provided)

    Args:
        times: (iterable) times at which to compute positions and
            velocities for each star
        masses: (iterable) mass for each star in [solMass]
        smas: (iterable) semi-major axis for each orbit [AU]
        eccs: (iterable) eccentricities for each orbit
        incls: (iterable) inclinations for each orbit [rad]
        per0s: (iterable) longitudes of periastron for each orbit [rad]
        long_ans: (iterable) longitudes of the ascending node for each
            orbit [rad]
        mean_anoms: (iterable) mean anomalies for each orbit
        t0: (float) time at which to start the integrations
        stepsize: (float, optional) stepsize of the integrations
            [default: 0.01]
        orbiterror: (float, optional) orbiterror of the integrations
            [default: 1e-16]
        ltte: (bool, default False) whether to account for light travel time effects.

    Returns:
        t, xs, ys, zs, vxs, vys, vzs.  t is a numpy array of all times,
        the remaining are a list of numpy arrays (a numpy array per
        star - in order given by b.hierarchy.get_stars()) for the cartesian
        positions and velocities of each star at those same times.

    """

    if not _can_bs:
        raise ImportError("photodynam is not installed (http://github.com/phoebe-project/photodynam)")

    times = _ensure_tuple(times)
    masses = _ensure_tuple(masses)
    smas = _ensure_tuple(smas)
    eccs = _ensure_tuple(eccs)
    incls = _ensure_tuple(incls)
    per0s = _ensure_tuple(per0s)
    long_ans = _ensure_tuple(long_ans)
    mean_anoms = _ensure_tuple(mean_anoms)

    # TODO: include vgamma!!!!
    # print "*** bs.do_dynamics", masses, smas, eccs, incls, per0s, long_ans, mean_anoms, t0
    d = photodynam.do_dynamics(times, masses, smas, eccs, incls, per0s, long_ans, mean_anoms, t0, stepsize, orbiterror, ltte, return_roche_euler)
    # d is in the format: {'t': (...), 'x': ( (1,2,3), (1,2,3), ...), 'y': ..., 'z': ...}

    nobjects = len(masses)
    ntimes = len(times)

    # TODO: need to return euler angles... if that even makes sense?? Or maybe we
    # need to make a new place in orbit??

    au_to_solrad = (1*u.AU).to(u.solRad).value

    ts = np.array(d['t'])
    xs = [(-1*np.array([d['x'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
    ys = [(-1*np.array([d['y'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
    zs = [(np.array([d['z'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
    vxs = [(-1*np.array([d['vx'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
    vys = [(-1*np.array([d['vy'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
    vzs = [(np.array([d['vz'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]

    if return_roche_euler:
        # raise NotImplementedError("euler angles for BS not currently supported")
        # a (sma), e (ecc), in (incl), o (per0?), ln (long_an?), m (mean_anom?)
        ds = [(np.array([d['kepl_a'][ti][oi] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        # TODO: fix this
        Fs = [(np.array([1.0 for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        # TODO: check to make sure this is the right angle
        # TODO: need to add np.pi for secondary component?
        # true anomaly + periastron
        ethetas = [(np.array([d['kepl_o'][ti][oi]+d['kepl_m'][ti][oi]+np.pi/2 for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        # elongans = [(np.array([d['kepl_ln'][ti][oi]+long_ans[0 if oi==0 else oi-1] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        elongans = [(np.array([d['kepl_ln'][ti][oi]+long_ans[0 if oi==0 else oi-1] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        # eincls = [(np.array([d['kepl_in'][ti][oi]+incls[0 if oi==0 else oi-1] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]
        eincls = [(np.array([d['kepl_in'][ti][oi]+np.pi-incls[0 if oi==0 else oi-1] for ti in range(ntimes)])*au_to_solrad) for oi in range(nobjects)]



        # d, solRad, solRad/d, rad
        return ts, xs, ys, zs, vxs, vys, vzs, ds, Fs, ethetas, elongans, eincls

    else:

        # d, solRad, solRad/d
        return ts, xs, ys, zs, vxs, vys, vzs
