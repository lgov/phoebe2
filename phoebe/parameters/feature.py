

from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe import u
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def _component_allowed_for_feature(feature_kind, component_kind):
    _allowed = {}
    _allowed['spot'] = ['star', 'envelope']
    _allowed['pulsation'] = ['star', 'envelope']

    return component_kind in _allowed.get(feature_kind, [None])

def _dataset_allowed_for_feature(feature_kind, dataset_kind):
    _allowed = {}
    _allowed['gaussian_process'] = ['lc', 'rv', 'lp']

    return dataset_kind in _allowed.get(feature_kind, [None])

def _solver_allowed_for_feature(feature_kind, solver_kind):
    _allowed = {}
    _allowed['emcee_move'] = ['emcee']

    return solver_kind in _allowed.get(feature_kind, [None])


def spot(feature, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a spot feature.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Allowed to attach to:
    * components with kind: star
    * datasets: not allowed
    * solver: not allowed

    Arguments
    ----------
    * `colat` (float/quantity, optional): colatitude of the center of the spot
        wrt spin axis.
    * `long` (float/quantity, optional): longitude of the center of the spot wrt
        spin axis.
    * `radius` (float/quantity, optional): angular radius of the spot.
    * `relteff` (float/quantity, optional): temperature of the spot relative
        to the intrinsic temperature.

    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params = []

    params += [FloatParameter(qualifier="colat", value=kwargs.get('colat', 0.0), default_unit=u.deg, description='Colatitude of the center of the spot wrt spin axis')]
    params += [FloatParameter(qualifier="long", value=kwargs.get('long', 0.0), default_unit=u.deg, description='Longitude of the center of the spot wrt spin axis')]
    params += [FloatParameter(qualifier='radius', value=kwargs.get('radius', 1.0), default_unit=u.deg, description='Angular radius of the spot')]
    # params += [FloatParameter(qualifier='area', value=kwargs.get('area', 1.0), default_unit=u.solRad, description='Surface area of the spot')]

    params += [FloatParameter(qualifier='relteff', value=kwargs.get('relteff', 1.0), limits=(0.,None), default_unit=u.dimensionless_unscaled, description='Temperature of the spot relative to the intrinsic temperature')]
    # params += [FloatParameter(qualifier='teff', value=kwargs.get('teff', 10000), default_unit=u.K, description='Temperature of the spot')]

    constraints = []

    return ParameterSet(params), constraints

def gaussian_process(feature, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for a gaussian_process feature.

    Requires celerite to be installed.  See https://celerite.readthedocs.io/en/stable/.
    If using gaussian processes, consider citing:
    * https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F

    See also:
    * <phoebe.frontend.bundle.Bundle.references>

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Allowed to attach to:
    * components: not allowed
    * datasets with kind: lc
    * solvers: not allowed

    If `compute_times` or `compute_phases` is used: the underlying model without
    gaussian_processes will be computed at the given times/phases but will then
    be interpolated into the times of the underlying dataset to include the
    contribution of gaussian processes and will be exposed at the dataset
    times (with a warning in the logger and in
    <phoebe.frontend.bundle.Bundle.run_checks_compute>).  If the system is
    time-dependent without GPs
    (see <phoebe.parameters.HierarchyParameter.is_time_dependent>), then
    the underlying model will need to cover the entire dataset or an error
    will be raised by <phoebe.frontend.bundle.Bundle.run_checks_compute>.


    Arguments
    ----------
    * `kernel` (string, optional, default='matern32'): Kernel for the gaussian
        process (see https://celerite.readthedocs.io/en/stable/python/kernel/)
    * `log_S0` (float, optional, default=0): only applicable if `kernel` is
        'sho'. Log of the GP parameter S0.
    * `log_Q` (float, optional, default=0): only applicable if `kernel` is
        'sho'.  Log of the GP parameter Q.
    * `log_omega0` (float, optional, default=0): only applicable if `kernel` is
        'sho'.  Log of the GP parameter omega0.
    * `log_sigma` (float, optional, default=0): only applicable if `kernel` is
        'matern32'.  Log of the GP parameter sigma.
    * `log_rho` (float, optional, default=0): only applicable if `kernel` is
        'matern32'.  Log of the GP parameter rho.
    * `eps` (float, optional, default=0): only applicable if `kernel` is
        'matern32'.  Log of the GP parameter epsilon.


    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params = []

    params += [ChoiceParameter(qualifier='kernel', value=kwargs.get('kernel', 'matern32'), choices=['matern32', 'sho'], description='Kernel for the gaussian process (see https://celerite.readthedocs.io/en/stable/python/kernel/)')]

    params += [FloatParameter(visible_if='kernel:sho', qualifier='log_S0', value=kwargs.get('log_S0', 0), default_unit=u.dimensionless_unscaled, description='Log of the GP parameter S0')]
    params += [FloatParameter(visible_if='kernel:sho', qualifier='log_Q', value=kwargs.get('log_Q', 0), default_unit=u.dimensionless_unscaled, description='Log of the GP parameter Q')]
    params += [FloatParameter(visible_if='kernel:sho', qualifier='log_omega0', value=kwargs.get('log_omega0', 0), default_unit=u.dimensionless_unscaled, description='Log of the GP parameter omega0')]

    params += [FloatParameter(visible_if='kernel:matern32', qualifier='log_sigma', value=kwargs.get('log_sigma', 0), default_unit=u.dimensionless_unscaled, description='Log of the GP parameter sigma')]
    params += [FloatParameter(visible_if='kernel:matern32', qualifier='log_rho', value=kwargs.get('log_rho', 0), default_unit=u.dimensionless_unscaled, description='Log of the GP parameter rho')]
    params += [FloatParameter(visible_if='kernel:matern32', qualifier='eps', value=kwargs.get('eps', 0.01), limits=(0,None), default_unit=u.dimensionless_unscaled, description='GP parameter epsilon')]

    # params += [FloatParameter(visible_if='kernel:jitter', qualifier='log_sigma', value=kwargs.get('log_sigma', np.log(0.01)), default_unit=u.dimensionless_unscaled, description='Log of the amplitude of the white noise')]

    constraints = []

    return ParameterSet(params), constraints

def emcee_move(feature, **kwargs):
    """
    Create a <phoebe.parameters.ParameterSet> for an emcee_move feature to attach
    to an <phoebe.parameters.solver.sampler.emcee> solver.

    Generally, this will be used as an input to the kind argument in
    <phoebe.frontend.bundle.Bundle.add_feature>.  If attaching through
    <phoebe.frontend.bundle.Bundle.add_feature>, all `**kwargs` will be
    passed on to set the values as described in the arguments below.  Alternatively,
    see <phoebe.parameters.ParameterSet.set_value> to set/change the values
    after creating the Parameters.

    Allowed to attach to:
    * components: not allowed
    * datasets: not allowed
    * solvers with kind: emcee


    Arguments
    ----------
    * `move` (string, optional, default='Stretch'): Type of move
        (see https://emcee.readthedocs.io/en/stable/user/moves/)
    * `weight` (float, optional, default=1.0): Weighted probability to apply to
        move.  Weights across all enabled emcee_move features will be renormalized
        to sum to 1 before passing to emcee.
    * `nsplits` (int, optional, default=2):
    * `randomize_split` (bool, optional, default=True):
    * `a` (float, optional, default=2.0):
    * `smode` (string, optional, default='auto'):
    * `s` (int, optional, default=16):
    * `bw_method` (string, optional, default='scott'):
    * `bw_constant` (float, optional, default=1.0):
    * `sigma` (float, optional, default=1e-5):
    * `gamma0_mode` (string, optional, default='auto'):
    * `gamma0` (float, optional, default=0.5):
    * `gammas` (float, optional, default=1.7):


    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params = []

    params += [ChoiceParameter(qualifier='move', value=kwargs.get('move', 'Stretch'), choices=['Stretch', 'Walk', 'KDE', 'DE', 'DESnooker'], description='Type of move (see https://emcee.readthedocs.io/en/stable/user/moves/)')]
    params += [FloatParameter(qualifier='weight', value=kwargs.get('weight', 1.0), limits=(0,None), default_unit=u.dimensionless_unscaled, description='Weighted probability to apply to move.  Weights across all enabled emcee_move features will be renormalized to sum to 1 before passing to emcee.')]

    # NOTE: RedBlue requires subclassing
    # params += [IntParameter(visible_if='move:RedBlue', qualifier='nsplits', value=kwargs.get('nsplits', 2), limits=(1,100), description='Passed directly to emcee. The number of sub-ensembles to use. Each sub-ensemble is updated in parallel using the other sets as the complementary ensemble. The default value is 2 and you probably won’t need to change that.')]
    # params += [BoolParameter(visible_if='move:RedBlue', qualifier='randomize_split', value=kwargs.get('randomize_split', True), description='Passed directly to emcee. Randomly shuffle walkers between sub-ensembles. The same number of walkers will be assigned to each sub-ensemble on each iteration.')]

    params += [FloatParameter(visible_if='move:Stretch', qualifier='a', value=kwargs.get('a', 2.0), limits=(None, None), default_units=u.dimensionless_unscaled, description='Passed directly to emcee.  The stretch scale parameter.')]

    params += [ChoiceParameter(visible_if='move:Walk', qualifier='smode', value=kwargs.get('smode', 'auto'), choices=['auto', 'manual'], description='Whether to manually provide the s parameter (number of helper walkers) or use all walkers in the complement by passing None to emcee.')]
    params += [IntParameter(visible_if='move:Walk,smode:manual', qualifier='s', value=kwargs.get('s', 16), limits=(1,None), description='Passed directly to emcee.  The number of helper walkers to use.')]

    params += [ChoiceParameter(visible_if='move:KDE', qualifier='bw_method', value=kwargs.get('bw_method', 'scott'), choices=['scott', 'silverman', 'constant'], description='Passed directly to emcee. The bandwidth estimation method.  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html')]
    params += [FloatParameter(visible_if='move:KDE,bw_method:constant', qualifier='bw_constant', value=kwargs.get('bw_constant', 1.0), limits=(None, None), default_unit=u.dimensionless_unscaled, description='Bandwidth estimation kde factor.  See https://docs.scipy.org/docs/scipy/reference/generated/scipy.stats.gaussian_kde.html')]

    params += [FloatParameter(visible_if='move:DE', qualifier='sigma', value=kwargs.get('sigma', 1e-5), limits=(0,None), default_unit=u.dimensionless_unscaled, description='Passed directly to emcee. The standard deviation of the Gaussian used to stretch the proposal vector.')]
    params += [ChoiceParameter(visible_if='move:DE', qualifier='gamma0_mode', value=kwargs.get('gamma0_mode', 'auto'), choices=['auto', 'manual'], description='Whether to manually provide gamma0 or default to 2.38/sqrt(2 * ndim)')]
    params += [FloatParameter(visible_if='move:DE,gamma0_mode:manual', qualifier='gamma0', value=kwargs.get('gamma0', 0.5), limits=(0,None), default_unit=u.dimensionless_unscaled, description='Passed directly to emcee.  The mean stretch factor for the proposal vector.')]

    params += [FloatParameter(visible_if='move:DESnooker', qualifier='gammas', value=kwargs.get('gammas', 1.7), limits=(0,None), default_unit=u.dimensionless_unscaled, description='Passed directly to emcee.  The mean stretch factor of the proposal vector.')]

    # NOTE: MH not implemented as it requires a callable
    # NOTE: Gaussian not implemented as it requires a covariance (as scalar, vector, or matrix)

    constraints = []

    return ParameterSet(params), constraints



# del deepcopy
# del _component_allowed_for_feature
# del download_passband, list_installed_passbands, list_online_passbands, list_passbands, parameter_from_json, parse_json, send_if_client, update_if_client
# del fnmatch
