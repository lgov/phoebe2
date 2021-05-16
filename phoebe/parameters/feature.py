

from phoebe.parameters import *
from phoebe.parameters import constraint
from phoebe import u
from phoebe import conf

### NOTE: if creating new parameters, add to the _forbidden_labels list in parameters.py

def _component_allowed_for_feature(feature_kind, component_kind):
    _allowed = {}
    _allowed['spot'] = ['star', 'envelope']
    _allowed['pulsation'] = ['star', 'envelope']
    _allowed['gaussian_process'] = [None]

    return component_kind in _allowed[feature_kind]

def _dataset_allowed_for_feature(feature_kind, dataset_kind):
    _allowed = {}
    _allowed['spot'] = [None]
    _allowed['pulsation'] = [None]
    _allowed['gaussian_process'] = ['lc', 'rv', 'lp']

    return dataset_kind in _allowed[feature_kind]

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
    * `S0` (float, optional, default=0): only applicable if `kernel` is
        'sho'. GP parameter S0.
    * `Q` (float, optional, default=0): only applicable if `kernel` is
        'sho'.  GP parameter Q.
    * `omega0` (float, optional, default=0): only applicable if `kernel` is
        'sho'.  GP parameter omega0.
    * `sigma` (float, optional, default=0): only applicable if `kernel` is
        'matern32'.  GP parameter sigma.
    * `rho` (float, optional, default=0): only applicable if `kernel` is
        'matern32'.  GP parameter rho.
    * `eps` (float, optional, default=0): only applicable if `kernel` is
        'matern32'.  GP parameter epsilon.


    Returns
    --------
    * (<phoebe.parameters.ParameterSet>, list): ParameterSet of all newly created
        <phoebe.parameters.Parameter> objects and a list of all necessary
        constraints.
    """

    params = []

    params += [ChoiceParameter(qualifier='kernel', value=kwargs.get('kernel', 'matern32'), choices=['matern32', 'sho'], description='Kernel for the gaussian process (see https://celerite.readthedocs.io/en/stable/python/kernel/)')]

    params += [FloatParameter(visible_if='kernel:sho', qualifier='S0', value=kwargs.get('S0', 0), default_unit=u.dimensionless_unscaled, description='GP parameter S0')]
    params += [FloatParameter(visible_if='kernel:sho', qualifier='Q', value=kwargs.get('Q', 0), default_unit=u.dimensionless_unscaled, description='GP parameter Q')]
    params += [FloatParameter(visible_if='kernel:sho', qualifier='omega0', value=kwargs.get('omega0', 0), default_unit=u.dimensionless_unscaled, description='GP parameter omega0')]

    params += [FloatParameter(visible_if='kernel:matern32', qualifier='sigma', value=kwargs.get('sigma', 0), default_unit=u.dimensionless_unscaled, description='GP parameter sigma')]
    params += [FloatParameter(visible_if='kernel:matern32', qualifier='rho', value=kwargs.get('rho', 0), default_unit=u.dimensionless_unscaled, description='GP parameter rho')]
    params += [FloatParameter(visible_if='kernel:matern32', qualifier='eps', value=kwargs.get('eps', 0.01), limits=(0,None), default_unit=u.dimensionless_unscaled, description='GP parameter epsilon')]

    # params += [FloatParameter(visible_if='kernel:jitter', qualifier='sigma', value=kwargs.get('sigma', np.log(0.01)), default_unit=u.dimensionless_unscaled, description='amplitude of the white noise')]

    constraints = []

    return ParameterSet(params), constraints



# del deepcopy
# del _component_allowed_for_feature
# del download_passband, list_installed_passbands, list_online_passbands, list_passbands, parameter_from_json, parse_json, send_if_client, update_if_client
# del fnmatch
