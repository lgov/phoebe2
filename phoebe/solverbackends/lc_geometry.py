import numpy as np
from numpy import sin, cos, pi, sqrt, tan, arccos, arcsin
try:
    from scipy.optimize import newton, minimize, curve_fit
    from scipy.signal import find_peaks, savgol_filter
except ImportError:
    _can_compute_eclipse_params = False
else:
    _can_compute_eclipse_params = True

import logging
logger = logging.getLogger("SOLVER")
logger.addHandler(logging.NullHandler())

 # PREPROCESSING METHODS

def find_eclipse(phases, fluxes):
    '''
    Determines initial estimates for the eclipse position 
    by computing the median-crossings of the light curve.
    
    Parameters
    ----------
    phases: array-like
        Input phase array
    fluxes: array-like
        Corresponding fluxes
        
    Returns
    -------
    phase_min: float
        Center of eclipse
    edge_left: float
        Estimate of the left edge of the eclipse
    edge_right: float
        Estimate of the right edge of the eclipse
    '''
    phase_min = phases[np.nanargmin(fluxes)]
    ph_cross = phases[fluxes - np.nanmedian(fluxes) > 0]
    # this part looks really complicated but it really only accounts for eclipses split
    # between the edges of the phase range - if a left/right edge is not found, we look for 
    # it in the phases on the other end of the range
    # we then mirror the value back on the side of the eclipse position for easier width computation
    try:
        arg_edge_left = np.argmin(np.abs(phase_min - ph_cross[ph_cross<phase_min]))
        edge_left = ph_cross[ph_cross<phase_min][arg_edge_left]
    except:
        arg_edge_left = np.argmin(np.abs((phase_min+1)-ph_cross[ph_cross<(phase_min+1)]))
        edge_left = ph_cross[ph_cross<(phase_min+1)][arg_edge_left]-1
    try:
        arg_edge_right = np.argmin(np.abs(phase_min-ph_cross[ph_cross>phase_min]))
        edge_right = ph_cross[ph_cross>phase_min][arg_edge_right]
    except:
        arg_edge_right = np.argmin(np.abs((phase_min-1)-ph_cross[ph_cross>(phase_min-1)]))
        edge_right = ph_cross[ph_cross>(phase_min-1)][arg_edge_right]+1
                            
    return phase_min, edge_left, edge_right


def estimate_eclipse_positions_widths(phases, fluxes, diagnose_init=False):
    '''
    Initial estimates for both eclipse positions and widths.
    
    Parameters
    ----------
    phases: array-like
        Input phases
    fluxes: array-like
        Corresponding fluxes
    diagnose_init: bool
        If true, a diagnostic plot will be made
    
    Returns
    -------
        Dict of eclipse positions and widths.
    
    '''
    pos1, edge1l, edge1r = find_eclipse(phases, fluxes)
    fluxes_sec = fluxes.copy()
    fluxes_sec[((phases > edge1l) & (phases < edge1r)) | ((phases > edge1l+1) | (phases < edge1r-1))] = np.nan
    pos2, edge2l, edge2r = find_eclipse(phases, fluxes_sec)
    

    if diagnose_init:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,8))
        plt.plot(phases, fluxes, '.')
        plt.axhline(y=np.median(fluxes), c='orange')
        for i,x in enumerate([pos1, edge1l, edge1r]):
            ls = '-' if i==0 else '--'
            plt.axvline(x=x, c='r', ls=ls)
        for i,x in enumerate([pos2, edge2l, edge2r]):
            ls = '-' if i==0 else '--'
            plt.axvline(x=x, c='g', ls=ls)

    return {'ecl_positions': [pos1, pos2], 'ecl_widths': [edge1r-edge1l, edge2r-edge2l]}


class TwoGaussianModel(object):

    def __init__(self, phases, fluxes, sigmas):

        '''
        Computes the two-Gaussian model light curves of the input data.

        Parameters
        ----------
        phases: array-like
            Input orbital phases, must be on the range [-0.5,0.5]
        fluxes: array-like
            Input fluxes
        sigmas: array-like
            Input sigmas (corresponding flux uncertainities)
        '''

        self.phases = phases
        self.fluxes = fluxes
        self.sigmas = sigmas

        self.fit_twoGaussian_models()
        # compute all model light curves
        self.compute_twoGaussian_models()
        # compute corresponding BIC values
        self.compute_twoGaussian_models_BIC()
        
        # this is just all the parameter names for each model
        self.params = {'C': ['C'],
                'CE': ['C', 'Aell', 'mu1'],
                'CG': ['C', 'mu1', 'd1', 'sigma1'],
                'CGE': ['C', 'mu1', 'd1', 'sigma1', 'Aell'],
                'CG12': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2'],
                'CG12E1': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2', 'Aell'],
                'CG12E2': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2', 'Aell']}
        
        # choose the best fit as the one with highest BIC
        self.best_fit = list(self.models.keys())[np.nanargmax(list(self.bics.values()))]


    def extend_phasefolded_lc(self):
        '''
        Takes a phase-folded light curve on the range [-0.5,0.5] and extends it on range [-1,1]
        
        Parameters
        ----------
        phases: array-like
            Array of input phases spanning the range [-0.5,0.5]
        fluxes: array-like
            Corresponding fluxes, length must be equal to that of phases
        sigmas: array-like
            Corresponsing measurement uncertainties, length must be equal to that og phases
            
        Returns
        -------
        phases_extend, fluxes_extend, sigmas_extend: array-like
            Extended arrays on phase-range [-1,1]
        
        '''
        #make new arrays that would span phase range -1 to 1:
        fluxes_extend = np.hstack((self.fluxes[(self.phases > 0)], self.fluxes, self.fluxes[self.phases < 0.]))
        phases_extend = np.hstack((self.phases[self.phases>0]-1, self.phases, self.phases[self.phases<0]+1))

        if self.sigmas is not None:
            sigmas_extend = np.hstack((self.sigmas[self.phases > 0], self.sigmas, self.sigmas[self.phases < 0.]))
        else:
            sigmas_extend = None

        return phases_extend, fluxes_extend, sigmas_extend
    # HELPER FUNCTIONS

    def ellipsoidal(self, phi, Aell, phi0):
        # just the cosine component with the amplitude and phase offset as defined in Mowlavi (2017)
        return 0.5*Aell*np.cos(4*np.pi*(phi-phi0))

    def gaussian(self, phi, mu, d, sigma):
        # one Gaussian
        return d*np.exp(-(phi-mu)**2/(2*sigma**2))

    def gsum(self, phi, mu, d, sigma):
        # sum of Gaussians at phase range [-2,2]
        # default is [-0.5, 0.5] or [0,1] but we add the components left and right 
        # for better fit to the eclipses which can fall on the margins
        gauss_sum = np.zeros(len(phi))
        for i in range(-2,3,1):
            gauss_sum += self.gaussian(phi,mu+i,d,sigma)
        return gauss_sum

    # MODELS as defined in Mowalvi (2017)

    def const(self, phi, C):
        # constant term
        return C*np.ones(len(phi))

    def ce(self, phi, C, Aell, phi0):
        # constant + ellipsoidal
        return self.const(phi, C) - self.ellipsoidal(phi, Aell, phi0)

    def cg(self, phi, C, mu, d,  sigma):
        # constant + one gaussian (just primary eclipse)
        return self.const(phi, C) - self.gsum(phi, mu, d, sigma)

    def cge(self, phi, C, mu, d, sigma, Aell):
        # constant + one gaussian + ellipsoidal (just primary eclipse and elliposidal variations)
        return self.const(phi, C) - self.ellipsoidal(phi, Aell, mu) - self.gsum(phi, mu, d, sigma)

    def cg12(self, phi, C, mu1, d1, sigma1, mu2, d2, sigma2):
        # constant + two gaussians (both eclipses)
        return self.const(phi, C) - self.gsum(phi, mu1, d1, sigma1) - self.gsum(phi, mu2, d2, sigma2)

    def cg12e1(self, phi, C, mu1, d1, sigma1, mu2, d2, sigma2, Aell):
        # constant + two gaussians + elliposoidal centered on the primary eclipse
        return self.const(phi, C) - self.gsum(phi, mu1, d1, sigma1) - self.gsum(phi, mu2, d2, sigma2) - self.ellipsoidal(phi, Aell, mu1)

    def cg12e2(self, phi, C, mu1, d1, sigma1, mu2, d2, sigma2, Aell):
        # constant + two gaussians + elliposoidal centered on the secondary eclipse
        return self.const(phi, C) - self.gsum(phi, mu1, d1, sigma1) - self.gsum(phi, mu2, d2, sigma2) - self.ellipsoidal(phi, Aell, mu2)



    # FITTING

    @staticmethod
    def lnlike(y, yerr, ymodel):
        # returns the merit function value for optimization
        if yerr is not None:
            return -np.sum(np.log((2*np.pi)**0.5*yerr)+(y-ymodel)**2/(2*yerr**2))
        else:
            return -np.sum((y-ymodel)**2)

    def bic(self, ymodel, nparams):
        # returns the BIC number of a single model
        if self.sigmas is not None:
            return 2*self.lnlike(self.fluxes, self.sigmas, ymodel) - nparams*np.log(len(self.fluxes))
        else:
            return self.lnlike(self.fluxes, self.sigmas, ymodel)


    def fit_twoGaussian_models(self):
        '''
        Fits all seven models to the input light curve.
        '''
        # setup the initial parameters

        # fit all of the models to the data
        self.twogfuncs = {'C': self.const, 'CE': self.ce, 'CG': self.cg, 'CGE': self.cge, 'CG12': self.cg12, 'CG12E1': self.cg12e1, 'CG12E2': self.cg12e2}

        C0 = self.fluxes.max()
        ecl_dict = estimate_eclipse_positions_widths(self.phases, self.fluxes, diagnose_init=False)
        self.init_positions, self.init_widths = ecl_dict['ecl_positions'], ecl_dict['ecl_widths']
        mu10, mu20 = self.init_positions
        sigma10, sigma20 = self.init_widths
        d10 = self.fluxes.max()-self.fluxes[np.argmin(np.abs(self.phases-mu10))]
        d20 = self.fluxes.max()-self.fluxes[np.argmin(np.abs(self.phases-mu20))]
        Aell0 = 0.001

        init_params = {'C': [C0,],
            'CE': [C0, Aell0, mu10],
            'CG': [C0, mu10, d10, sigma10],
            'CGE': [C0, mu10, d10, sigma10, Aell0],
            'CG12': [C0, mu10, d10, sigma10, mu20, d20, sigma20],
            'CG12E1': [C0, mu10, d10, sigma10, mu20, d20, sigma20, Aell0],
            'CG12E2': [C0, mu10, d10, sigma10, mu20, d20, sigma20, Aell0]}

        # parameters used frequently for bounds
        fmax = self.fluxes.max()
        fmin = self.fluxes.min()
        fdiff = fmax - fmin

        bounds = {'C': ((0),(fmax)),
            'CE': ((0, 1e-6, -0.5),(fmax, fdiff, 0.5)),
            'CG': ((0., -0.5, 0., 0.), (fmax, 0.5, fdiff, 0.5)),
            'CGE': ((0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, fdiff)),
            'CG12': ((0.,-0.5, 0., 0., -0.5, 0., 0.),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5)),
            'CG12E1': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff)),
            'CG12E2': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff))}

        fits = {}

        # extend light curve on phase range [-1,1]
        phases_ext, fluxes_ext, sigmas_ext = self.extend_phasefolded_lc()

        for key in self.twogfuncs.keys():
            try:
                fits[key] = curve_fit(self.twogfuncs[key], phases_ext, fluxes_ext, p0=init_params[key], sigma=sigmas_ext, bounds=bounds[key])
            except Exception as err:
                logger.warning("2G model {} failed with error: {}".format(key, err))
                fits[key] = np.array([np.nan*np.ones(len(init_params[key]))])

        self.fits = fits


    def compute_twoGaussian_models(self):
        '''
        Computes the model light curves given the fit solutions.
        '''
        models = {}

        for fkey in self.fits.keys():
            models[fkey] = self.twogfuncs[fkey](self.phases, *self.fits[fkey][0])

        self.models = models


    def compute_twoGaussian_models_BIC(self):
        '''
        Computes the BIC value of each model light curve.
        '''
        bics = {}
        nparams = {'C':1, 'CE':3, 'CG':4, 'CGE':5, 'CG12':7, 'CG12E1':8, 'CG12E2':8}

        for mkey in self.models.keys():
            bics[mkey] = self.bic(self.models[mkey], nparams[mkey])

        self.bics = bics


    def compute_eclipse_params(self, diagnose=False):

        model_params = self.params[self.best_fit]

        sigma1 = self.fits[self.best_fit][0][model_params.index('sigma1')] if 'sigma1' in model_params else np.nan
        sigma2 = self.fits[self.best_fit][0][model_params.index('sigma2')] if 'sigma2' in model_params else np.nan
        mu1 = self.fits[self.best_fit][0][model_params.index('mu1')] if 'mu1' in model_params else np.nan
        mu2 = self.fits[self.best_fit][0][model_params.index('mu2')] if 'mu2' in model_params else np.nan
        C = self.fits[self.best_fit][0][model_params.index('C')]

        if not np.isnan(mu1) and not np.isnan(sigma1) and np.abs(sigma1) < 0.5:
            pos1 = mu1
            width1 = min(5.6*np.abs(sigma1), 0.5)
            depth1 = C - self.fluxes[np.argmin(np.abs(self.phases-pos1))]
        else:
            pos1 = np.nan
            width1 = np.nan
            depth1 = np.nan
        if not np.isnan(mu2) and not np.isnan(sigma2) and np.abs(sigma2) < 0.5:
            pos2 = mu2
            width2 = min(5.6*np.abs(sigma2), 0.5)
            depth2 = C - self.fluxes[np.argmin(np.abs(self.phases-pos2))]
        else:
            pos2 = np.nan
            width2 = np.nan
            depth2 = np.nan

        eclipse_edges = [pos1 - 0.5*width1, pos1+0.5*width1, pos2-0.5*width2, pos2+0.5*width2]


        if diagnose:
            phases_w, fluxes_w, sigmas_w = self.extend_phasefolded_lc()
            [ecl1_l, ecl1_r, ecl2_l, ecl2_r] = eclipse_edges

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            ax.plot(phases_w, fluxes_w, 'k.')
            plt.plot(phases_w, self.twogfuncs[self.best_fit](phases_w, *self.fits[self.best_fit][0]), '-', label=self.best_fit)
            lines = []
            lines.append(ax.axvline(x=pos1, c='#2B71B1', lw=2, label='primary'))
            lines.append(ax.axvline(x=pos2, c='#FF702F', lw=2, label='secondary'))
            lines.append(ax.axvline(x=ecl1_l, c='#2B71B1', lw=2, ls='--'))
            lines.append(ax.axvline(x=ecl1_r, c='#2B71B1', lw=2, ls='--'))
            lines.append(ax.axvline(x=ecl2_l, c='#FF702F', lw=2, ls='--'))
            lines.append(ax.axvline(x=ecl2_r, c='#FF702F', lw=2, ls='--'))
            drs = []
            for l,label in zip(lines,['pos1', 'pos2', 'ecl1_l', 'ecl1_r', 'ecl2_l', 'ecl2_r']):   
                dr = DraggableLine(l)
                dr.label = label
                dr.connect()   
                drs.append(dr) 
            ax.legend()
            plt.show(block=True)

            print('adjusting values')

            pos1 = drs[0].point.get_xdata()[0]
            pos2 = drs[1].point.get_xdata()[0]
            ecl1_l = drs[2].point.get_xdata()[0]
            ecl1_r = drs[3].point.get_xdata()[0]
            ecl2_l = drs[4].point.get_xdata()[0]
            ecl2_r = drs[5].point.get_xdata()[0]
            width1 = ecl1_r - ecl1_l
            width2 = ecl2_r - ecl2_l
            
            eclipse_edges = [ecl1_l, ecl1_r, ecl2_l, ecl2_r]


        return {
            'primary_width': width1,
            'secondary_width': width2,
            'primary_position': pos1,
            'secondary_position': pos2,
            'primary_depth': depth1,
            'secondary_depth': depth2,
            'eclipse_edges': eclipse_edges
        }


class DraggableLine:

    def __init__(self, p):
        self.point = p
        self.press = None

    def connect(self):
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.button_press_event)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.button_release_event)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)

    def disconnect(self):
        #disconnect all the stored connection ids
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)


    def button_press_event(self,event):
        if event.inaxes != self.point.axes:
            return
        contains = self.point.contains(event)[0]
        if not contains: return
        self.press = self.point.get_xdata(), event.xdata

    def button_release_event(self,event):
        self.press = None
        self.point.figure.canvas.draw()

    def motion_notify_event(self, event):
        if self.press is None: return
        if event.inaxes != self.point.axes: return
        xdata, xpress = self.press
        dx = event.xdata-xpress
        self.point.set_xdata(xdata+dx)
        self.point.figure.canvas.draw()


class GeometryParams(object):

    def __init__(self, eclipse_params, fit_eclipses=True, **kwargs):

        '''
        Computes estimates of ecc, w, rsum and teffratio based on the eclipse parameters.

        Parameters
        ----------
        eclipse_params: dict
            Dictionary of the eclipse parameters determined from the two-Gaussian model or manually.
        refine_with_ellc: bool
            If true, an ellc.lc model will be fitted to the eclipses only to further refine 
            rsum, teffratio, as well as rratio and incl.

        '''

        self.pos1 = eclipse_params['primary_position']
        self.pos2 = eclipse_params['secondary_position']
        self.width1 = eclipse_params['primary_width']
        self.width2 = eclipse_params['secondary_width']
        self.depth1 = eclipse_params['primary_depth']
        self.depth2 = eclipse_params['secondary_depth']
        self.edges = eclipse_params['eclipse_edges']
         # computation fails if sep<0, so we need to adjust for it here.
        sep = self.pos2 - self.pos1
        if sep < 0:
            self.sep = 1+sep
        else:
            self.sep = sep

        self._ecc_w()
        self._teffratio()
        self._rsum()

        if fit_eclipses:
            phases = kwargs.get('phases', [])
            fluxes = kwargs.get('fluxes', [])
            sigmas = kwargs.get('sigmas', [])

            if len(phases) == 0 or len(fluxes) == 0 or len(sigmas) == 0:
                raise ValueError('Please provide values for the phases, fluxes and sigmas of the light curve!')

            self.refine_with_ellc(phases, fluxes, sigmas)
        else:
            self.rratio = 1.
            self.incl = 90.


    @staticmethod
    def _f (psi, sep): # used in pf_ecc_psi_w
        return psi - sin(psi) - 2*pi*sep

    @staticmethod
    def _df (psi, sep): # used in pf_ecc_psi_w
        return 1 - cos(psi) +1e-6

    def _ecc_w(self):

        if np.isnan(self.sep) or np.isnan(self.width1) or np.isnan(self.width2):
            logger.warning('Cannot esimate eccentricty and argument of periastron: incomplete geometry information')
            return 0., np.pi/2
            
       
        psi = newton(func=self._f, x0=(12*np.pi*self.sep)**(1./3), fprime=self._df, args=(self.sep,), maxiter=5000)
        # ecc = sqrt( (0.25*(tan(psi-pi))**2+(swidth-pwidth)**2/(swidth+pwidth)**2)/(1+0.25*(tan(psi-pi))**2) )
        ecc = (np.sin(0.5*(psi-np.pi))**2+((self.width2-self.width1)/(self.width2+self.width1))**2*np.cos(0.5*(psi-np.pi))**2)**0.5
        try:
            w1 = np.arcsin((self.width1-self.width1)/(self.width2+self.width1)/ecc)
            w2 = np.arccos((1-ecc**2)**0.5/ecc * np.tan(0.5*(psi-np.pi)))

            w = w2 if w1 >= 0 else 2*pi-w2
        except:
            w = pi/2

        self.ecc = ecc
        self.per0 = w
        self.esinw = ecc*np.sin(w)
        self.ecosw = ecc*np.cos(w)


    def _t0_from_geometry(self, times, period=1, t0_supconj = 0, t0_near_times = True):

            delta_t0 = self.pos1*period
            t0 = t0_supconj + delta_t0

            if t0_near_times:
                if t0 >= times.min() and t0 <= times.max():
                    return t0
                else:
                    return t0 + int((times.min()/period)+1)*(period)
            else:
                return t0
        

    def _teffratio(self):
        self.teffratio = (self.depth2/self.depth1)**0.25

    
    def _rsum(self):
        self.rsum = np.pi * np.average([self.width1, self.width2])


    def refine_with_ellc(self, phases, fluxes, sigmas):
        try:
            import ellc
        except:
            raise ImportError('ellc is required for parameter refinement, please install it before running this step.')
        
        def wrap_around_05(phases):
            phases[phases>0.5] = phases[phases>0.5] - 1
            phases[phases<-0.5] = phases[phases<-0.5] + 1
            return phases

        def mask_eclipses():

            edges = wrap_around_05(np.array(self.edges))
            ecl1 = edges[:2]
            ecl2 = edges[2:]

            if ecl1[1]>ecl1[0]:
                mask1 = (phases>=ecl1[0]) & (phases<=ecl1[1])
            else:
                mask1 = (phases>=ecl1[0]) | (phases<=ecl1[1])

            if ecl2[1]>ecl2[0]:
                mask2 = (phases>=ecl2[0]) & (phases<=ecl2[1])
            else:
                mask2 = (phases>=ecl2[0]) | (phases<=ecl2[1])

            phases_ecl, fluxes_ecl, sigmas_ecl = phases[mask1 | mask2], fluxes[mask1 | mask2], sigmas[mask1 | mask2]
            phases_outofecl, fluxes_outofecl, sigmas_outofecl = phases[~(mask1 | mask2)], fluxes[~(mask1 | mask2)], sigmas[~(mask1 | mask2)]

            meanf = np.mean(fluxes_outofecl) - 1
            return phases_ecl, fluxes_ecl, sigmas_ecl, meanf


        def lc_model(phases_mask, rsum, rratio, teffratio, incl, meanf):

            r1 = rsum/(1+rratio)
            r2 = rsum*rratio/(1+rratio)
            sbratio = np.sign(teffratio) * teffratio**4

            return ellc.lc(phases_mask, r1, r2, sbratio, incl, 
                light_3 = 0, 
                t_zero = self.pos1, period = 1,
                q = 1,
                f_c = self.ecc**0.5*np.cos(self.per0), f_s = self.ecc**0.5*np.sin(self.per0),
                shape_1='roche', shape_2='roche', 
                ld_1='lin', ld_2='lin', ldc_1=0.5, ldc_2=0.5,
                gdc_1=0.32, gdc_2=0.32, heat_1=0.3, heat_2=0.3) + meanf

        def chi2(params, lc_data, meanf):
            rsum, rratio, teffratio, incl = params
            try:
                lcm = lc_model(lc_data[:,0], rsum, rratio, teffratio, incl, meanf)
                return 0.5 * np.sum((lc_data[:,1] - lcm) ** 2 / lc_data[:,2]**2)
            except Exception as e:
                return np.inf


        
        phases_mask, fluxes_mask, sigmas_mask, meanf = mask_eclipses()
        # phases_mask, fluxes_mask, sigmas_mask = phases, fluxes, sigmas
        lc_data = np.array([phases_mask, fluxes_mask, sigmas_mask]).T
        rsum_0 = self.rsum
        rratio_0 = 1.0
        teffratio_0 = self.teffratio 
        incl_0 = 90. 
        params_0 = [rsum_0, rratio_0, teffratio_0, incl_0]

        res = minimize(chi2, params_0, args=(lc_data, meanf), method='nelder-mead', options={'maxiter':10000})

        [self.rsum, self.rratio, self.teffratio, self.incl] = res.x