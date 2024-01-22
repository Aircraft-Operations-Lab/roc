#!/usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import warnings
from casadi import interpolant, vertcat
from scipy.signal import convolve2d, convolve
from datetime import datetime

from roc3.weather import *
from roc3.apm import smooth_zero_bound

class Convection_2Dt(object):
    bitriangular_filter = np.array([[.0625, .125, .0625], [.125, .25, .125], [.0625, .125, .0625]])
    triangular_filter = np.array([.25, .5, .25])
    categories = ['moderate', 'heavy', 'severe']
    def __init__(self, path, **config):
        self.cfg = {
            'format': np.float16,
            'll_resolution': 1.0,
            'convolution_mode': 'same',
            'severity': 'all',
            'weights': (0.25, 0.5, 0.25),
            'time_offset': 0.
            }
        eps = 1e-3
        atan = np.arctan
        self.P_transform = {
            'P2u': lambda y, eps=eps: 2/np.pi*np.tan(np.pi/2*((2*y -1)*(1 - 2*eps) + eps)),
            'u2P': lambda x, eps=eps: (2/np.pi*atan(np.pi/2*x)*0.5 + 0.5)/(1 - 2*eps) - eps,
            }
        self.cfg.update(config)
        self.npz = np.load(path, allow_pickle=True)
        self.axes = {}
        self.axes['lat'] = self.npz['PROB_MODERATE_CONVECTION'].item()['lats']
        self.axes['lon'] = self.npz['PROB_MODERATE_CONVECTION'].item()['longs']
        self.npz_resolution = self.axes['lat'][1] - self.axes['lat'][0]
        self.axes['levels'] = list(self.npz['PROB_MODERATE_CONVECTION'].item()['levels'])
        self.axes['times'] = self.npz['PROB_MODERATE_CONVECTION'].item()['times']+self.cfg["time_offset"]
        self.values = {}
        self.values['moderate'] = self.npz['PROB_MODERATE_CONVECTION'].item()['values'][:,0,:,:]
        self.values['heavy'] = self.npz['PROB_HEAVY_CONVECTION'].item()['values'][:,0,:,:]
        try:
            self.values['moderate_thr44'] = self.npz['PROB_MODERATE_CONVECTION_THR44'].item()['values'][:,0,:,:]
        except KeyError:
            warnings.warn("Could not find PROB_MODERATE_CONVECTION_THR44 in the convection npz")
        try:
            self.values['heavy_thr48'] = self.npz['PROB_HEAVY_CONVECTION_THR48'].item()['values'][:,0,:,:]
        except KeyError:
            warnings.warn("Could not find PROB_HEAVY_CONVECTION_THR48 in the convection npz")
        self.values['severe'] = self.npz['PROB_SEVERE_CONVECTION'].item()['values'][:,0,:,:]
        if self.cfg['severity'] == 'all':
            w = self.cfg['weights']
            self.values['convection'] = sum(w[i]*self.values[self.categories[i]] for i in range(3))
        else:
            self.values['convection'] = self.values[self.cfg['severity']]
        self.downsample_steps = int(np.log2(self.cfg['ll_resolution']//self.npz_resolution))
    def get_interpolant(self):
        # Caches the interpolant
        if not hasattr(self, 'I'):
            y = self.decimate_3d(self.values['convection'])
            n = 2**self.downsample_steps
            coords = (self.axes['lat'][::n], self.axes['lon'][::n],  self.axes['times'])
            y = y.transpose((0, 2, 1)) # from t, φ, λ to t, λ, φ
            y = self.P_transform['P2u'](y.flatten())
            # Since evaluation is (φ, λ, t) we need to give (t, λ, φ) before the flatten
            # to the casadi interpolant (quirk of this function)
            self.I = interpolant('conv', 'bspline', coords, y, {})
        return self.I
    def get_c_function(self):
        I = self.get_interpolant()
        return lambda lat, lon, t: self.P_transform['u2P'](I(vertcat(lat, lon, t)))
    def decimate(self, array):
        array = array.astype(self.cfg['format'])
        for i in range(self.downsample_steps):
            array = self.down2(array)
        return array
    def decimate_3d(self, array):
        for i in range(array.shape[0]):
            arr2 = self.decimate(array[i,:,:])
            nlat, nlon = arr2.shape
            array[i,:nlat,:nlon] = arr2
        return array[:,:nlat,:nlon]
    def down2(self, array):
        '''
        Decimates a 2D array by a factor of two after applying a triangular filter
        '''
        if len(array.shape) != 2:
            raise ValueError(f"Array to decimate must be 2D. Input has shape {array.shape}")
        filtered = convolve2d(array, self.bitriangular_filter, boundary='symm', mode=self.cfg['convolution_mode'])
        return filtered[::2,::2]
    def reduce_domain(self, bounds):
        slice_idx = {}
        if 'times' in bounds:
            bounds['times'] = list(bounds['times'])
            for i in (0, 1):
                bti = bounds['times'][i]
                if type(bti) == datetime:
                    bounds['times'][i] = bti.timestamp()
        else:
            slice_idx['times'] = slice(0, len(self.axes['times']) - 1)
        for ax_name, ax_bounds in bounds.items():
            slc = get_bound_indexes(self.axes[ax_name], ax_bounds)
            slice_idx[ax_name] = slc
            self.axes[ax_name] = self.axes[ax_name][slc]
        self.values['convection'] = self.values['convection'][slice_idx['times'],
                                                slice_idx['lat'],
                                                slice_idx['lon']]
