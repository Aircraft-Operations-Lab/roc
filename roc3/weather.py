#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage as ndimage
import casadi
import copy
import xarray as xr

from abc import ABC, abstractmethod
from roc3.apm import *

from datetime import datetime
from scipy.signal import convolve2d, convolve

inf = casadi.inf
vertcat = casadi.vertcat


def get_bound_indexes(arr, bounds, verbose=False):
    if verbose:
        print("[gbi] ", bounds, arr)
    if type(arr) == list:
        arr = np.array(arr)
    try:
        assert bounds[0] < bounds[1]
    except AssertionError:
        raise ValueError

    if arr[0] >= bounds[0]:
        low = 0
    else:
        low = np.argmax(arr > bounds[0])
        if low:
            low -= 1

    if arr[-1] <= bounds[-1]:
        high = None
    else:
        high = np.argmin(arr < bounds[1])
        if verbose:
            print(f"high is {high}")
        if high < len(arr):
            high += 1

    return slice(low, high)



def build_4D_interpolant(coords, values, name):
    y = values.transpose((0, 1, 3, 2))  # from t, P, φ, λ to (t, P, λ, φ)
    # if name == 'aCCF_nCont' or name == 'pcfa':
    #     y = ndimage.filters.gaussian_filter(y, sigma = 0.6, mode='nearest')
    #     print('Hey')
    # Since evaluation is (φ, λ, P, t) we need to give the array with (t, P, λ, φ) axes before the flatten
    # to the casadi interpolant (C2uirk of this function)
    cP = [100 * hPa for hPa in coords['levels']]
    xx = (coords['lat'], coords['lon'], cP, coords['times'])
    return casadi.interpolant(name, 'bspline', xx, y.flatten(), {})


class DummyWeather(object):
    def __init__(self, u=0, v=0, T=211, z=11000):
        self.u = u
        self.v = v
        self.T = T
        self.z = z
        self.n_members = 1

    def filter_pl_step(self, *args, **kwargs):
        pass

    def get_interpolants(self):
        wi = {'u': [lambda llt: self.u],
              'v': [lambda llt: self.v],
              'T': [lambda llt: self.T],
              'z': [lambda llt: self.z],
              }
        return wi


class WeatherScenario(ABC):
    @staticmethod
    def interpolant_builder(*args):
        return None

    @abstractmethod
    def u(self, lat, lon, pressure, t):
        pass

    @abstractmethod
    def v(self, lat, lon, pressure, t):
        pass

    @abstractmethod
    def T(self, lat, lon, pressure, t):
        pass

    @abstractmethod
    def H(self, lat, lon, pressure, t):
        pass

    def env_state(self, lat, lon, pressure, t):
        args = (lat, lon, pressure, t)
        uv = (self.u(*args), self.v(*args))
        return EnvironmentState(pressure, self.T(*args), wind_vector=uv)

    @classmethod
    def init_from_arrays(cls, coords, U, V, T, aCCF_H2O,r,C2,C1, pcfa, aCCF_NOx, olr,Z):
        arrs_names = zip((U, V, T,aCCF_H2O,r,C2,C1, pcfa, aCCF_NOx, olr, Z), ('U', 'V', 'T', 'aCCF_H2O','r','C2','C1', 'pcfa', 'aCCF_NOx', 'olr', 'Z'))
        if Z is None:
            arrs_names = list(arrs_names)[:-1]
        return cls(*[cls.interpolant_builder(coords, arr, name) for arr, name in arrs_names])


class WeatherModel(list):
    def __init__(self, scenarios, bounds):
        self.n_members = len(scenarios)  # number of scenarios
        self.bounds = bounds
        self.extend(scenarios)

    def get_slice_with_first_member(self):
        return self.__class__(self[:1], self.bounds)

    def get_slice_with_n_member(self,n):
        print('yes')
        return self.__class__(self[:n], self.bounds)

    def __repr__(self):
        return f"roc3.weather.WeatherModel with {self.n_members} members and bounds {self.bounds}"
    # def __getattr__(self, idx):
    # return self.scenarios[idx]
    # def __setattr__(self, idx, value):
    # self.scenarios[idx] = value


class WeatherArrays4D(object):

    def __init__(self, axes, u, v, T, Z=None):
        self.axes = axes
        self.u = u
        self.v = v
        self.T = T
        self.Z = Z
        self.n = 1
        if len(self.u.shape) == 5:
            self.n = self.u.shape[2]

    @classmethod 
    def init_from_dataset(cls, ds):
        return cls(ds.coords, ds['U'], ds['V'], ds['T'], ds['aCCF_H2O'], ds['r'], ds['C2'], ds['C1'], ds['pcfa'], ds['aCCF_NOx'], ds['olr'])

    @classmethod
    def load_from_file(cls, f):
        npz = np.load(f, allow_pickle=True)
        ax_names = ['times', 'levels', 'member', 'lat', 'lon']
        axes = {name: npz[name] for name in ax_names if name in npz}
        return cls(axes, npz['u'], npz['v'], npz['T'], npz['aCCF_H2O'], npz['r'], npz['C2'], npz['C1'], npz['pcfa'], npz['aCCF_NOx'], npz['olr'], npz['Z'])

    def get_average_arrays(self, num_format=np.float32):
        arrs = [self.u, self.v, self.T]
        return [np.mean(arr, axis=2).astype(num_format) for arr in arrs]

    def get_constants_dictionary(self):
        d = {}
        for label, ax_la in self.axes.items():
            ax = np.array(ax_la)
            d[f'n_{label}'] = ax.shape[0]
            d[f'{label}_min'] = ax.min()
            d[f'{label}_max'] = ax.max()
            d[f'{label}_range'] = ax.max() - ax.min()
            d[f'{label}_step'] = (ax.max() - ax.min()) / (ax.shape[0] - 1)
        return d

    def save(self, f):
        axarr = {k: np.array(v) for k, v in self.axes.items()}
        np.savez_compressed(f,
                            u=self.u, v=self.v, T=self.T, Z=self.Z,
                            **axarr)


class ISAWeather(WeatherScenario):
    def __init__(self):
        self.isa = ISA()
        self.n_members = 1

    def filter_pl_step(self, *args, **kwargs):
        pass

    def u(self, lat, lon, pressure, t):
        return 0

    def v(self, lat, lon, pressure, t):
        return 0

    def T(self, lat, lon, pressure, t):
        return self.isa.IT(H2h(P2Hp(pressure)))

    def H(self, lat, lon, pressure, t):
        return H2h(P2Hp(pressure))




class WeatherScenario4D(WeatherScenario):
    interpolant_builder = build_4D_interpolant

    def __init__(self, Iu, Iv, IT, IaCCF_H2O, Ir, IC2, IC1, Ipcfa, IaCCF_NOx, Iolr, IZ=None):
        self.Iu = Iu
        self.Iv = Iv
        self.IT = IT
        self.IaCCF_H2O = IaCCF_H2O
        self.Ir  = Ir
        self.IC2 = IC2
        self.IC1 = IC1
        self.Ipcfa = Ipcfa
        self.Iolr = Iolr
        self.IaCCF_NOx = IaCCF_NOx
        self.IZ = IZ

    # @classmethod
    # def init_from_arrays(cls, coords, U, V, T, Z):
    #     arrs_names = zip((U, V, T, Z), ('U', 'V', 'T', 'Z'))
    #     if Z is None:
    #         arrs_names = list(arrs_names)[:-1]
    #     return cls(*[build_4D_interpolant(coords, arr, name) for arr, name in arrs_names])
    def u(self, lat, lon, pressure, t):
        return self.Iu(vertcat(lat, lon, pressure, t))

    def v(self, lat, lon, pressure, t):
        return self.Iv(vertcat(lat, lon, pressure, t))

    def T(self, lat, lon, pressure, t):
        return self.IT(vertcat(lat, lon, pressure, t))

    def aCCF_H2O(self, lat, lon, pressure, t):
        return self.IaCCF_H2O(vertcat(lat, lon, pressure, t))
        
    def r(self, lat, lon, pressure, t):
        return self.Ir(vertcat(lat, lon, pressure, t))  

    def C2(self, lat, lon, pressure, t):
        return self.IC2(vertcat(lat, lon, pressure, t))   

    def C1(self, lat, lon, pressure, t):
        return self.IC1(vertcat(lat, lon, pressure, t))  

    def pcfa(self, lat, lon, pressure, t):
        return self.Ipcfa(vertcat(lat, lon, pressure, t))  

    def aCCF_NOx(self, lat, lon, pressure, t):
        return self.IaCCF_NOx(vertcat(lat, lon, pressure, t))       

    def olr(self, lat, lon, pressure, t):
        return self.Iolr(vertcat(lat, lon, pressure, t))       

    def H(self, lat, lon, pressure, t):
        if self.IZ is None:
            return P2Hp(pressure)
        else:
            return self.IZ(vertcat(lat, lon, pressure, t))


class GeoArrayHandler(object):
    bitriangular_filter = np.array([[.0625, .125, .0625], [.125, .25, .125], [.0625, .125, .0625]])
    triangular_filter = np.array([.25, .5, .25])
    axes: dict

    def get_coords(self):
        new_axes = {}
        for name, ax in self.axes.items():
            if not self.cfg['predecimate'] and name in ('lat', 'lon'):
                new_axes[name] = ax[::2 ** self.downsample_steps]
            else:
                new_axes[name] = ax[:]
        return new_axes

    def decimate(self, array):
        array = array.astype(self.cfg['format'])
        for i in range(self.downsample_steps):
            array = self.down2(array)
        return array

    def decimate_3d(self, array):
        for i in range(array.shape[0]):
            arr2 = self.decimate(array[i, :, :])
            nlat, nlon = arr2.shape
            array[i, :nlat, :nlon] = arr2
        return array[:, :nlat, :nlon]

    def decimate_4d(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                arr2 = self.decimate(array[i, j, :, :])
                nlat, nlon = arr2.shape
                array[i, j, :nlat, :nlon] = arr2
        return array[:, :, :nlat, :nlon]

    def decimate_5d(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                for k in range(array.shape[2]):
                    arr2 = self.decimate(array[i, j, k, :, :])
                    nlat, nlon = arr2.shape
                    array[i, j, k, :nlat, :nlon] = arr2
        return array[:, :, :, :nlat, :nlon]

    @classmethod
    def down2_coord(cls, array):
        return array[::2]

    def down2(self, array):
        '''
        Decimates a 2D array by a factor of two after applying a triangular filter
        '''
        if len(array.shape) != 2:
            raise ValueError(f"Array to decimate must be 2D. Input has shape {array.shape}")
        filtered = convolve2d(array, self.bitriangular_filter, boundary='symm', mode=self.cfg['convolution_mode'])
        return filtered[::2, ::2]


class WeatherStore(GeoArrayHandler):
    def __init__(self):
        pass








# class ArraySlicer(object):
#     def __init__(self, array, axes_list):
#         self.array = array
#         self.axes_list = axes_list
#     def __getattr__(self, )


class WeatherStore_4D(WeatherStore):

    def __init__(self, path, flipud='auto', **weather_config):
        # values array axes: times, levels, members, lats, lons
        self.path = path
        self.cfg = {
            'format': np.float32,
            'downsample_format': np.float16,
            'll_resolution': 1.0,
            'convolution_mode': 'same',
            'skip_geopotential': True,
            'predecimate': True,
            'time_offset': 0.,
        }
        self.cfg.update(weather_config)
        if self.cfg['skip_geopotential']:
            self.variable_names = ['U', 'V', 'T', 'aCCF_H2O','r','C2','C1', 'pcfa', 'aCCF_NOx', 'olr']
        else:
            self.variable_names = ['U', 'V', 'T', 'aCCF_H2O','r','C2','C1', 'pcfa', 'aCCF_NOx', 'olr', 'Z']
        self.npz = xr.open_dataset(path)
        self.axes = {}
        self.axes['lat'] = self.npz['lats'].values
        if flipud == 'auto':
            flipud = self.axes['lat'][1] < self.axes['lat'][0]
        if flipud:
            self.axes['lat'] = self.axes['lat'][::-1]
        self.axes['lon'] = self.npz['longs'].values
        self.npz_resolution = self.axes['lat'][1] - self.axes['lat'][0]
        self.axes['levels'] = list(self.npz['levels'].values)
        self.axes['times'] = self.npz['times'].values + self.cfg["time_offset"]
        self.n_members = len(self.npz['member'].values)
        self.members = range(self.n_members)
        if self.cfg['ll_resolution'] == self.npz_resolution:
            self.downsample_steps = 0
        else:
            self.downsample_steps = int(np.log2(self.cfg['ll_resolution'] // self.npz_resolution))
        self.values = {}
        if 'level' in self.cfg:
            if self.cfg['level'] not in self.axes['levels']:
                raise ValueError
        for tag in self.variable_names:
            if flipud:
                A = self.npz[tag].values[:, :, :, ::-1, :].astype(self.cfg['format'])
            else:
                A = self.npz[tag].values[:, :, :, :, :].astype(self.cfg['format'])
            if self.cfg['predecimate']:
                A = self.decimate_5d(A)
            self.values[tag] = A
        if self.cfg['predecimate']:
            self.axes['lat'] = self.axes['lat'][::2 ** self.downsample_steps]
            self.axes['lon'] = self.axes['lon'][::2 ** self.downsample_steps]

    def reduce_domain(self, bounds, verbose=False):
        slice_idx = {}
        if 'times' in bounds:
            bounds['times'] = list(bounds['times'])
            for i in (0, 1):
                bti = bounds['times'][i]
                if type(bti) == datetime:
                    bounds['times'][i] = bti.timestamp()
        else:
            slice_idx['times'] = slice(0, len(self.axes['times']))
        for ax_name, ax_bounds in bounds.items():
            try:
                slc = get_bound_indexes(self.axes[ax_name], ax_bounds)
            except ValueError:
                print(f"Could not reduce domain along the '{ax_name}' axis")
                print(f"Current bounds: ({self.axes[ax_name][0]}, {self.axes[ax_name][-1]})")
                print(f"Desired bounds: {ax_bounds}")
                raise
            slice_idx[ax_name] = slc
            self.axes[ax_name] = self.axes[ax_name][slc]
        for tag in self.variable_names:
            if verbose:
                print("Reducing domain from shape: ", self.values[tag].shape)
            self.values[tag] = self.values[tag][slice_idx['times'],
                               :,  # get all members
                               :,  # get all levels
                               slice_idx['lat'],
                               slice_idx['lon']]
            if verbose:
                print("Reducing domain to   shape: ", self.values[tag].shape)
                verbose = False

    def get_weather_model(self, n_members=casadi.inf):
        coords = self.get_coords()
        scenarios = []
        n_members = min(self.n_members, n_members)
        for i in range(n_members):
            variable_arrays = []
            for v in self.variable_names:
                arr = self.values[v][:, :, i, :, :]
                if not self.cfg['predecimate']:
                    arr = self.decimate_4d(arr)
                variable_arrays.append(arr)
            if 'Z' not in self.variable_names:
                variable_arrays.append(None)
            scenario = WeatherScenario4D.init_from_arrays(coords, *variable_arrays)
            scenarios.append(scenario)
        bounds = {}
        for name, axis in coords.items():
            bounds[name] = (min(axis), max(axis))
        return WeatherModel(scenarios, bounds)

    def get_weather_arrays(self, n_members=casadi.inf):
        variable_arrays = []
        n_members = min(self.n_members, n_members)
        i_lvl_50 = self.axes['levels'].index(50)
        i_lvl_300 = self.axes['levels'].index(300)
        for v in self.variable_names:
            arr = self.values[v][:, i_lvl_50:(i_lvl_300 + 1), :n_members, :, :]
            if not self.cfg['predecimate']:
                arr = self.decimate_5d(arr)
            variable_arrays.append(arr)
        if 'Z' not in self.variable_names:
            variable_arrays.append(None)
        va = variable_arrays
        new_axes = copy.deepcopy(self.axes)
        new_axes['levels'] = new_axes['levels'][i_lvl_50:(i_lvl_300 + 1)]
        return WeatherArrays4D(new_axes, *va)

    @property
    def ax_names(self):
        return ['times', 'levels', 'member', 'lat', 'lon']

    @staticmethod
    def sort_axes(axes):
        axes_sorted = {}
        for k in self.ax_names:
            axes_sorted[k] = axes[k]
        return axes_sorted

    def as_xarray_dataset(self):
        import xarray as xr
        axes = copy.deepcopy(self.axes)
        axes['member'] = np.array(range(self.n_members))
        variables = {}
        for var, values in self.values.items():
            variables[var] = xr.DataArray(values,
                                          dims=self.ax_names,
                                          coords=[axes[a] for a in self.ax_names])
        return xr.Dataset(data_vars=variables)

    def get_weather_arrays_resampled(self, shape_lat_lon_t=(32, 32, 16), n_members=casadi.inf):
        import xarray as xr
        new_axes = {}
        for ax_name, ax_length in zip(('lat', 'lon', 'times'), shape_lat_lon_t):
            new_axes[ax_name] = np.linspace(self.axes[ax_name][0], self.axes[ax_name][-1], ax_length)
        new_axes['levels'] = [50, 200, 250, 300]  # hardcoded
        n_members = min(self.n_members, n_members)
        new_axes['member'] = np.array(range(n_members))
        ds_in = self.as_xarray_dataset()
        new_shape = [len(new_axes[a]) for a in self.ax_names]
        placeholder_dataarray = xr.DataArray(data=np.zeros(new_shape),
                                             dims=self.ax_names,
                                             coords=[new_axes[a] for a in self.ax_names])
        return WeatherArrays4D.init_from_dataset(ds_in.interp_like(placeholder_dataarray))

