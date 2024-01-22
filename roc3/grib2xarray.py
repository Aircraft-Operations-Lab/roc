#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import xarray as xr
import pygrib
import numpy

def get_perturbation_numbers(gribs):
    gribs.rewind()
    s = set()
    for grb in gribs:
        s.add(grb.perturbationNumber - 1)
    return list(s)

def get_pressure_levels(gribs):
    gribs.rewind()
    s = set()
    for grb in gribs:
        s.add(float(grb.level))
    return sorted(list(s))

def get_fcst_times(gribs):
    gribs.rewind()
    s = set()
    for grb in gribs:
        s.add(int(grb.stepRange))
    return list(s)

def get_par_names(gribs):
    gribs.rewind()
    s = set()
    for grb in gribs:
        s.add(grb.parameterName)
    return list(s)

def pl_gribs_to_xr_tbomet(gribs):
    """Return an xarray Dataset containing the ensemble information
    from gribs in the format used in TBO-MET

    Args:
    gribs - a gribs object as obtained with pygrib.open
    """
    latlons = gribs[1].latlons()
    lats = latlons[0][:,0]
    lons = latlons[1][0,:]
    levels = get_pressure_levels(gribs)
    params = get_par_names(gribs)
    pert_num = get_perturbation_numbers(gribs)
    fcst_times = get_fcst_times(gribs)
    assert len(fcst_times) == 1
    coordinates = [
        ('lat',  lats),
        ('lon',  lons),
        ('pl',    levels),
        ('ens_n', pert_num),
        ('fcst_step', fcst_times),
    ]
    axis_sizes = [len(c_values) for (c_name, c_values) in coordinates]
    nan_array = numpy.empty(axis_sizes)
    nan_array.fill(numpy.NaN)
    data_arrays = dict([(p, xr.DataArray(nan_array.copy(), coordinates)) for p in params])
    gribs.rewind()
    for grib in gribs:
        selector = {
            'pl': grib.level,
            'ens_n': grib.perturbationNumber - 1,
            'fcst_step': fcst_times[0],
        }
        data_arrays[grib.parameterName].loc[selector] = grib.values
    return xr.Dataset(data_arrays, dict(coordinates))

def read_day_from_directory(path, label):
    import time
    files_in_dir = os.listdir(path)
    matching_files = [f for f in files_in_dir if f.startswith(label)]
    matching_files.sort(key=lambda fname: int(fname[-7:-5]))
    ds_list = [pl_gribs_to_xr_tbomet(pygrib.open(path + f)) for f in matching_files]
    #return xr.merge(ds_list)
    #ds0 = ds_list[0]
    #for ds in ds_list[1:]:
    #    ds0.update(ds)
    #ds_list.sort(key=lambda ds: ds['Temperature'].coords['fcst_step'])
    return xr.concat(ds_list, dim='fcst_step')