from roc3.weather import *
from importlib import reload
from roc3.rfp import *
from roc3.apm import *
from roc3.bada4 import *
import pickle
import xarray as xr
import numpy as np


# output of climaccf should be inputted here as ds with BFFM2_Coef = True
def weather_intrp_save_pickle (ds, path_save):
    
    n_la = len(ds['latitude'].values)
    n_lo = len(ds['longitude'].values)
    n_t =  len(ds['time'].values)
    n_l =  len(ds['level'].values)
    n_m =  len(ds['number'].values)

    T = np.zeros((n_t,n_l,n_m,n_la,n_lo))
    U = np.zeros((n_t,n_l,n_m,n_la,n_lo))
    V = np.zeros((n_t,n_l,n_m,n_la,n_lo))

    aCCF_H2O = np.zeros((n_t,n_l,n_m,n_la,n_lo))
    r = np.zeros((n_t,n_l,n_m,n_la,n_lo))
    aCCF_NOx = np.zeros((n_t,n_l,n_m,n_la,n_lo))
    C1 = np.zeros((n_t,n_l,n_m,n_la,n_lo))
    C2 = np.zeros((n_t,n_l,n_m,n_la,n_lo))
    pcfa = np.zeros((n_t,n_l,n_m,n_la,n_lo))
    olr = np.zeros((n_t,n_l,n_m,n_la,n_lo))

    for i in range(n_t):
        for j in range(n_l):
            T [i,j,:,:,:]  = ds['t'].values[i,:, j,:,:]
            U [i,j,:,:,:]  = ds['u'].values[i,:, j,:,:]
            V [i,j,:,:,:]  = ds['v'].values[i,:, j,:,:] 
            aCCF_H2O [i,j,:,:,:] = ds['aCCF_H2O'].values[i,:, j,:,:]
            aCCF_NOx [i,j,:,:,:] = ds['aCCF_O3'].values[i,:, j,:,:] + ds['aCCF_CH4'].values[i,:, j,:,:]
            r [i,j,:,:,:]  = ds['r'].values[i,:, j,:,:] 
            C1 [i,j,:,:,:] = ds['C1'].values[i,:, j,:,:]  
            C2 [i,j,:,:,:] = ds['C2'].values[i,:, j,:,:]
            pcfa [i,j,:,:,:] = ds['pcfa'].values[i,:, j,:,:]
            olr [i,j,:,:,:] = ds['olr'].values[i,:, j,:,:]      
            

    times = []

    for ii in range (n_t):
        timE = 3*ii
        times.append(1526190000.0 + timE * 3600)


    ds1 = xr.Dataset(
            {
                "T": (("times", "levels","member", "lats", "longs"), T),
                "U": (("times", "levels","member", "lats", "longs"), U),
                "V": (("times", "levels","member", "lats", "longs"), V),

                "C1": (("times", "levels","member", "lats", "longs"), C1),
                "r": (("times", "levels","member", "lats", "longs"), r/100),
                "C2": (("times", "levels","member", "lats", "longs"), C2),
                "aCCF_H2O": (("times", "levels","member", "lats", "longs"), aCCF_H2O),
                "aCCF_NOx": (("times", "levels","member", "lats", "longs"), aCCF_NOx),
                "pcfa": (("times", "levels","member", "lats", "longs"), pcfa),
                "olr": (("times", "levels","member", "lats", "longs"), olr),

            },
            {"times": times,
            "member": np.arange(0,n_m,1),
            "levels":  ds['level'].values,
            "lats": ds['latitude'].values,
            "longs" : ds['longitude'].values},
            )

    ds1.to_netcdf(path_save + "intrp.nc")

    ws = WeatherStore_4D(path_save + "intrp.nc")
    
    # print(ws.values['U'].shape)
    # ws.reduce_domain({'lat': (43, 57), 'lon': (7, 32)})
    # print(ws.values['U'].shape)

    wm = ws.get_weather_model(n_m)

    with open(path_save + "pickle.nc", 'wb') as handle:
        pickle.dump(wm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return wm 