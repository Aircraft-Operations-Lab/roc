import pyomo.environ as pe
import pyomo.opt as po
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from roc3.accf import *
from roc3.apm import *

import matplotlib as mpl
from scipy import integrate


def compute_EI_NOx(fuel_flow, Mach,  C1, C2):
    ff_ci = C1 * (fuel_flow/2) * np.exp(0.2 * (Mach ** 2))
    return C2 * np.exp( 3.215 +  0.7783 * np.log(ff_ci))


def aCCF_O3(t, z):
    accf_o3 = 15 * (-5.20e-11 + 2.30e-13 * t + 4.85e-16 * z - 2.04e-18 * t * z) * 1.37 / 11
    return accf_o3


def aCCF_CH4(z, fin):
    accf_ch4 = 10.5 * (-9.83e-13 + 1.99e-18 * z - 6.32e-16 * fin + 6.12e-21 * z * fin)* 1.29 * 1.18 / 35
    return accf_ch4


def aCCF_H2O(pv):
    accf_h2o =  15 * (4.05e-16 + 1.48e-10 * pv) / 3
    return accf_h2o


def aCCF_dCont(olr):
    accf_dCont = 13.9 * 0.0151 * 1e-10 * (-1.7 - 0.0088 * olr) * 0.42/ 3
    return accf_dCont


def aCCF_nCont(t):
    accf_nCont = 13.9 * 0.0151 * (1e-10 * (0.0073 * np.power(10, 0.0107 * t) - 1.03)) * 0.42/ 3
    return accf_nCont


def G (P):
    G_ = (1.23 * 1004 * P)/(0.6222 * 43130000 * 0.7)
    return G_

def T_crit (G):
    T_c = -46.46 + 9.43 * np.log(G - 0.053) + 0.72 * np.power(np.log(G - 0.053),2) + 273.15
    return T_c

def rlw (q, P, el_t_amb):
    r_h = (q * P * (461.51/287.05)) / el_t_amb
    return r_h

def rh(q, T, p):
    R_d = 287.05
    R_v = 461.51
    rh = (q * p * (R_v / R_d)) / e_sat_liquid(T)
    return rh

def e_sat_liquid (t):
    eL_T = 100 * np.exp(-6096.9385/t + 16.635794 - 0.02711193 * t + 1.673952 * 1e-5 * np.power(t, 2) + 2.433502 * np.log(t))
    return eL_T



def rhi(q, T, p):
    R_d = 287.05
    R_v = 461.51
    return (q * p * (R_v /R_d)) / e_sat_ice(T)



def e_sat_ice(t):
    return 100 * np.exp(  # type: ignore[return-value]
        (-6024.5282 / t)
        + 24.7219
        + (0.010613868 * t)
        - (1.3198825e-5 * (t**2))
        - 0.49382577 * np.log(t)
    )

def r_cont (G, t, t_c, el_t_c, el_t_amb):
    r_sac = (G * (t - t_c) + el_t_c) / el_t_amb
    return r_sac


T0_MSL = 288.15
P0_MSL = 101325.0
lapse_rate_troposphere = -0.0065

def Hp2P(Hp):
    return P0_MSL*(1 + lapse_rate_troposphere*Hp/T0_MSL)**5.2561


def FL2P (FL):
    Hp = FL * 30.48
    P = Hp2P(Hp)/100 #hpa
    return P


def sigmd (x, c, x_cut):
    return 1 / (1 + np.exp( -c * (x-x_cut) ) )


def sac (r_sac, r_hm, T, T_crit):
    T_condition = sigmd (T_crit, 3, T)
    R_condition = sigmd (r_hm, 20, r_sac)
    sac_ = R_condition * T_condition
    return sac_
    

def pcfa(r, t, r_thr, t_thr, r_pa, t_pa):
    t_condition = 1/(1+np.exp(-t_pa*(-t+t_thr)))
    r_condition = 1/(1+np.exp(-r_pa*( r-r_thr)))
    pcfa_ = t_condition * r_condition
    return pcfa_


def plot_prof (trj_output, path, flight):
    fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(14,13), sharex=False)
    font = {'family' : 'normal',
            'size'   : 11}

    alpha = list(trj_output.keys())

    cmap_dp = mpl.cm.viridis
    norm_dp = mpl.colors.Normalize(vmin=0, vmax=float(len(alpha))-1)
    col = []

    for i in range (0, len(alpha)):
        col.append (cmap_dp(norm_dp(float(i))))

    atr_per = []
    soc_per = []
    for i, trj_index in enumerate(trj_output):
        distance_flown = trj_output[trj_index]['distance_flown']
        ATR_cont = trj_output[trj_index]['Cont']
        ATR_H2O = trj_output[trj_index]['H2O']
        ATR_NOx = trj_output[trj_index]['NOx']
        ATR_net = trj_output[trj_index]['ATR_Net']

        ATR = trj_output[trj_index]['ATR']
        SOC = trj_output[trj_index]['SOC']


        lat = trj_output[trj_index]['lat']
        lon = trj_output[trj_index]['lon']
        h = trj_output[trj_index]['h']
        tas = trj_output[trj_index]['tas']

        ax = axes[0,0]

        ax.plot(distance_flown*0.539957/1000, np.mean(ATR_cont,axis = 0), label = 'EI$_m$ = {}'.format(alpha[i]) , color = col[i])   
        ax.fill_between(distance_flown*0.539957/1000, np.min(ATR_cont,axis = 0), np.max(ATR_cont,axis = 0), color = col[i], alpha=0.3) 
        ax.set_ylabel("ATR of Contrails [K]")
        ax.legend(ncol = 2)


        ax = axes[0,1]
        ax.plot(distance_flown*0.539957/1000, np.mean(ATR_NOx,axis = 0), color = col[i])    
        ax.fill_between(distance_flown*0.539957/1000, np.min(ATR_NOx,axis = 0), np.max(ATR_NOx,axis = 0), color = col[i], alpha=0.3) 
        ax.set_ylabel("ATR of NO$_{x}$ emis. [K]")


        ax = axes[1,0]
        ax.plot(distance_flown*0.539957/1000, np.mean(ATR_H2O,axis = 0), color = col[i])    
        ax.fill_between(distance_flown*0.539957/1000, np.min(ATR_H2O,axis = 0), np.max(ATR_H2O,axis = 0), color = col[i], alpha=0.3) 
        ax.set_ylabel("ATR of Water Vapour [K]")
        ax.set_xlabel("Distance flown [nm]")

        ax = axes[1,1]
        ax.plot(distance_flown*0.539957/1000, np.mean(ATR_net,axis = 0), color = col[i])    
        ax.fill_between(distance_flown*0.539957/1000, np.min(ATR_net,axis = 0), np.max(ATR_net,axis = 0), color = col[i], alpha=0.3) 
        ax.set_ylabel("Net ATR [K]")
        ax.set_xlabel("Distance flown [nm]")


        ax = axes[2,0]
        ax.plot(distance_flown*0.539957/1000, h, color = col[i])    
        ax.set_ylabel("Altitude [m]")
        ax.set_xlabel("Distance flown [nm]")


        ax = axes[2,1]
        ax.plot(distance_flown*0.539957/1000, tas, color = col[i])    
        ax.set_ylabel("True airspeed [m/s]")
        ax.set_xlabel("Distance flown [nm]")


        ax = axes[3,0]
        ax.plot(lon, lat, color = col[i])    
        ax.set_ylabel("latitude [deg]")
        ax.set_xlabel("longitude [deg]")
        
        
        
        ax = axes[3,0]
        ax.plot(lon, lat, color = col[i])    
        ax.set_ylabel("latitude [deg]")
        ax.set_xlabel("longitude [deg]")
        
        atr_per.append(np.mean(ATR))
        soc_per.append(np.mean(SOC))


    atr_per_ = np.array(atr_per)
    soc_per_ = np.array(soc_per)

    atr_per_ = 100* (atr_per_ - atr_per_ [0]) / np.abs(atr_per_[0])
    soc_per_ = 100* (soc_per_ - soc_per_ [0]) / np.abs(soc_per_[0]) 
        
    ax = axes[3,1]
    ax.plot(atr_per_, soc_per_, color = col[i])    
    ax.set_ylabel("$\Delta$ ATR [%]")
    ax.set_xlabel("$\Delta$ SOC [%]")
    plt.savefig(path + f'profile_flight_{flight}.png',bbox_inches='tight')
    

def process_trjs (trj_output):
    C_ATR = list(trj_output.keys())
    dfs_p = {}
    atr_par = []
    cost_par = []
    for i, CI in enumerate(list(trj_output.keys())):
        atr_par.append(np.mean (trj_output[CI]['ATR']))
        cost_par.append(1e-3 * np.mean (trj_output[CI]['SOC']))
        
    min_cost = np.min (cost_par)
    
    solver = po.SolverFactory('glpk')

    index_selected = []

    routing  = np.arange(0,15,0.1)

    for opt in range (len(routing)):
        per = routing[opt]

        
        model = pe.ConcreteModel()
        model.N = pe.RangeSet(1, len(C_ATR))

        a = {}
        c = {}


        for index in range (1, len(C_ATR)+1):
            c[index] = 1e12*atr_par[index-1]
            a[index] = cost_par[index-1]

                    
        model.c = pe.Param(model.N, initialize=c)
        model.a = pe.Param(model.N, initialize=a)

        model.b = pe.Param(initialize= (min_cost+min_cost*per/100))

        model.x = pe.Var(model.N, domain=pe.Binary)

        obj_expr  = sum(model.c[i] * model.x[i] for i in model.N)
        model.obj = pe.Objective(sense=pe.minimize, expr=obj_expr)

        con_lhs_expr = sum(model.a[i] * model.x[i] for i in model.N)
        con_rhs_expr = model.b
        model.con    = pe.Constraint(expr=(con_lhs_expr <= con_rhs_expr))


        model.tasks_done = pe.ConstraintList()


        lhs = sum(model.x[i] for i in range (1, len(C_ATR)+1))
        rhs = 1
        model.tasks_done.add(lhs == rhs)



        result = solver.solve(model)
        index_selected.append (np.where(np.array([pe.value(model.x[i]) for i in model.N]) == 1) [0][0])

        # for i in model.N:
        #     print(pe.value(model.x[i]))
        # print(pe.value(model.obj))

        # SOC_net.append(sum(pe.value(model.a[i]) * pe.value(model.x[i])for i in model.N))
        # ATR_net.append(sum(pe.value(model.c[i]) * pe.value(model.x[i])for i in model.N))
    list_idx = np.unique (index_selected, return_index=True)[1]
    list_sel = [index_selected[index] for index in sorted(list_idx)]
    for i, index_sel in enumerate (list_sel):
        dict_copy = {}
        EI_coef = C_ATR [i]  
        dict_copy = trj_output [C_ATR [index_sel]] 
        dfs_p [EI_coef] = dict_copy   
    return dfs_p



from scipy import integrate
def get_profile (interp_wther, trj_set, confg):
    N = 10
    weights = list(trj_set.keys())
    trj_output = {}
    for i, alp in enumerate(weights):
        profile = {}
        
        trj = trj_set[alp]
        profile['lat'] = np.array(trj.x.lat)
        profile['lon'] = np.array(trj.x.lon)
        profile['lon'] = np.array(trj.x.lon)
        profile['tas'] = np.array(trj.x.tas)
        profile['distance_flown'] = np.array(trj.t)
        h = np.array(trj.x.h)
        profile ['h'] = h

        P = Hp2P(h) # Pa
            
        ATR_cont = np.zeros ((N,len(trj.x['h'])))  
        ATR_H2O  = np.zeros ((N,len(trj.x['h'])))  
        ATR_NOx  = np.zeros ((N,len(trj.x['h'])))  
        ATR_CO2  = np.zeros ((N,len(trj.x['h'])))  
        ATR_net  = np.zeros ((N,len(trj.x['h']))) 

        time = np.zeros (N)
        mass = np.zeros (N)

        distance_flown =  np.array(trj.t)

        ds = np.zeros (len(trj.x['h']))
        for j in range(N):
            time[j] = trj.x[f't_{0}'][-1] - trj.x[f't_{0}'][0]
            mass[j] = trj.x[f'm_{0}'][0]  - trj.x[f'm_{0}'][-1]

            for k in range(1,len(trj.x.lat)):
                ds[k] = distance_flown[k] - distance_flown[k-1]
                delta_mass = -(trj.x[f'm_{0}'][k] - trj.x[f'm_{0}'][k-1])/ds[k]

                T = interp_wther['t'][j]([trj.x['lat'][k], trj.x['lon'][k], P[k]])
                R = interp_wther['r'][j]([trj.x['lat'][k], trj.x['lon'][k], P[k]])
                Q = interp_wther['q'][j]([trj.x['lat'][k], trj.x['lon'][k], P[k]])
                aCCF_CH4 = interp_wther['aCCF_CH4'][j]([trj.x['lat'][k], trj.x['lon'][k], P[k]])
                Z = interp_wther['z'][j]([trj.x['lat'][k], trj.x['lon'][k], P[k]])
                PV = interp_wther['pv'][j]([trj.x['lat'][k], trj.x['lon'][k], P[k]])
                OLR = interp_wther['olr'][j]([trj.x['lat'][k], trj.x['lon'][k], P[k]])

                ## Ozone
                EI_NOx = 12
                accf_O3 = 0.001 * aCCF_O3 (T, Z)  * EI_NOx  * delta_mass
                        
                if accf_O3<0:
                    accf_O3=0 
                accf_CH4 = 0.001 * aCCF_CH4 * EI_NOx  * delta_mass   
                if accf_CH4>0:
                    accf_CH4=0   
                                
                temp = T    
                q_met = Q
                G_ = G (P[k])
                T_cr = T_crit (G_)
                elt_t_amb = e_sat_liquid (temp)
                elt_t_cr = e_sat_liquid (T_cr)
                r_sac = r_cont (G_, temp, T_cr, elt_t_cr, elt_t_amb)
                r_hm = rlw (q_met, P[i], elt_t_amb)
                
                r_i =  R
                
                if r_i > confg['r_thr'] and temp < confg['t_thr']:
                    pcfa = 1
                else: 
                    pcfa = 0  
                    
                if r_sac <= r_hm and T_cr >= temp:
                    sac = 1.0
                else:
                    sac = 0.0
                    

                ATR_cont[j,k] = 1e-3 * (confg['daytime'] * aCCF_dCont (OLR) + confg['nighttime'] * aCCF_nCont (temp))  * pcfa #* sac
                ATR_H2O [j,k] = aCCF_H2O (PV) * delta_mass
                ATR_NOx [j,k] = accf_O3 + accf_CH4
                ATR_CO2 [j,k] = 10.1 * 6.94e-16 * delta_mass

            

            ATR_cont[j,1::] = integrate.cumtrapz(ATR_cont[j,:], distance_flown)
            ATR_H2O[j,1::] = integrate.cumtrapz(ATR_H2O[j,:], distance_flown)
            ATR_NOx[j,1::] = integrate.cumtrapz(ATR_NOx[j,:], distance_flown)
            ATR_CO2[j,1::] = integrate.cumtrapz(ATR_CO2[j,:], distance_flown)
            ATR_net[j,1::] =  ATR_NOx[j,1::] + ATR_H2O[j,1::] + ATR_cont[j,1::] + ATR_CO2[j,1::] 
        
        profile['Cont'] = ATR_cont
        profile['H2O'] = ATR_H2O
        profile['NOx'] = ATR_NOx
        profile['CO2'] = ATR_CO2
        profile['ATR_Net'] = ATR_net
        profile['ATR'] = ATR_net[:,-1]
        profile['SOC'] = 0.75 * time + 0.51 * mass
        trj_output[alp] = profile
    return trj_output
