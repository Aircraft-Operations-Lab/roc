#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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