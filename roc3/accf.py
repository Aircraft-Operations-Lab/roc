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
    accf_nCont = 13.9 * 0.0151 * (1e-10 * (0.0073 * 10 ** (0.0107 * t) - 1.03)) * 0.42/ 3
    return accf_nCont


def pcfa(r, t, r_thr, t_thr, r_pa, t_pa):
    t_condition = 1/(1+np.exp(-t_pa*(-t+t_thr)))
    r_condition = 1/(1+np.exp(-r_pa*( r-r_thr)))
    pcfa_ = t_condition * r_condition
    return pcfa_