# -*- coding: utf-8 -*-

import numpy as np
from roc3.apm import *
from roc3.bada4 import BADA4_jet_CR
from roc3.geometry import *
from roc3.rfp import FlightPlanningProblem_FreeRouting_SinglePhase as FPP
from roc3.occopy.dynamics import DiscreteTrajectory
from scipy.interpolate import interp1d as I
from scipy.optimize import brentq

def latlon2xyz(lat, lon):
    return np.array([
        np.sin(lon)*np.cos(lat),
        -np.cos(lat)*np.cos(lon),
        np.sin(lat),
        ])
def xyz2latlon(xyz):
    return np.arcsin(xyz[2]), np.arctan2(xyz[0], -xyz[1])

class LinearAltitudeAirspeedProfile(object):
    def __init__(self, tas_0=160, tas_10k=225):
        self.tas_0 = tas_0
        self.tas_10k = tas_10k
    def v(self, h: float):
        return self.tas_0 + (self.tas_10k - self.tas_0)*h/10e3
    def dvdh(self):
        return (self.tas_10k - self.tas_0)/10e3

class ControlLaws(object):
    def __init__(self, apm: BADA4_jet_CR, h_cruise=10e3, profiles=None):
        if profiles is None:
            profiles = {}
        self.apm = apm
        self.profiles = {'climb': LinearAltitudeAirspeedProfile(),
                         'descent': LinearAltitudeAirspeedProfile(), }
        self.profiles.update(profiles)
        self.h_cruise = h_cruise
    def get_climb_controls(self, env: EnvironmentState, acs: AircraftState, dvdh: float=None) -> dict:
        if dvdh is None:
            dvdh = self.profiles['climb'].dvdh()
        CT = 1*self.apm.CT_max_MCRZ(env, acs)
        CL_factor = 2*acs.m*g/(env.rho*acs.TAS**2*self.apm.S)
        def slope(gamma: float) -> float:
            h_dot = acs.TAS*np.sin(gamma)
            CL = CL_factor*np.cos(gamma)
            D = self.apm.D(env, acs, CL)
            T = self.apm.T(env, CT)
            v_dot = (T - D)/acs.m - g*np.sin(gamma)
            # print(f"{gamma} -> vdot: {v_dot}, hdot: {h_dot}")
            return v_dot/h_dot
        slope_diff = lambda gamma: slope(gamma) - dvdh
        output = brentq(slope_diff, 1e-6, 0.5, full_output=True)
        assert output[1].converged
        gamma = output[0]
        CL = CL_factor*np.cos(gamma)
        return {'CT': CT, 'gamma': gamma, 'CL':CL}
    def get_descent_controls(self, env: EnvironmentState, acs: AircraftState, dvdh: float=None) -> dict:
        if dvdh is None:
            dvdh = self.profiles['climb'].dvdh()
        CT = 1.05*self.apm.CT_min(env, acs)
        CL_factor = 2*acs.m*g/(env.rho*acs.TAS**2*self.apm.S)
        def slope(gamma: float) -> float:
            h_dot = acs.TAS*np.sin(gamma)
            CL = CL_factor*np.cos(gamma)
            D = self.apm.D(env, acs, CL)
            T = self.apm.T(env, CT)
            v_dot = (T - D)/acs.m - g*np.sin(gamma)
            return v_dot/h_dot

        slope_diff = lambda gamma: slope(gamma) - dvdh
        output = brentq(slope_diff, -0.5, -1e-6, full_output=True)
        assert output[1].converged
        gamma = output[0]
        CL = CL_factor*np.cos(gamma)
        return {'CT': CT, 'gamma': gamma, 'CL':CL}
    def get_cruise_controls(self, env: EnvironmentState, acs: AircraftState) -> dict:
        CL = self.apm.CL(env, acs)
        D = self.apm.D(env, acs, CL)
        CT = self.apm.CT(env, D)
        return {'CT': CT, 'CL': CL}

def patch_mass_and_altitude(fpp: FPP, trj, m0, n_nodes=8000):
    """
    Uses Euler integration to generate the mass and altitude profile
    """
    apm = fpp.apm
    cl = ControlLaws(apm)
    ig = DiscreteTrajectory()
    eps = 1e-8*(trj.tf - trj.t0)
    ig.t0 = trj.t0 + eps
    ig.tf = trj.tf - eps
    ig.x['lat'] = trj.x['lat']
    ig.x['lon'] = trj.x['lon']
    ig.u['course'] = trj.u['course']
    if 'tasf' in fpp.pcfg:
        tasf = fpp.pcfg['tasf']
    else:
        tasf = 150
    initial_conds = {
        'h': fpp.pcfg['h0'],
        'tas': fpp.pcfg['tas0'],
        't': trj.x['t_0'](trj.t0),
        's': trj.t0,
        'hdg': trj.u['hdg_0'](trj.t0),
    }
    final_conds = {
        'h': fpp.pcfg['hf'],
        'tas': tasf,
        't': trj.x['t_0'](trj.tf),
        's': trj.tf,
        'hdg': trj.u['hdg_0'](trj.tf),
    }
    if 'm0' in fpp.pcfg:
        initial_conds['m'] = fpp.pcfg['m0']
        mode = 'forward'
    else:
        initial_conds['m'] = m0
    if 'mf_min' in fpp.pcfg:
        final_conds['m'] = fpp.pcfg['mf_min']
        mode = 'reverse'
    else:
        final_conds['m'] = fpp.apm.OEW*1.1
    assert mode in {'reverse', 'forward'}
    lat = np.zeros(n_nodes)
    lon = np.zeros(n_nodes)
    v = np.zeros(n_nodes)
    gs = np.zeros(n_nodes)
    m = np.zeros(n_nodes)
    t = np.zeros(n_nodes)
    h = np.zeros(n_nodes)
    gamma = np.zeros(n_nodes)
    h_slope = np.zeros(n_nodes)
    CL = np.zeros(n_nodes)
    CT = np.zeros(n_nodes)
    hdg = np.zeros(n_nodes)
    crs = np.zeros(n_nodes)
    s = np.zeros(n_nodes)
    P = np.zeros(n_nodes)
    gamma = np.zeros(n_nodes)
    tas_slopes = np.zeros(n_nodes)
    atr = np.zeros(n_nodes)

    m[0] = initial_conds['m']
    t[0] = initial_conds['t']
    h[0] = initial_conds['h']
    s[0] = initial_conds['s']
    v[0] = initial_conds['tas']

    m[-1] = final_conds['m']
    t[-1] = final_conds['t']
    h[-1] = final_conds['h']
    s[-1] = final_conds['s']
    v[-1] = final_conds['tas']

    s_step = (trj.tf - trj.t0)/(n_nodes - 1)*(1 - 1e-9)
    if v[0] > cl.profiles['climb'].v(h[0]):
        dvdh_list_climb = (0, None)
        switch_cond_climb = lambda v_, h_: v_ < cl.profiles['climb'].v(h_)
    else:
        dvdh_list_climb = (cl.profiles['climb'].dvdh()*3, None) # 1 is actually high enough
        switch_cond_climb = lambda v_, h_: v_ > cl.profiles['climb'].v(h_)

    if v[-1] > cl.profiles['descent'].v(h[-1]):
        dvdh_list_desc = (0, None)
        switch_cond_desc = lambda v_, h_: v_ < cl.profiles['descent'].v(h_)
    else:
        dvdh_list_desc = (cl.profiles['descent'].dvdh()*3, None) # 1 is actually high enough
        switch_cond_desc = lambda v_, h_: v_ > cl.profiles['descent'].v(h_)

    for i in range(int(n_nodes // 2)):
        acs = AircraftState(v[i], hdg[i], 0, m[i])
        lat[i] = trj.x['lat'](s[i])
        lon[i] = trj.x['lon'](s[i])
        crs[i] = trj.u['course'](s[i])
        P[i] = Hp2P(h2H(h[i]))
        env = fpp.wm[0].env_state(lat[i], lon[i], P[i], t[i])
        if h[i] < cl.h_cruise:
            dvdh = dvdh_list_climb[1 if switch_cond_climb(v[i], h[i]) else 0]
            controls = cl.get_climb_controls(env, acs, dvdh)
            CL[i] = controls['CL']
            CT[i] = controls['CT']
            gamma[i] = controls['gamma']
            acs.gamma = gamma[i]
        else: # Cruise
            controls = cl.get_cruise_controls(env, acs)
            CL[i] = controls['CL']
            CT[i] = controls['CT']
            gamma[i] = 0
        gs[i], hdg[i] = groundspeed_and_heading(v[i], crs[i], gamma[i], env.wind)
        acs.heading = hdg[i]
        if i == n_nodes - 1:
            break
        # Differential equations
        s[i+1] = s[i] + s_step
        t[i+1] = t[i] + s_step/gs[i]
        delta_t = t[i+1] - t[i]
        vc = VerticalControls(CT[i], CL[i])
        v[i+1] = v[i] + delta_t*apm.v_dot(env, acs, vc)
        h[i+1] = h[i] + delta_t*v[i]*np.sin(gamma[i])
        m[i+1] = m[i] + delta_t*apm.m_dot(env, acs, vc)
        tas_slopes[i] = (v[i+1] - v[i])/s_step
        h_slope[i] = (h[i+1] - h[i])/s_step

    for i in range(-1, -int(n_nodes//2)-1, -1):
        acs = AircraftState(v[i], hdg[i], 0, m[i])
        lat[i] = trj.x['lat'](s[i])
        lon[i] = trj.x['lon'](s[i])
        crs[i] = trj.u['course'](s[i])
        P[i] = Hp2P(h2H(h[i]))
        env = fpp.wm[0].env_state(lat[i], lon[i], P[i], t[i])
        if h[i] < cl.h_cruise:
            dvdh = dvdh_list_desc[1 if switch_cond_desc(v[i], h[i]) else 0]
            controls = cl.get_descent_controls(env, acs, dvdh)
            CL[i] = controls['CL']
            CT[i] = controls['CT']
            gamma[i] = controls['gamma']
            acs.gamma = gamma[i]
        else:
            controls = cl.get_cruise_controls(env, acs)
            CL[i] = controls['CL']
            CT[i] = controls['CT']
            gamma[i] = 0
        gs[i], hdg[i] = groundspeed_and_heading(v[i], crs[i], gamma[i], env.wind)
        acs.heading = hdg[i]
        if i == n_nodes - 1:
            break
        # Differential equations
        s[i-1] = s[i] - s_step
        t[i-1] = t[i] - s_step/gs[i]
        delta_t = t[i] - t[i-1]
        vc = VerticalControls(CT[i], CL[i])
        v[i-1] = v[i] - delta_t*apm.v_dot(env, acs, vc)
        h[i-1] = h[i] - delta_t*v[i]*np.sin(gamma[i])
        m[i-1] = m[i] - delta_t*apm.m_dot(env, acs, vc)
        tas_slopes[i] = (v[i] - v[i-1])/s_step
        h_slope[i] = (h[i] - h[i-1])/s_step

    i_gap = int(n_nodes//2) - 1
    m_gap = m[i_gap+1] - m[i_gap] - (m[i_gap] - m[i_gap-1])

    if mode == 'forward':
        m[int(n_nodes//2) - 1:] += m_gap
    else:
        m[:int(n_nodes // 2)] -= m_gap

    # Make headings / courses continuous
    for i in range(n_nodes - 1):
        while hdg[i+1] - hdg[i] > np.pi:
            hdg[i+1:] -= 2*np.pi
        while hdg[i+1] - hdg[i] < -np.pi:
            hdg[i+1:] += 2*np.pi
    for i in range(n_nodes - 1):
        while crs[i+1] - crs[i] > np.pi:
            crs[i+1:] -= 2*np.pi
        while crs[i+1] - crs[i] < -np.pi:
            crs[i+1:] += 2*np.pi
    ig.x['h'] = I(s, h) 
    ig.u['h_slope'] = I(s, h_slope) 
    ig.u['gamma_0'] = I(s, gamma)   
    ig.x['tas'] = I(s, v)
    ig.x['t_0'] = I(s, t)
    ig.x['m_0'] = I(s, m)
    if fpp.pcfg['airspeed'] != 'constant' and fpp.pcfg['climate_impact'] and fpp.pcfg['climate_modeling'] == 'mayer' and fpp.pcfg['climate_std']:
        ig.x['atr_0'] = I(s, atr)
        ig.p['min_atr'] = 0
        ig.p['max_atr'] = 0
    if fpp.pcfg['airspeed'] != 'constant' and fpp.pcfg['climate_impact'] and fpp.pcfg['climate_modeling'] == 'mayer' and not fpp.pcfg['climate_std']: 
        ig.x['atr'] = I(s, atr)  
    ig.p['lft'] = t[-1]
    ig.p['eft'] = t[-1]
    ig.u['tas_slope'] = I(s, tas_slopes)
    ig.u['course'] = I(s, crs)
    ig.u['gs_0'] = I(s, gs)
    ig.u['hdg_0'] = I(s, hdg)
    ig.u['CL_0'] = I(s, CL)
    ig.u['CT_0'] = I(s, CT)
    ig.y['P_0'] = I(s, P)
    return ig


def patch_mass(fpp, trj):
    """
    Uses Euler integration to generate the mass profile
    """
    if 'tas' not in trj.x:
        tas = lambda s: fpp.pcfg['tas0']
    else:
        tas = lambda s: trj.x['tas'](s)
    if 'm0' in fpp.pcfg:
        m0 = fpp.pcfg['m0']
        mode = 'forward'
    else:
        mf = fpp.pcfg['mf_min']
        mode = 'reverse'
    s = trj.t
    n = len(s)
    m_nodes = [0]*n
    
    CL_nodes = []
    CT_nodes = []
    if mode == 'forward':
        m_nodes[0] = m0
        for i in range(n-1):
            def deriv(x, i=i):
                si = s[i]
                si1 = s[i+1]
                return (x(si1) - x(si))/(si1 - si)
            si = s[i]
            P = trj.y['P_0'](si)
            t = trj.x['t_0'](si)
            lat = trj.x['lat'](si)
            lon = trj.x['lon'](si)
            gs = trj.u['gs_0'](si)
            mi = m_nodes[i]
            T = fpp.wm[0].T(lat, lon, P, t)
            env = EnvironmentState(P, T)
            acs = AircraftState(tas(si), 0, 0, mi)
            M = acs.M(env)
            tas_dot = deriv(tas)*gs
            CL = mi*g/acs.q(env)/fpp.apm.S
            drag = fpp.apm.D(env, acs, CL)
            thrust = drag + mi*tas_dot
            CT = fpp.apm.CT(env, thrust)
            CF = fpp.apm.CF_from_CT_M(CT, M)
            fc = fpp.apm.fc(env, CF)
            m_nodes[i+1] = mi - fc*(trj.x['t_0'](s[i+1]) - t)
            CL_nodes.append(CL)
            CT_nodes.append(CT)
        CL_nodes += [CL_nodes[-1]]
        CT_nodes += [CT_nodes[-1]]
    elif mode == 'reverse':
        m_nodes[-1] = mf
        for i in range(n-1, 0, -1):
            def deriv(x, i=i):
                si = s[i]
                si1 = s[i-1]
                return (x(si1) - x(si))/(si1 - si)
            si = s[i]
            P = trj.y['P_0'](si)
            t = trj.x['t_0'](si)
            lat = trj.x['lat'](si)
            lon = trj.x['lon'](si)
            gs = trj.u['gs_0'](si)
            mi = m_nodes[i]
            T = fpp.wm[0].T(lat, lon, P, t)
            env = EnvironmentState(P, T)
            acs = AircraftState(tas(si), 0, 0, mi)
            M = acs.M(env)
            tas_dot = deriv(tas)*gs
            CL = mi*g/acs.q(env)/fpp.apm.S
            drag = fpp.apm.D(env, acs, CL)
            thrust = drag + mi*tas_dot
            CT = fpp.apm.CT(env, thrust)
            CF = fpp.apm.CF_from_CT_M(CT, M)
            fc = fpp.apm.fc(env, CF)
            assert not np.isnan(float(fc))
            assert fc > 0
            m_nodes[i-1] = mi - fc*(trj.x['t_0'](s[i-1]) - t)
            CL_nodes.append(CL)
            CT_nodes.append(CT)
        CL_nodes.reverse()
        CT_nodes.reverse()
        CL_nodes += [CL_nodes[-1]]
        CT_nodes += [CT_nodes[-1]]

    trj.x['m_0'] = I(s, m_nodes)
    trj.x['tas'] = tas
    trj.u['CL_0'] = I(s, np.array(CL_nodes)[:,0,0])
    trj.u['CT_0'] = I(s, np.array(CT_nodes)[:,0,0])
    trj.u['tas_slope'] = lambda s: 0
    return trj

def ig_2D_ortho(fpp, num_points=10):
    """
    fpp: FlightPlanningProblem-like 
    num_points: int
    """
    isa = ISA()
    n_points = num_points
    from numpy.linalg import norm
    

    def orthogonal_rodrigues_rotation(v, k, angle):
        return v*np.cos(angle) + np.cross(k, v)*np.sin(angle)
    def rq2ll(r, q):
        pr = orthogonal_rodrigues_rotation(p_mid, ax_r, 0.5*theta*r)
        return xyz2latlon(orthogonal_rodrigues_rotation(pr, ax_q, theta*0.25*q))
    R = 6.371e6
    lat0 = fpp.pcfg['origin'][0]*np.pi/180
    lon0 = fpp.pcfg['origin'][1]*np.pi/180
    latf = fpp.pcfg['destination'][0]*np.pi/180
    lonf = fpp.pcfg['destination'][1]*np.pi/180
    p0 = latlon2xyz(lat0, lon0)
    pf = latlon2xyz(latf, lonf)
    p_mid = 0.5*(p0 + pf)
    p_mid /= norm(pf)
    ax_r = np.cross(p0, pf - p0)
    ax_r /= norm(ax_r)
    ax_q = pf - p0
    ax_q /= norm(ax_q)
    theta = 2*np.arcsin(norm(pf - p0)/2)
    r_range = np.linspace(-1, 1, num_points)
    points = [rq2ll(r, 0) for r in r_range]
    dist_nodes_approx = R*theta/(n_points - 1)
    times = np.zeros(n_points)
    if 't0' in fpp.pcfg:
        times[0] = fpp.pcfg['t0']
    courses = np.zeros_like(times)
    courses0 = np.zeros_like(times)
    headings = np.zeros_like(times)
    groundspeeds = np.zeros_like(times)
    VTas=np.zeros_like(times)
    masses=np.zeros_like(times)
    accelerations = np.zeros_like(times)
    CLs = np.zeros_like(times)
    CTs = np.zeros_like(times)
    for i in range(num_points - 1):
        DeltaLat = points[i+1][0] - points[i][0]
        DeltaLon = points[i+1][1] - points[i][1]
        midlat = 0.5*(points[i+1][0] + points[i][0])
        crs = np.arctan2(DeltaLon/np.cos(midlat), DeltaLat)
        courses0[i] = crs
    courses0[0] = courses0[1]
    def adjust_to_0_2pi(angle):
        while angle <= 0:
            angle += 2*pi
        while angle >= 2*pi:
            angle -= 2*pi
        return angle
    def avg_crs(course_1, course_2):
        while course_1 >= course_2 + np.pi:
            course_1 -= 2*np.pi
        while course_1 <= course_1 - np.pi:
            course_1 += 2*np.pi
        res = 0.5*(course_1 + course_2)
        return adjust_to_0_2pi(res)
    for i in range(num_points - 2):
        courses[i+1] = avg_crs(courses0[i], courses0[i+1])

    courses[0] = adjust_to_0_2pi(courses[1] + 2*(courses0[0] - courses[1]))
    courses[-1] = adjust_to_0_2pi(courses[-2] + 2*(courses0[-2] - courses[-2]))

    tas = fpp.pcfg['tas0']
    if 'm0' in fpp.pcfg:
        m0 = fpp.pcfg['m0']
    else:
        m0 = 0.95*fpp.apm.MTOW
    
    Hp = h2H(fpp.pcfg['h0'])
    rho = isa.get_environment_state(AircraftPosition(0, 0, Hp, 0)).rho
    P0 = isa.get_environment_state(AircraftPosition(0, 0, Hp, 0)).P
    

    for i in range(num_points):
        VTas[i]=tas #constant tas
        masses[i]= m0 #constant mass
        accelerations[i] = 0
        CLs[i] = m0*g/.5*fpp.apm.S*rho*tas**2.0
        CTs[i] = 0.15
    for i in range(num_points):
        p = points[i]
        p_deg = np.array(p)*180/np.pi
        wind_u = fpp.wm[0].u(p_deg[0], p_deg[1], 0, 0)
        wind_v = fpp.wm[0].v(p_deg[0], p_deg[1], 0, 0)
        crs = courses[i]

        w_at = np.sin(crs)*wind_u + np.cos(crs)*wind_v
        w_ct = np.cos(crs)*wind_u - np.sin(crs)*wind_v
        v_proj = (tas**2 - w_ct**2)**0.5
        groundspeeds[i] = v_proj + w_at
        headings[i] = adjust_to_0_2pi(courses[i] + np.arctan2(w_ct, v_proj))
        
    # Make headings / courses continuous
    for i in range(num_points - 1):
        while headings[i+1] - headings[i] > np.pi:
            headings[i+1:] -= 2*np.pi
        while headings[i+1] - headings[i] < -np.pi:
            headings[i+1:] += 2*np.pi
        
    for i in range(num_points - 1):
        while courses[i+1] - courses[i] > np.pi:
            courses[i+1:] -= 2*np.pi
        while courses[i+1] - courses[i] < -np.pi:
            courses[i+1:] += 2*np.pi
        
    for i in range(num_points-1):
        times[i+1] = times[i] + dist_nodes_approx/groundspeeds[i]**0.5/groundspeeds[i+1]**0.5
    ig = DiscreteTrajectory()
    ig.t0 = 0 + 1e-6
    ig.tf = R*theta -1e-6
    ig.t = list(np.linspace(0, R*theta, num_points))
    ig.tu = ig.t
    distances = dist_nodes_approx*(np.cumsum(np.ones_like(times))-1)

    ig.x['lat'] = I(distances, [p[0]*180/np.pi for p in points])
    ig.x['lon'] = I(distances, [p[1]*180/np.pi for p in points])
    ig.x['tas'] = I(distances, VTas)
    ig.x['t_0'] = I(distances, times)
    ig.x['m_0'] = I(distances, masses)
    ig.p['lft'] = times[-1]
    ig.p['eft'] = times[-1]
    ig.u['tas_slope'] = I(distances, accelerations)
    ig.u['course'] = I(distances, courses)
    ig.u['gs_0'] = I(distances, groundspeeds)
    ig.u['hdg_0'] = I(distances, headings)
    ig.u['CL_0'] = I(distances, CLs)
    ig.u['CT_0'] = I(distances, CTs)
    ig.y['P_0'] = lambda *foo: P0
    
    # Fix mass profile (approx)
    
    def CL(t):
        lat = ig.x['lat'](t)
        lon = ig.x['lon'](t)
        hdg = ig.u['hdg_0'](t)
        v = ig.x['tas'](t)
        m = ig.x['m_0'](t)
        z = fpp.pcfg['h0']
        position = AircraftPosition(lat, lon, z, t)
        environment = isa.get_environment_state(position)
        ac_state = AircraftState(v, hdg, 0, m, 0)
        return fpp.apm.CL(environment, ac_state)

    def CT(t):
        lat = ig.x['lat'](t)
        lon = ig.x['lon'](t)
        hdg = ig.u['hdg_0'](t)
        v = ig.x['tas'](t)
        m = ig.x['m_0'](t)
        z = fpp.pcfg['h0']
        position = AircraftPosition(lat, lon, z, t)
        environment = isa.get_environment_state(position)
        ac_state = AircraftState(v, hdg, 0, m, 0)
        drag = fpp.apm.D(environment, ac_state)
        thrust = drag
        return fpp.apm.CT(environment, thrust)

    t0 = fpp.pcfg['t0']
    lat = ig.x['lat'](0)
    lon = ig.x['lon'](0)
    hdg = ig.u['hdg_0'](0)
    v = ig.x['tas'](0)
    z = fpp.pcfg['h0']
    position = AircraftPosition(lat, lon, z, t0)
    environment = isa.get_environment_state(position)
    ac_state = AircraftState(v, hdg, 0, m0, mu=0)
    drag = fpp.apm.D(environment, ac_state)
    thrust = drag
    CT0 = fpp.apm.CT(environment, thrust)
    CF = fpp.apm.CF(environment, ac_state, CT0)
    fc = fpp.apm.fc(environment, CF)
    ig.x['m_0'] = lambda d: fpp.pcfg['m0'] - fc*(d/v - t0)

    ig.u['CL_0'] = lambda d: CL(d/v)
    ig.u['CT_0'] = lambda d: CT(d/v)
    
    return ig

def fix_mass_profile(ig):
    """
    Gets amn
    """

