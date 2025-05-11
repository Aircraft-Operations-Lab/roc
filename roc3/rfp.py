#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import scipy.interpolate
import datetime
from casadi import *
from roc3.accf import *
from roc3.occopy.dynamics import *
from roc3.occopy.transcription import *
from roc3.weather import *
from roc3.apm import *
from scipy.integrate import ode, solve_ivp

d2r = pi / 180
r2d = d2r**-1

class FlightPlanningProblem_FreeRouting_SinglePhase(object):
    def __init__(self, apm, weather_model, weather_model_b, dynamical_core, problem_config, hazards={}):
        self.apm = apm
        self.wm = weather_model
        self.wm_b = weather_model
        self.pcfg = {  # Default settings
            'model_mass_evo': False,
            'amplitude' : pi * 0.45, # Course / heading amplitude w.r.t. loxodrome
            'airspeed' : 'variable',
            'altitude': 'constant',
            'payload' : 0,
            'earth_model': 'spherical',
            'marginal_fuel_burn': 0,
            }
        self.pcfg.update(problem_config)
        self.hazards = hazards
        self.lat_max = 90*(1 - 1e-5)
        self.lat_min = -90*(1 - 1e-5)
        self.lon_max = inf
        self.lon_min = -inf
        self.add_bounds(self.wm.bounds)
        self.n_members = self.wm.n_members
        self.n_members_b = self.wm_b.n_members
        self.pcfg['ll_bounds'] = self.ll_bounds
        self.dcore = dynamical_core
    def generate_core(self):
        self.core = self.dcore(self.apm, self.wm, self.pcfg)
    def add_bounds(self, bounds):
        '''
        Restricts problem bounds, convenient when using interpolants
        '''
        eps = 1e-9
        self.lat_min = max(self.lat_min, bounds['lat'][0] + eps)
        self.lat_max = min(self.lat_max, bounds['lat'][1] - eps)
        self.lon_min = max(self.lon_min, bounds['lon'][0] + eps)
        self.lon_max = min(self.lon_max, bounds['lon'][1] - eps)
        self.ll_bounds = {
            'lat_min' : self.lat_min,
            'lat_max' : self.lat_max,
            'lon_min' : self.lon_min,
            'lon_max' : self.lon_max,
        }
        if 'times' in self.wm.bounds:
            self.ll_bounds['t_min'] = bounds['times'][0] + eps
            self.ll_bounds['t_max'] = bounds['times'][1] - eps
        else:
            self.ll_bounds['t_min'] = -inf
            self.ll_bounds['t_max'] = inf
    def build_ocp(self):
        self.generate_core()
        olat, olon = self.pcfg['origin']
        dlat, dlon = self.pcfg['destination']
        avg_lat = 0.5 * (olat + dlat)
        delta_lat = dlat - olat
        delta_lon = dlon - olon
        self.avghdg = avg_heading = np.arctan2(cos(avg_lat * pi / 180) * delta_lon, delta_lat)
        self.dm, self.atr_net, self.atr_std = dm, atr_net, atr_std = self.core.get_dynamical_system()
        ocp = SinglePhaseOCP(self.dm)
        
        # Boundary Conditions
        # ===================
        
        # Location ------------
        ocp.add_bc(ocp.x0.lat, self.pcfg['origin'][0])
        ocp.add_bc(ocp.x0.lon, self.pcfg['origin'][1])
        ocp.add_bc(ocp.xf.lat, self.pcfg['destination'][0])
        ocp.add_bc(ocp.xf.lon, self.pcfg['destination'][1])
        
        # Along-track distance
        ocp.add_bc(ocp.t0, 0)
        ocp.add_bc(ocp.tf, (1, 2*pi*6371e3))

        
        # Time -------------
        latest_arrival_time = dm.add_param('lft')
        earliest_arrival_time = dm.add_param('eft')
        
        if self.pcfg['airspeed'] != 'constant' and self.pcfg['climate_impact'] and self.pcfg['climate_modeling'] == 'mayer' and self.pcfg['climate_std']:
            maximum_climate_impact = dm.add_param('max_atr')
            minimum_climate_impact = dm.add_param('min_atr')
            self.atr_range = maximum_climate_impact - minimum_climate_impact
            ocp.add_bc(self.atr_range, (0, 2))

        arrival_times = self.core.get_arrival_times(ocp)
        self.arrival_range = latest_arrival_time - earliest_arrival_time
        self.core.link_arrival_times(ocp, earliest_arrival_time, latest_arrival_time)
        self.average_departure_time = self.core.get_avg_t0(ocp)
        self.average_arrival_time = self.core.get_avg_tf(ocp)

        try:
            assert 't0' in self.pcfg or 'tf' in self.pcfg
        except AssertionError:
            raise ValueError("No boundary conditions for time specified")
        if 't0' in self.pcfg.keys(): 
            for departure_time in self.core.get_departure_times(ocp):
                ocp.add_bc(departure_time, self.pcfg['t0'])
        if 'tf' in self.pcfg.keys(): # AVERAGE arrival time
            ocp.add_bc(self.average_arrival_time, self.pcfg['tf'])
        if 'arrival_window_size' in self.pcfg.keys():
            ocp.add_bc(self.arrival_range, (0, self.pcfg['arrival_window_size']))
        if 'tf_earliest' in self.pcfg.keys():
            ocp.add_bc(earliest_arrival_time, (self.pcfg['tf_earliest'], inf))
        if 'tf_latest' in self.pcfg.keys():
            ocp.add_bc(latest_arrival_time, (-inf, self.pcfg['tf_latest']))

        #ocp.add_bc(self.average_arrival_time - self.pcfg['t0'], (0,8000)) ####################################??????????????????????????????##################################


        ## Impose initial conditions on the newly defined states including ATR
        if self.pcfg['airspeed'] != 'constant' and self.pcfg['climate_impact'] and self.pcfg['climate_modeling'] == 'mayer' and self.pcfg['climate_std']:
            for i in range(self.n_members):
                ocp.add_bc(ocp.x0[f'atr_{i}'], 0) 
                ocp.add_bc(ocp.xf[f'atr_{i}']  - minimum_climate_impact, (0, inf))
                ocp.add_bc(-ocp.xf[f'atr_{i}'] + maximum_climate_impact, (0, inf))

        if self.pcfg['airspeed'] != 'constant' and self.pcfg['climate_impact'] and self.pcfg['climate_modeling'] == 'mayer' and not self.pcfg['climate_std']:
            ocp.add_bc(ocp.x0['atr'], 0) 

        # Mass -------------
        model_mass = self.pcfg['airspeed'] == 'variable' or self.pcfg['model_mass_evo']
        self.model_mass = model_mass
        if model_mass:
            try:
                assert 'm0' in self.pcfg or 'mf_min' in self.pcfg
            except AssertionError:
                raise ValueError("No boundary conditions for mass specified")
            if 'm0' in self.pcfg.keys(): 
                for departure_mass in self.core.get_departure_mass(ocp):
                    ocp.add_bc(departure_mass, self.pcfg['m0'])
            else:
                masses = self.core.get_departure_mass(ocp)
                for m1, m2 in zip(masses[1:], masses[:-1]):
                    ocp.add_bc(m1 - m2)
            if 'mf_min' in self.pcfg.keys(): # MINIMUM arrival mass
                masses = self.core.get_arrival_mass(ocp)
                for mass_final in masses:
                    ocp.add_bc(mass_final, (self.pcfg['mf_min'], inf))
        # Airspeed
        if model_mass:
            if 'tas0' in self.pcfg:
                airspeeds = self.core.get_initial_airspeeds(ocp)
                for tas in airspeeds:
                    ocp.add_bc(tas, self.pcfg['tas0'])
            if 'tasf' in self.pcfg:
                airspeeds = self.core.get_final_airspeeds(ocp)
                for tas in airspeeds:
                    ocp.add_bc(tas, (self.pcfg['tasf']-10, self.pcfg['tasf'] + 10))
        # Altitude
        if self.pcfg['altitude'] == 'track':
            h0 = self.core.get_initial_altitude(ocp)
            ocp.add_bc(h0, self.pcfg['h0'])
            hf = self.core.get_final_altitude(ocp)
            ocp.add_bc(hf, self.pcfg['hf'])
        self.ocp = ocp

    def build_cost_functional(self, ig=None):
        # Cost functional
        # ===============
        ocp = self.ocp
        lat = self.dm.x['lat']
        lon = self.dm.x['lon']
        if self.pcfg['airspeed'] == 'constant':
            pass
        else:
            ## Consideration of climate impact in the form of Lagrangian
                # averaged climate impact:
            if self.pcfg['climate_impact']: 
                if self.pcfg['climate_modeling'] == 'lagr':
                    if self.pcfg['climate_avg']:
                        avg_ATR = sum(self.atr_net [i] for i in range(self.n_members)) / self.n_members
                        ocp.add_lagrangian(float(self.pcfg['C']) * float(self.pcfg['EI']) * avg_ATR)
                    if self.pcfg['climate_std']:
                        if self.n_members == 1:
                            std_ATR = 0
                        else:    
                            std_ATR = self.atr_std
                        ocp.add_lagrangian(float(self.pcfg['C']) * float(self.pcfg['EI_DP']) * std_ATR)
                elif self.pcfg['climate_modeling'] == 'mayer':
                    if not self.pcfg['climate_std']:
                       ocp.add_mayer (self.pcfg['EI'] * ocp.xf['atr'])
                    elif self.pcfg['climate_std']:
                        avg_atr_mayer = sum(ocp.xf[f'atr_{i}'] for i in range(self.n_members)) / self.n_members
                        ocp.add_mayer (self.pcfg['EI'] * avg_atr_mayer)
                        if self.n_members != 1:
                            ocp.add_mayer (self.pcfg['EI_DP'] * (self.atr_range**2))


        if 'convection' in self.hazards:
            c = self.hazards['convection'].get_c_function()
            times = self.core.get_times(self.dm)
            n = len(times)
            for t in times:
                ocp.add_lagrangian(self.pcfg['CP'] * c(lat, lon, t)/n)
        if self.model_mass:
            ocp.add_mayer(self.pcfg['C_m']*self.core.get_avg_fuel_burn(ocp))
            print (self.pcfg['C_m'])
            mfb = 0
        else:
            mfb = self.pcfg['marginal_fuel_burn'] 
            if not mfb:
                olat, olon = self.pcfg['origin']
                env = self.wm[0].env_state(olat, 
                                           olon, 
                                           Hp2P(h2H(self.pcfg['h0'])),
                                           self.pcfg['t0'],
                                           )
                t_ig = ig.x['t_0']
                dt = t_ig(ig.tf) - t_ig(ig.t0)
                if 'm0' in self.pcfg:
                    acs = AircraftState(self.pcfg['tas0'], 0, 0, self.pcfg['m0'])
                    mfb = self.apm.get_marginal_fuel_burn(env, acs, dt)
                elif 'mf_min'in self.pcfg:
                    acs = AircraftState(self.pcfg['tas0'], 0, 0, self.pcfg['mf_min'])
                    mfb = self.apm.get_marginal_fuel_burn(env, acs, -dt)
        ocp.add_mayer(self.pcfg['DP'] * self.arrival_range)
        ocp.add_mayer((self.pcfg['CI'] + mfb) * 0.75 * (self.average_arrival_time - self.pcfg['t0']))
        if 'tf_extra_cost' in self.pcfg.keys():
            for tf in arrival_times:
                ocp.add_mayer(self.pcfg['extra_cost'](tf))
        if 'tas_slope_regularization' in self.pcfg.keys() and 'tas_slope' in self.dm.u:
            ocp.add_lagrangian(self.pcfg['tas_slope_regularization']*self.dm.u['tas_slope']**2)
        if 'h_slope_regularization' in self.pcfg.keys() and 'h_slope' in self.dm.u:
            ocp.add_lagrangian(self.pcfg['h_slope_regularization']*self.dm.u['h_slope']**2)
    def solve(self, ig, solver_options={}, **options):
        self.build_cost_functional(ig)
        options_default = {
            'n_nodes': 40,
        }
        options_default.update(options)
        options = options_default
        pcfg = self.pcfg
        n_members_ig = 1 + max(int(name[2:]) for name in ig.x if name.startswith('t_'))
        if n_members_ig < self.wm.n_members:
            ig.demux(self.wm.n_members)
        self.tt = tt = TrapezoidalTranscription(options['n_nodes'])
        tt.transcript(self.ocp)
        self.sol = tt.solve(ig, solver_options)
        return self.sol

    def autoplot(self, save=None, latex=False, trj=None):
        N = self.wm.n_members
        model_mass = self.pcfg['airspeed'] == 'variable' or self.pcfg['model_mass_evo']
        if trj is None:
            trj = self.sol
        import numpy as np
        from matplotlib import rc
        if latex:
            rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
            rc('text', usetex=True)
        import matplotlib.pyplot as plt
        if self.pcfg['airspeed'] == 'constant':
            sp = (2,2)
        else:
            sp = (2,3)
        plt.subplot(sp[0], sp[1], 1)
        plt.plot(trj.x['lon'], trj.x['lat'], 'C0')
        plt.title("$ \\phi - \\lambda$")
        plt.subplot(sp[0], sp[1], 2)
        t = [np.array(trj.x['t_' + str(i)]) for i in range(N)]
        avg_t = sum(t) / len(t)
        alpha_ens = 0.2
        for i in range(N):
            plt.plot(np.array(trj.t) * 1e-3, t[i] - avg_t, 'C0', alpha=alpha_ens)
        plt.title("$ t(d)$")
        plt.subplot(sp[0], sp[1], 3)
        if model_mass:
            plt.plot( np.array( trj.t ) * 1e-3, trj.x['tas'], 'C0', lw=2.0 )
        for i in range(N):
            plt.plot(np.array(trj.tu) * 1e-3, trj.u['gs_' + str(i)], 'C2', alpha=alpha_ens)
        plt.title("Groundspeed")
        plt.subplot(sp[0], sp[1], 4)
        plt.plot(np.array(trj.tu) * 1e-3, trj.u['course'], 'C3', lw=2.0)
        for i in range(N):
            plt.plot(np.array(trj.tu) * 1e-3, trj.u['hdg_' + str(i)], 'C1', alpha=alpha_ens)
        plt.title("Heading, course")
        
        if model_mass:
            plt.subplot(sp[0], sp[1], 5 )
            for i in range( N ):
                plt.plot( np.array( trj.t ) * 1e-3, trj.x['m_' + str( i )], 'C0' )
            plt.title( "mass" )
            plt.subplot(sp[0], sp[1], 6 )
            if model_mass:
                for i in range( N ):
                    plt.plot( np.array( trj.tu ) * 1e-3, trj.u['CT_' + str( i )], '.', color='C3' )
                plt.title( "CT" )
            for i in range( N ):
                if model_mass:
                    plt.plot( np.array( trj.tu ) * 1e-3, trj.u['CL_' + str( i )], '.', color='C1', alpha=alpha_ens )
                plt.title( "$C_L$, $C_T$" )
        plt.tight_layout()
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        
class DynamicalCore(object):
    '''
    '''
    def __init__(self, apm, weather_model, problem_config):
        self.pcfg = problem_config
        self.apm = apm
        self.wm = weather_model
        self.n_members = self.wm.n_members
    def get_heading_limits(self):
        olat, olon = self.pcfg['origin']
        dlat, dlon = self.pcfg['destination']
        avg_lat = 0.5 * (olat + dlat)
        delta_lat = dlat - olat
        delta_lon = dlon - olon
        self.avghdg = avg_heading = np.arctan2(cos(avg_lat * pi / 180) * delta_lon, delta_lat)
        amplitude = self.pcfg['amplitude']
        hd_min = avg_heading - amplitude
        hd_max = avg_heading + amplitude
        #print("[ROC] Limiting headings to ({0:.4f}, {1:.4f})".format(hd_min, hd_max))
        return (hd_min, hd_max)
    def get_dynamical_system(self):
        N = self.wm.n_members
        model_mass = (self.pcfg['airspeed'] == 'variable') or \
                     (self.pcfg['model_mass_evo'])
        ll_bounds = self.pcfg['ll_bounds']
        hdg_limits = self.get_heading_limits()
        dm = DynamicalSystem(has_y=True, has_p=True)
        if self.pcfg['altitude'] == 'constant':
            gamma = [0]
            h = self.pcfg['h0']
            Hp = h2H(h) # assuming ISA geopotential
            Pfix = Hp2P(Hp)
            n_gamma = 1
        elif self.pcfg['altitude'] == 'track':
            gamma = []
            h = dm.add_state('h', bounds=(-1e3, 15e3))
            n_gamma = N
            for i in range(N):
                gamma.append(dm.add_control(f'gamma_{i}', bounds=(-0.5, 0.5)))
            Pfix = Hp2P(h2H(h))
            P_eps = 0.01 # Pascals
            Pbounds = (self.wm.bounds['levels'][0]*100 + P_eps, self.wm.bounds['levels'][1]*100- P_eps)
            dm.add_constraint(Pfix, Pbounds, ctr_type='x')
        lat = dm.add_state('lat', bounds=(ll_bounds['lat_min'], ll_bounds['lat_max']))
        lon = dm.add_state('lon', bounds=(ll_bounds['lon_min'], ll_bounds['lon_max']))   
        course = dm.add_control('course', bounds=hdg_limits)
        
        t = []
        gs = []
        hdg = []
        T = []
        P = []
        
        ATR_net = []
        ATR_std = []

        t_bounds = (ll_bounds['t_min'], ll_bounds['t_max'])
        for i in range(N):
            t.append(dm.add_state(f't_{i}', bounds=t_bounds))
            gs.append(dm.add_control(f'gs_{i}', bounds=(0, inf)))
            hdg.append(dm.add_control(f'hdg_{i}', bounds=(hdg_limits)))
            P.append(dm.add_algebraic(f'P_{i}', bounds=(100e2, 1500e2)))
            T.append(self.wm[i].T(lat, lon, P[i], t[i]))
            dm.add_constraint(P[i] - Pfix, ctr_type='x')
        fc = []        
        if self.pcfg['airspeed'] == 'constant':
            tas = self.pcfg['tas0']
        else:
            tas = dm.add_state('tas', bounds=(100,300))
            tas_slope = dm.add_control('tas_slope', bounds=(-1, 1))
            # 100 - 300 m/s are just sanity bounds, proper bounds are 
            # incorporated elsewhere
            m = []
            environments =  []
            CT_min = []
            CT_max = []
            CT = []
            CL = []
            CF = [] 
            CF_idle = []
            fb = []
            ac_states = []
            thrusts = []
            drags = []
            M = []
            fc = []
            atr = []

            for i in range(N):
                m_bounds = (self.apm.OEW + self.pcfg['payload'], self.apm.MTOW) #OEW: Operating empty weight, MTOW: the maximum mass
                m.append(dm.add_state(f'm_{i}', bounds=m_bounds))
                environments.append(EnvironmentState(P[i], T[i]))
                ac_states.append(AircraftState(tas, hdg[i], 0, m[i]))
            z_env_acs = list(zip(environments, ac_states)) 
            CL_max = [self.apm.CL_max(env, acs)      for env, acs in z_env_acs] 
            CT_max = [self.apm.CT_max_MCRZ(env, acs) for env, acs in z_env_acs] # maximum cruise
            CT_min = [self.apm.CT_min(env, acs)      for env, acs in z_env_acs]
            ATR_mean = []
            for i in range(N):
                CT.append(dm.add_control(f'CT_{i}'))
                CL.append(dm.add_control(f'CL_{i}'))
                drags.append(self.apm.D(environments[i], ac_states[i], CL[i]))
                thrusts.append(self.apm.T(environments[i], CT[i]))
                M.append(ac_states[i].M(environments[i]))
                #CF.append(self.apm.CF_from_CT_M(CT[i], M[i]))
                CF.append(self.apm.CF(environments[i], ac_states[i], CT[i]))
                fc.append(self.apm.fc(environments[i], CF[i]))
                CF_idle.append(self.apm.CF_idle(environments[i], ac_states[i]))
                #fc.append(self.apm.fc_from_thrust_P_M_T(thrusts[i], P, M[i], environments[i].T))
                
                if self.pcfg['climate_impact']:
                    index_member = 0
                    # Meteorological Data
                    t_met    = self.wm[index_member].T   (lat, lon, P[i], t[i])
                    olr_met  = self.wm[index_member].olr (lat, lon, P[i], t[i])

                    # algorithmic climate change functions (aCCF)
                    aCCF_NOx = self.wm[index_member].aCCF_NOx(lat, lon, P[i], t[i])
                    aCCF_H2O = self.wm[index_member].aCCF_H2O(lat, lon, P[i], t[i])

                    contrail = 0
                    count = 0
                    for k_member in range(1,11):
                        count += 1
                        attr_name = f"r{k_member}"
                        r_met = getattr(self.wm[0], attr_name)(lat, lon, P[i], t[i]) * 100 
                        # Contrails' climate impact (aCCF * persitent contrail formation areas (PCFA))                    
                        pcfa_ = pcfa (r_met, t_met, self.pcfg['r_thr'], self.pcfg['t_thr'], self.pcfg['r_pa'], self.pcfg['t_pa'])
                        
                        # if 1>0:#self.pcfg['sac']:
                        #     q_met = getattr(self.wm[0], attr_name)(lat, lon, P[i], t[i]) * 100 
                        #     G_ = G (P[i])
                        #     T_cr = T_crit (G_)
                        #     elt_t_amb = e_sat_liquid (t_met)
                        #     elt_t_cr = e_sat_liquid (T_cr)
                        #     r_sac = r_cont (G_, t_met, T_cr, elt_t_cr, elt_t_amb)
                        #     r_hm = rlw (q_met, P[i], elt_t_amb)
                        #     r_i = rhi(q_met, t_met, P[i])
                        #     pcfa_ = pcfa (r_i, t_met, self.pcfg['r_thr'], self.pcfg['t_thr'], self.pcfg['r_pa'], self.pcfg['t_pa'])
                        #     sac_ = sac (r_sac, r_hm, t_met, T_cr)
                            
                        # else:
                        r_met = getattr(self.wm[0], attr_name)(lat, lon, P[i], t[i]) * 100 
                        pcfa_ = pcfa (r_met, t_met, self.pcfg['r_thr'], self.pcfg['t_thr'], self.pcfg['r_pa'], self.pcfg['t_pa'])
                        sac_  = 1
                                            
                        aCCF_nContrail = aCCF_nCont (t_met)   * pcfa_ * sac_
                        aCCF_dContrail = aCCF_dCont (olr_met) * pcfa_ * sac_
                        aCCF_Contrail = self.pcfg['daytime'] * aCCF_dContrail + self.pcfg['nighttime'] * aCCF_nContrail
                        contrail+=aCCF_Contrail
                    contrail = contrail/11    
                    print(count)
                        

                    # Fuel flow per distance flown
                    fuelf_dist = fc[i]/gs[i]

                    # Net average temperature response (F-ATR20): CO2 + non-CO2
                    ATR_NET = self.pcfg['EI_contrail'] * 1e-3 * contrail + fuelf_dist * (0.012 * self.pcfg['EI_NOx'] * aCCF_NOx + self.pcfg['EI_H2O'] * aCCF_H2O + self.pcfg['EI_CO2'] * 10.1 * 6.94e-16)
                    ATR_net.append (ATR_NET)

                    if self.pcfg['climate_impact'] and self.pcfg['climate_modeling'] == 'lagr' and self.pcfg['climate_std']:
                        ATR_mean.append(self.pcfg['EI_contrail'] * 1e-3 * contrail)
        
            if self.pcfg['climate_impact'] and self.pcfg['climate_modeling'] == 'lagr' and self.pcfg['climate_std']:
                ATR_mean_ =  sum(ATR_mean [k] for k in range(self.n_members)) / self.n_members
                ATR_std   =  sum((ATR_mean [k] - ATR_mean_)**2 for k in range(self.n_members)) / self.n_members

        # Dynamics & path constraints
        if self.pcfg['earth_model'] == 'spherical':
            RM = R_mean
            RN = R_mean
        elif self.pcfg['earth_model'] == 'ellipsoidal':
            RM = R_M(lat)
            RN = R_N(lat)
        else:
            em = self.pcfg['earth_model']
            raise ValueError(f"Earth model '{em}' not recognized")
        lat.f = cos(course) / (RM + h) / d2r
        lon.f = sin(course) / (RN + h) / cos(lat * d2r) / d2r
        ll = vertcat(lat, lon)
        if model_mass:
            tas.f = tas_slope
        if self.pcfg['altitude'] == 'track':
            h_slope = dm.add_control('h_slope', bounds=(-.5, .5))
            # h_slope = dm.add_control('h_slope', bounds=(-.03, .03))
            h.f = h_slope
        atr_bounds = (-inf,inf)    
        for i in range(N):
            g_i = gamma[i % n_gamma]
            t[i].f = 1 / gs[i]
            ui = self.wm[i].u(lat, lon, P[i], t[i])
            vi = self.wm[i].v(lat, lon, P[i], t[i])
            dm.add_constraint(cos(course) * gs[i] - (tas * cos(hdg[i])*cos(g_i) + vi))
            dm.add_constraint(sin(course) * gs[i] - (tas * sin(hdg[i])*cos(g_i) + ui))
            if model_mass:
                m[i].f = -fc[i]/gs[i]
                dm.add_constraint(CL[i]*ac_states[i].q(environments[i])*self.apm.S - m[i]*g*cos(g_i))
                acc = ((thrusts[i] - drags[i])/m[i] -g*sin(g_i))
                dm. add_constraint(tas_slope*gs[i] - acc)
                dm.add_constraint(self.apm.M_MO - M[i], (0, inf), ctr_type='x')
                dm.add_constraint(CL_max[i] - CL[i], (0, inf))
                dm.add_constraint(CL[i] - 0, (0, inf))
                dm.add_constraint(CT_max[i] - CT[i], (0, inf))
                dm.add_constraint(CT[i] - CT_min[i], (0, inf))
                dm.add_constraint(CF[i] - CF_idle[i], (0, inf))
                dm.add_constraint(fc[i], (0, inf))
                dm.add_constraint(CT[i], (0, inf))

                # Definition of ATR as new state. Here two cases are considered: 1) when we want to penalize disperssion.
                # For this case, it is better to have N_eps new states. 2) we only interested in averaged performance. 
                # For this case, for computational purposes, we only define one variable. Todo: good ig for such defined states. 
                if self.pcfg['airspeed'] != 'constant' and self.pcfg['climate_impact'] and self.pcfg['climate_modeling'] == 'mayer' and self.pcfg['climate_std']:
                    atr.append(dm.add_state(f'atr_{i}', bounds = atr_bounds))
                    atr[i].f  = 1e10 * ATR_net [i]
                if i == 0 and self.pcfg['airspeed'] != 'constant' and self.pcfg['climate_impact'] and self.pcfg['climate_modeling'] == 'mayer' and not self.pcfg['climate_std']:
                    atr = dm.add_state('atr', bounds = atr_bounds)
                    atr.f = 1e10 * sum(ATR_net [k] for k in range(self.n_members)) / self.n_members
                if self.pcfg['altitude'] == 'track':
                    dm.add_constraint(h_slope*gs[i] - tas*sin(g_i))
        return dm, ATR_net, ATR_std
    def get_avg_t0(self, ocp):
        N = self.wm.n_members
        return sum(ocp.x0[f't_{i}'] for i in range(N))/N
    def get_times(self, dm):
        N = self.wm.n_members
        return [dm.x[f't_{i}'] for i in range(N)]
    def get_avg_tf(self, ocp):
        N = self.wm.n_members
        return sum(ocp.xf[f't_{i}'] for i in range(N))/N
    def get_initial_altitude(self, ocp):
        return ocp.x0['h']
    def get_final_altitude(self, ocp):
        return ocp.xf['h']
    def get_departure_times(self, ocp):
        N = self.wm.n_members
        return [ocp.x0[f't_{i}'] for i in range(N)]
    def get_arrival_times(self, ocp):
        N = self.wm.n_members
        return [ocp.xf[f't_{i}'] for i in range(N)]
    def get_departure_mass(self, ocp):
        N = self.wm.n_members
        return [ocp.x0[f'm_{i}'] for i in range(N)]
    def get_arrival_mass(self, ocp):
        N = self.wm.n_members
        return [ocp.xf[f'm_{i}'] for i in range(N)]
    def get_initial_airspeeds(self, ocp):
        return (ocp.x0['tas'],)
    def get_final_airspeeds(self, ocp):
        return (ocp.xf['tas'],)
    def link_arrival_times(self, ocp, earliest_arrival_time, latest_arrival_time):
        for i in range(self.n_members):
            ocp.add_bc(ocp.xf[f't_{i}'] - earliest_arrival_time, (0, inf))
            ocp.add_bc(-ocp.xf[f't_{i}'] + latest_arrival_time, (0, inf))
    def get_avg_fuel_burn(self, ocp):
        dep_masses = self.get_departure_mass(ocp)
        arr_masses = self.get_arrival_mass(ocp)
        return (sum(dep_masses) - sum(arr_masses) )/len(dep_masses)
    
    
