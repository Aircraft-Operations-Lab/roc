# -*- coding: utf-8 -*-

from roc3.occopy.dynamics import *
from roc3.geometry import *
from roc3.apm import *

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

class TrajectoryPredictor(object):
    def __init__(self, weather_model):
        self.wm = weather_model
    def propagate_mass(self, apm, trj, m0, tas):
        N_trj = len([var for var in trj.x if var.startswith('t_')])
        N_wm = self.wm.n_members
        assert N_wm >= N_trj
        N = N_trj
        n = len(trj.x['t_0'])
        for i in range(N):
            trj.x[f'm_{i}'] = [m0]
            for j in range(n - 1):
                m_ij = trj.x[f'm_{i}'][j]
                p1 = (trj.x['lat'][j], 
                      trj.x['lon'][j], 
                      trj.y[f'P_{i}'][j],  
                      trj.x[f't_{i}'][j], 
                     ) 
                p2 = (trj.x['lat'][j+1], trj.x['lon'][j+1], trj.y[f'P_{i}'][j+1],trj.x[f't_{i}'][j+1]) 
                m_new = self.evaluate_transition_Hp_cte(apm, i, p1, p2, tas, m_ij)
                trj.x[f'm_{i}'].append(m_new)
        return trj
                
    def evaluate_transition_Hp_cte(self, apm, scenario_idx, point1, point2, tas, mass):
        i = scenario_idx
        lat1, lon1, P1, t1 = point1
        lat2, lon2, P2, t2 = point2
        H1 = self.wm[i].H(*point1[:4])
        H2 = self.wm[i].H(*point2[:4])
        h1 = H2h(H1)
        h2 =  H2h(H2)
        P = P1 # P1 = P2 lol
        h_avg = 0.5*(h1 + h2)
        points, theta = generate_gcircle_point_list(point1, point2, angle_resolution_deg=0.001, get_theta=True)
        total_dist = total_distance = (6.371e6 + h_avg)*theta
        n_interp = points.shape[0]
        distances_array = np.linspace(0, total_distance, n_interp)
        Ilat = interp1d(distances_array, points[:,0], kind='cubic')
        Ilon = interp1d(distances_array, points[:,1], kind='cubic')
        It = lambda s: t1 + s/total_dist*(t2 - t1)
        dtds = (t2 - t1)/total_dist
        s_span = [0, total_distance*(1 - 1e-9)]
        def f(s, y):
            m, = y
            t = It(s)
            lat = Ilat(s)
            lon = Ilon(s)
            W = m*g
            p = (lat, lon, P, t)
            env = self.wm[i].env_state(*p)
            acs = AircraftState(tas, 0, 0, m, 0)
            D = apm.D(env, acs)
            fc = apm.fc_from_thrust_P_M_T(D, P, acs.M(env), env.T)
            return -fc*dtds,
        y0 = np.array([mass])
        
        output = solve_ivp(f, s_span, y0, method='LSODA', first_step=total_distance/10)
        assert output.success
        return output.y[0,-1]
                
