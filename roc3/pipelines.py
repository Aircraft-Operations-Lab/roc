# -*- coding: utf-8 -*-

import numpy as np

from .weather import WeatherModel
from .rfp import *
from .initial_guess import *


class Routing2DStandard(object):
    """
    Standard initialization pipeline for 2D routing problems (constant airspeed)
    Performs the following steps:
    1. Generate 2D initial guess from orthodromic path and ens member #0
    2. Solve the deterministic problem with n = 40 (configurable)
    4. Solve the robust problem with n = 80 (configurable)
    5. Solve the robust problem with the desired number of nodes
    """
    def __init__(self, apm, weather_model, problem_config, n_nodes, hazards={}, config={}, solver_options={}):
        self.apm = apm
        self.wm = weather_model
        self.hazards = hazards
        self.wm_det = weather_model.get_slice_with_first_member()
        self.pcfg = problem_config
        self.n_nodes = n_nodes
        
        self.config = {
            'n_ig': 200,
            'n_det1': 30,
            'n_det2': 60,
            'n_robust1': 60,
            }        
        
        self.config.update(config)
        # solver_options ['mu_init'] = 10**-9
        self.solver_options = solver_options

    def get_solver_cpu_times(self):
        return [trj.solver_stats['t_wall_mainloop'] for trj in self.trjs_steps[1:]]

    def compare_patched(self):
        trj_patch = patch_mass(self.fpp_det, self.trjs_steps[0].get_interpolator(patch_tu=True))
        trj_after = self.trjs_steps[1].get_interpolator(patch_tu=True)
        sp = trj_patch.t
        spu = trj_patch.tu
        sa = trj_after.t
        sau = trj_after.tu
        plt.subplot(2,2,1)
        plt.plot(trj_patch.x['lon'](sp), trj_patch.x['lat'](sp), 'C1')
        plt.plot(trj_after.x['lon'](sa), trj_after.x['lat'](sa), 'C0')
        plt.subplot(2,2,2)
        plt.plot(sp, trj_patch.x['tas'](sp) * np.ones_like(sp), 'C1')
        plt.plot(sa, trj_after.x['tas'](sa), 'C0')
        plt.subplot(2,2,3)
        plt.plot(sp, trj_patch.x['m_0'](sp), 'C1')
        plt.plot(sa, trj_after.x['m_0'](sa), 'C0')
        plt.subplot(2,2,4)
        plt.plot(spu, trj_patch.u['CL_0'](spu), 'C1')
        plt.plot(sau, trj_after.u['CL_0'](sau), 'C0')
        plt.plot(spu, trj_patch.u['CT_0'](spu), 'C1')
        plt.plot(sau, trj_after.u['CT_0'](sau), 'C0')
        plt.show()

    def check_trj_in_wm_bounds(self, trj, wm: WeatherModel, n_samples=100):
        s_range = np.linspace(trj.t0, trj.tf, n_samples)
        lat = 0
        lon = 0
        P = 0
        t = 0
        try:
            for s in s_range:
                lat = float(trj.x['lat'](s))
                lon = float(trj.x['lon'](s))
                t = float(trj.x['t_0'](s))
                P = float(trj.y['P_0'](s))
                assert wm.bounds['lat'][0] <= lat <= wm.bounds['lat'][1]
                assert wm.bounds['lon'][0] <= lon <= wm.bounds['lon'][1]
                if 'times' in wm.bounds:
                    assert wm.bounds['times'][0] <= t <= wm.bounds['times'][1]
                if 'levels' in wm.bounds and wm.bounds['levels'][0] < wm.bounds['levels'][-1] :
                    assert wm.bounds['levels'][0] <= P/100 <= wm.bounds['levels'][1]
        except AssertionError:
            raise ValueError(f"""Point in initial guess trajectory:
                              (ϕ, λ, P, t) = ({lat:.1f}, {lon:.1f}, {P:.1f}, {t:.1f})
                             is outside bounds of weather model {wm}""")

    def solve(self, stop=False, estimations={}):
        orig_airspeed_setting = self.pcfg['airspeed']
        orig_altitude_setting = self.pcfg['altitude']
        orig_h0 = self.pcfg['h0']
        self.trjs_steps = []
        self.dyncore = DynamicalCore
        if orig_airspeed_setting == 'variable':
            self.pcfg['airspeed'] = 'constant'
        if orig_altitude_setting != 'constant':
            self.pcfg['h0'] = 10e3
            self.pcfg['altitude'] = 'constant'
        self.pcfg['airspeed'] = 'constant'

        FPP = FlightPlanningProblem_FreeRouting_SinglePhase
        fpp_det = FPP(self.apm, self.wm_det, self.wm, self.dyncore, self.pcfg, self.hazards)
        fpp_det.build_ocp()
        self.fpp_det = fpp_det
        #1st step
        ig_trj = ig_2D_ortho(fpp_det, self.config['n_ig'])

        self.check_trj_in_wm_bounds(ig_trj, self.wm_det)
        self.ig_trj = ig_trj
        #self.trjs_steps.append(ig_trj)
        # 2nd step
        
        trj = fpp_det.solve(ig_trj, n_nodes=self.config['n_det1'], solver_options=self.solver_options)
        self.trjs_steps.append(trj)
        if stop:
            return trj
        # 3rd step
        new_ig = trj.get_interpolator(patch_tu=True)
        if orig_airspeed_setting == 'variable':
            self.pcfg['airspeed'] = orig_airspeed_setting
            fpp_det.pcfg['airspeed'] = orig_airspeed_setting
            fpp_det.build_ocp()
            if orig_altitude_setting != self.pcfg['altitude']:
                fpp_det.pcfg['h0'] = orig_h0
                fpp_det.pcfg['altitude'] = orig_altitude_setting
                self.pcfg['h0'] = orig_h0
                self.pcfg['altitude'] = orig_altitude_setting
                if 'm0' in estimations:
                    m0_estimated = estimations['m0']
                else:
                    hcte_ig = patch_mass(fpp_det, new_ig)
                    m0_estimated = hcte_ig.x['m_0'](hcte_ig.t0 + 1e-9)
                new_ig = patch_mass_and_altitude(fpp_det, new_ig, m0_estimated)
            else:
                new_ig = patch_mass(fpp_det, new_ig)
        self.check_trj_in_wm_bounds(new_ig, self.wm_det)
        fpp_det.build_ocp()
        trj = fpp_det.solve(new_ig, n_nodes=self.config['n_det2'], solver_options=self.solver_options)
        # trj = fpp_det.solve(new_ig, n_nodes=self.config['n_det2'], solver_options={'max_iter':0})
        # fpp_det.autoplot()
        self.trjs_steps.append(trj)


        fpp = FPP(self.apm, self.wm, self.wm, self.dyncore, self.pcfg, self.hazards)
        fpp.build_ocp()

        new_ig = trj.get_interpolator(patch_tu=True)
        so = self.solver_options.copy()
        so['max_iter'] = 2000
        trj = fpp.solve(new_ig, n_nodes=self.config['n_det2'], solver_options=so)
        self.trjs_steps.append(trj)

        new_ig = trj.get_interpolator(patch_tu=True)
        trj = fpp.solve(new_ig, n_nodes=self.n_nodes, solver_options=so)
        self.trjs_steps.append(trj)
        
        self.fpp = fpp
        return trj
