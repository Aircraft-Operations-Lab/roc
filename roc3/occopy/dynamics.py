#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from enum import Enum
from collections import OrderedDict
from casadi import *
import json

from copy import deepcopy as cp

class AttrOrderedDict(OrderedDict):
    def __getattr__(self, v):
        try:
            return super(AttrOrderedDict, self).__getattr__(v)
        except AttributeError:
            if v in self.keys():
                return self[v]
            else:
                raise AttributeError

CX = MX

class DynamicalSystem(object):
    def __init__(self, has_t=True, has_y=False, has_p=False):
        self._has_t = has_t
        self._has_y = has_y
        self._has_p = has_p
        self.f_argcount = 2 + sum(1 if a else 0 for a in [has_t, has_y, has_p])
        self.x = AttrOrderedDict()
        self.u = AttrOrderedDict()
        self.y = AttrOrderedDict()
        self.p = AttrOrderedDict()
        self.nx = 0
        self.ny = 0
        self.np = 0
        self.nu = 0
        self.constraints = []
        self.t = CX.sym('t')
    def add_state(self, name, bounds=(-inf, inf)):
        v = CX.sym(name)
        v.bounds = bounds
        self.x[name] = v
        self.nx += 1
        return v
    def add_control(self, name, bounds=(-inf, inf)):
        v = CX.sym(name)
        v.bounds = bounds
        self.u[name] = v
        self.nu += 1
        return v
    def add_param(self, name, bounds=(-inf, inf)):
        assert self._has_p
        v = CX.sym(name)
        v.bounds = bounds
        self.p[name] = v
        self.np += 1
        return v
    def add_algebraic(self, name, bounds=(-inf, inf)):
        assert self._has_y
        v = CX.sym(name)
        v.bounds = bounds
        self.y[name] = v
        self.ny += 1
        return v
    def assert_dynamics(self):
        for x, var in self.x.items():
            assert hasattr(var, 'f')
    def add_constraint(self, *args, **kwargs):
        self.constraints.append(Constraint(*args, **kwargs))
        
class Constraint(object):
    def __init__(self, expr, bounds=(0, 0), ctr_type='xu'):
        try:
            a = len(bounds)
        except TypeError:
            bounds = (bounds, bounds)
        self.expr = expr
        self.bounds = bounds
        self.ctr_type = ctr_type
        assert ctr_type in ['x', 'xu', 'bc']
    
class SinglePhaseOCP(object):
    def __init__(self, dm):
        self.dm = dm
        self.bc = []
        self.mayer = 0
        self.lagrangian = 0
        self.x0 = AttrOrderedDict()
        self.xf = AttrOrderedDict()
        self.y0 = AttrOrderedDict()
        self.yf = AttrOrderedDict()
        self.t0 = CX.sym('t0')
        self.tf = CX.sym('tf')
        for k in self.dm.x.keys():
            self.x0[k] = CX.sym(k)
            self.xf[k] = CX.sym(k)
        for k in self.dm.y.keys():
            self.y0[k] = CX.sym(k)
            self.yf[k] = CX.sym(k)
    def add_mayer(self, expr):
        self.mayer += expr
    def add_lagrangian(self, expr):
        self.lagrangian += expr
    def add_bc(self, expr, bounds=(0, 0)):
        self.bc.append(Constraint(expr, bounds, 'bc'))

class DiscreteTrajectory(object):
    def __init__(self):
        self.t0 = 0
        self.tf = 1
        self.x = AttrOrderedDict()
        self.y = AttrOrderedDict()
        self.u = AttrOrderedDict()
        self.p = AttrOrderedDict()
    @classmethod
    def load_from_json(cls, f):
        
        d = json.load(f)
        trj = cls()
        trj.x.update(d['x'])
        trj.y.update(d['y'])
        trj.u.update(d['u'])
        trj.p.update(d['p'])
        trj.t = d['t']
        trj.tu = d['tu'] 
        trj.t0 = trj.t[0]
        trj.tf = trj.t[-1]
        if 'status' in d:
            trj.status = d['status']
        else:
            print("Warning: no solver status found in the trajectory")
        if 'J' in d:
            trj.J = d['J']
        return trj
    def get_interpolator(self, patch_tu=False):
        from scipy.interpolate import interp1d as I
        #eps = 2*np.finfo(float).eps
        t = self.t
        tu = self.tu
        if patch_tu:
            tu[0] = t[0]
            tu[-1] = t[-1]
        trj_i = DiscreteTrajectory()
        for name, values in self.x.items():
            trj_i.x[name] = I(t, values)
        for name, values in self.y.items():
            trj_i.y[name] = I(t, values)
        for name, values in self.u.items():
            trj_i.u[name] = I(tu, values)
        trj_i.p = self.p
        trj_i.t = t
        trj_i.tu = tu
        trj_i.t0 = t[0]
        trj_i.tf = t[-1]
        return trj_i
    
    def get_state_control_sequence(self, 
                                   translation_dict=None,
                                   assign_controls_to_previous_t=True):
        if not assign_controls_to_previous_t:
            raise NotImplementedError
        tx = np.array(self.t)
        tu = np.array(self.tu)
        xi2ui = np.zeros_like(tx, dtype=np.int)
        ui2xi = np.zeros_like(tu, dtype=np.int)
        for i in range(tx.shape[0]):
            posterior_controls = np.argwhere(tx[i] <= tu)
            if posterior_controls.size > 0:
                xi2ui[i] = min(posterior_controls)
            else:
                xi2ui[i] = tu.size - 1
        for i in range(tu.shape[0]):
            previous_states =  np.argwhere(tx <= tu[i])
            if previous_states.size > 0:
                ui2xi[i] = max(previous_states)
            else:
                ui2xi = 0
        point_list = []
        indep_label = 'independent_variable'
        if translation_dict:
            if indep_label in translation_dict:
                indep_label = translation_dict[indep_label]
        point_list = []
        for i, t in enumerate(tx):
            point = self.get_state_control_at_index(i, xi2ui[i], translation_dict)
            point[indep_label] = t
            point_list.append(point)
        return point_list
            
    def get_state_control_at_index(self, 
                                   xi: int, 
                                   ui: int, 
                                   translation_dict: dict=None):
        if translation_dict is None:
            translation_dict = {}
        point = {}
        indexes = {'x': xi, 'u': ui, 'p': None}
        trj_vars = {
            'x': (self.x, self.y),
            'u': (self.u,),
            'p': (self.p,),
            }
        for idx, variable_lists in trj_vars.items():
            for variable_list in variable_lists:
                for name, values in variable_list.items():
                    if name in translation_dict:
                        name = translation_dict[name]
                    i = indexes[idx]
                    if i is None:
                        point[name] = values
                    else:
                        point[name] = values[i]
        return point
    def autoplot(self, style='.-', columns=3, scale_t=1.0):
        import matplotlib.pyplot as plt
        n_states = len(self.x)
        n_deps = len(self.y)
        n_ctl = len(self.u)
        n_plots = n_states + n_ctl + n_deps 
        nrows = n_plots//columns + 1
        i = 0
        t = np.array(self.t)*scale_t
        tu = np.array(self.tu)*scale_t
        for name, values in self.x.items():
            i += 1
            plt.subplot(nrows, columns, i)
            plt.plot(t, values, style, color='C0')
            plt.title(name)
        for name, values in self.y.items():
            i += 1
            plt.subplot(nrows, columns, i)
            plt.plot(t, values, style, color='C2')
            plt.title(name)
        for name, values in self.u.items():
            i += 1
            plt.subplot(nrows, columns, i)
            plt.plot(tu, values, style, color='C1')
            plt.title(name)
        plt.tight_layout()
        plt.show()
    def _get_dict(self):
        d = {
            'x': self.x,
            'y': self.y,
            'u': self.u,
            'p': self.p,
            't': self.t,
            'tu': self.tu,
            'status': self.status,
            'J': self.J,
            }
        return d
    def to_json(self):
        d = self._get_dict()
        return json.dumps(d)
    def save_to_json(self, f):
        json.dump(self._get_dict(), f)
    def demux(self, n):
        for d in (self.x, self.y, self.u):
            self._demux_dict(n, d) 
    def _demux_dict(self, n, d):
        for x, v in list(d.items()): 
            if x.endswith('_0'):
                for i in range(n-1):
                    d[x[:-2] + f'_{i+1}'] = v



