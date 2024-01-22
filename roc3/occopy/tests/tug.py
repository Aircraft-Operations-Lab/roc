#!/usr/bin/env python
# -*- coding: utf-8 -*-

from occopy.dynamics import *
from occopy.transcription import *

dm = DynamicalSystem()

x = dm.add_state('x')
v = dm.add_state('v')
u = dm.add_control('u')
x.f = v
v.f = u

ocp = SinglePhaseOCP(dm)
ocp.add_lagrangian(0.5*u**2)
ocp.add_mayer(72*ocp.tf)

ocp.add_bc(ocp.x0.x, 0)
ocp.add_bc(ocp.xf.x, 128)
ocp.add_bc(ocp.x0.v, 0)
ocp.add_bc(ocp.xf.v, 0)
ocp.add_bc(ocp.t0, 0)


tt = TrapezoidalTranscription(51)
tt.transcript(ocp)

ig = DiscreteTrajectory()
ig.t0 = 0
ig.tf = 8
ig.x['x'] = lambda t: 0.5*12*t**2 - 1/6*3*t**3
ig.x['v'] = lambda t: 12*t - 1.5*t**2
ig.u['u'] = lambda t: 12-3*t

sol = tt.solve(ig, solver_options={'linear_solver': 'ma97'})
sol.autoplot()

print(sol.x['v'])
print(sol.u['u'])
print(sol.J)
