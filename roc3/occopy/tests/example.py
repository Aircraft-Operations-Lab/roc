#!/usr/bin/env python
# -*- coding: utf-8 -*-

from occopy.dynamics import *
from occopy.transcription import *

dm = DynamicalSystem()

x1 = dm.add_state('x1')
x2 = dm.add_state('x2')
u = dm.add_control('u')
x1.f = u-dm.t
x2.f = 1

ocp = SinglePhaseOCP(dm)
ocp.add_lagrangian(0.5*u**2 + 0.5*x2**2)
#ocp.add_mayer(72*ocp.tf)

ocp.add_bc(ocp.x0.x1, 0)
ocp.add_bc(ocp.xf.x1, 1)
ocp.add_bc(ocp.x0.x2, 0)
#ocp.add_bc(ocp.xf.v, 0)
ocp.add_bc(ocp.t0, 0)


tt = TrapezoidalTranscription(51)
tt.transcript(ocp)

ig = DiscreteTrajectory()
ig.t0 = 0
ig.tf = (2/7)**0.5
ig.x['x1'] = lambda t: 8/14**0.5*t-0.5*t**2
ig.x['x2'] = lambda t: t
ig.u['u'] = lambda t: 8/14**.5

sol = tt.solve(ig, solver_options={'linear_solver': 'ma97'})
#sol.autoplot()
print(sol.t[-1])
print(sol.x['x1'])
print(sol.u['u'])
