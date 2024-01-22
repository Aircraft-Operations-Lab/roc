#!/usr/bin/env python
# -*- coding: utf-8 -*-

from occopy.dynamics import *
from occopy.transcription import *

dm = DynamicalSystem()

x = dm.add_state('x')
u = dm.add_control('u', bounds=(-1, 1))
x.f = u

ocp = SinglePhaseOCP(dm)
ocp.add_lagrangian(x**2)

ocp.add_bc(ocp.x0.x, 1)
ocp.add_bc(ocp.xf.x, 1)
ocp.add_bc(ocp.t0, 0)
ocp.add_bc(ocp.tf, 3)


tt = TrapezoidalTranscription(30)
tt.transcript(ocp)

ig = DiscreteTrajectory()
ig.t0 = 0
ig.tf = 3
ig.x['x'] = lambda t: 1
ig.u['u'] = lambda t: 0

sol = tt.solve(ig, solver_options={'linear_solver': 'ma97'})
sol.autoplot()

