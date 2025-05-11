# -*- coding: utf-8 -*-

import numpy as np
import casadi

from roc3.bada4 import *
from roc3.apm import *

from casadi import interpolant, vertcat

class Label(object):
    def __init__(self, path):
        self.arrs = np.load(path)
        self.CL_coords = self.arrs['CL'][0,:]
        self.M_coords = self.arrs['MACH'][0,:]
        self.D_values = self.arrs['jetCR.D'][0,:]
    def get_CD_interpolant(self):
        xx = [self.CL_coords, self.M_coords]
        y = self.D_values.T.flatten()
        I = interpolant('CD', 'bspline', xx, y, {})
        return lambda CL, M: I(vertcat(CL, M))

class ModifiedBADA(BADA4_jet_CR):
    def __init__(self, label, array_path):
        self.isa = ISA()
        self.label_tabular = Label(array_path)
        super().__init__(label)
        self.I_CD = self.label_tabular.get_CD_interpolant()
   # def fc_from_thrust_P_M_T(self, thrust, P, M, T):
   #     Hp = P2Hp(P)
   #     T_ISA = self.isa.IT(Hp)
   #     uncorrected_fc = self.I_fc(vertcat(thrust, Hp, M))
   #     return uncorrected_fc*(T/T_ISA)**.5
    def CD_from_CL_M(self, CL, M):
        return self.I_CD(CL, M)
