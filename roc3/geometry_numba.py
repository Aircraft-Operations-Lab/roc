# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
from typing import Tuple
from numba import njit, jitclass, float32, float64

float_type = np.float64

d2r = np.pi/180.0
r2d = d2r**-1

@njit
def latlon2xyz(lat, lon):
    return np.array([
        np.sin(lon)*np.cos(lat),
        -np.cos(lat)*np.cos(lon),
        np.sin(lat),
        ])

@njit
def xyz2latlon(xyz):
    return np.arcsin(xyz[2]), np.arctan2(xyz[0], -xyz[1])

@njit
def orthogonal_rodrigues_rotation(v, k, angle):
    return v*np.cos(angle) + cross(k, v)*np.sin(angle)

rq_spec = [
    ('lat0', float64),
    ('lon0', float64),
    ('latf', float64),
    ('lonf', float64),
    ('theta', float64),
    ('p0', float64[:]),
    ('pf', float64[:]),
    ('p_mid', float64[:]),
    ('ax_r', float64[:]),
    ('ax_q', float64[:]),
]

@njit(float64[:](float64[:], float64[:]))
def cross(a, b):
    """
    Numba cannot np.cross
    :param a: 3-dimensional vector
    :param b: 3-dimensional vector
    :return: the cross-product a^b
    """
    c = np.zeros(3, dtype=float64)
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c

@jitclass(spec=rq_spec)
class RQCoords(object):
    def __init__(self, orig, dest):
        self.lat0 = orig[0]*d2r
        self.lon0 = orig[1]*d2r
        self.latf = dest[0]*d2r
        self.lonf = dest[1]*d2r
        self.p0 = latlon2xyz(self.lat0, self.lon0)
        self.pf = latlon2xyz(self.latf, self.lonf)
        self.p_mid = 0.5*(self.p0 + self.pf)
        self.p_mid /= norm(self.p_mid)
        self.ax_r = cross(self.p0, self.pf - self.p0)
        self.ax_r /= norm(self.ax_r)
        self.ax_q = self.pf - self.p0
        self.ax_q /= norm(self.ax_q)
        self.theta = 2*np.arcsin(norm(self.pf - self.p0)/2)
    def rq2ll(self, r, q):
        pr = orthogonal_rodrigues_rotation(self.p_mid, self.ax_r, 0.5*self.theta*r)
        return np.array(xyz2latlon(orthogonal_rodrigues_rotation(pr, self.ax_q, self.theta*0.25*q)))*r2d

@njit
def generate_gcircle_point_list(origin, destination, min_points=4, angle_resolution_deg=0.2):
    """
    Generates an array of points between origin and destination 
    in a great circle. 
    """
    rq = RQCoords(origin, destination)
    angle_resolution = angle_resolution_deg*d2r
    num_points = max(int(rq.theta//angle_resolution) + 2, min_points)
    # num_segments = num_points - 1
    r_range = np.linspace(-1, 1, int(num_points))
    points = np.zeros((num_points, 2))
    for i, r in enumerate(r_range):
        points[i,:] = rq.rq2ll(r, 0)
    # points = np.array([rq.rq2ll(r, 0) for r in r_range])
    return points

@njit
def project_wind_vector_onto_course(uv, crs):
    wind_u, wind_v = uv
    w_at = np.sin(crs)*wind_u + np.cos(crs)*wind_v
    w_ct = np.cos(crs)*wind_u - np.sin(crs)*wind_v
    return w_at, w_ct

@njit
def groundspeed_and_heading(tas: float, course: float, gamma: float, wind: Tuple[float]) -> Tuple[float]:
    w_at, w_ct = project_wind_vector_onto_course(wind, course)
    v_planar = tas*np.cos(gamma)
    v_along_course = (v_planar**2 - w_ct**2)**0.5
    phase_diff = np.arcsin(w_ct/v_planar)
    return (v_along_course + w_at, course + phase_diff)