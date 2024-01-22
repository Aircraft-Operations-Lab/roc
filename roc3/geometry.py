# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
from typing import Tuple


d2r = np.pi / 180.0
r2d = d2r ** -1

RE_average = 6.371e6

def latlon2xyz(lat, lon):
    return np.array([
        np.sin(lon) * np.cos(lat),
        -np.cos(lat) * np.cos(lon),
        np.sin(lat),
    ])


def xyz2latlon(xyz):
    return np.arcsin(xyz[2]), np.arctan2(xyz[0], -xyz[1])


def orthogonal_rodrigues_rotation(v, k, angle):
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle)


class RQCoords(object):
    def __init__(self, orig, dest):
        self.lat0 = orig[0] * d2r
        self.lon0 = orig[1] * d2r
        self.latf = dest[0] * d2r
        self.lonf = dest[1] * d2r
        self.p0 = latlon2xyz(self.lat0, self.lon0)
        self.pf = latlon2xyz(self.latf, self.lonf)
        self.p_mid = 0.5 * (self.p0 + self.pf)
        self.p_mid /= norm(self.p_mid)
        self.ax_r = np.cross(self.p0, self.pf - self.p0)
        self.ax_r /= norm(self.ax_r)
        self.ax_q = self.pf - self.p0
        self.ax_q /= norm(self.ax_q)
        self.ax_qxr = np.cross(self.ax_q, self.ax_r)
        self.theta = 2 * np.arcsin(norm(self.pf - self.p0) / 2)

    def rq2ll(self, r, q):
        pr = orthogonal_rodrigues_rotation(self.p_mid, self.ax_r, 0.5 * self.theta * r)
        return np.array(xyz2latlon(orthogonal_rodrigues_rotation(pr, self.ax_q, self.theta * 0.25 * q))) * r2d

    def get_distance_to_plane(self, lat, lon):
        return - np.dot(self.ax_r, latlon2xyz(lat * d2r, lon * d2r))

    @property
    def distance_o2d_m(self):
        return RE_average * self.theta

    @property
    def distance_o2d_nm(self):
        return self.distance_o2d_m / 1852

    # def ll2rq(self, ll):
    #     xyz = latlon2xyz(ll[0]*d2r, ll[1]*d2r)
    #


def generate_gcircle_point_list(origin, destination, angle_resolution_deg=0.2, get_theta=False):
    """
    Generates an array of points between origin and destination
    in a great circle.
    """
    rq = RQCoords(origin, destination)
    angle_resolution = angle_resolution_deg * d2r
    num_segments = rq.theta // angle_resolution + 1
    return generate_gcircle_point_list_n(origin, destination, num_segments, get_theta=get_theta, rq=rq)


def generate_gcircle_point_list_n(origin, destination, num_segments, get_theta=False, rq=None):
    if rq is None:
        rq = RQCoords(origin, destination)
    num_points = num_segments + 1
    r_range = np.linspace(-1, 1, num_points)
    points = np.array([rq.rq2ll(r, 0) for r in r_range])
    if get_theta:
        return points, rq.theta
    else:
        return points


def project_wind_vector_onto_course(uv: Tuple[float, float], crs: float) -> Tuple[float, float]:
    wind_u, wind_v = uv
    w_at = np.sin(crs) * wind_u + np.cos(crs) * wind_v
    w_ct = np.cos(crs) * wind_u - np.sin(crs) * wind_v
    return w_at, w_ct


def groundspeed_and_heading(tas: float, course: float, gamma: float, wind: Tuple[float, float]) -> Tuple[float, float]:
    w_at, w_ct = project_wind_vector_onto_course(wind, course)
    v_planar = tas * np.cos(gamma)
    v_along_course = (v_planar ** 2 - w_ct ** 2) ** 0.5
    phase_diff = np.arcsin(w_ct / v_planar)
    return v_along_course + w_at, course + phase_diff
