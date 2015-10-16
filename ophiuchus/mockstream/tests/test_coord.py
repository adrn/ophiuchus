# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Project
from .._coord import (_test_sat_rotation_matrix,
                      _test_car_to_cyl_roundtrip, _test_cyl_to_car_roundtrip,
                      _test_vcar_to_cyl_roundtrip, _test_vcyl_to_car_roundtrip)

def test_sat_rotation_matrix():
    _test_sat_rotation_matrix()

def test_car_to_cyl_roundtrip():
    _test_car_to_cyl_roundtrip()

def test_cyl_to_car_roundtrip():
    _test_cyl_to_car_roundtrip()

def test_vcar_to_cyl_roundtrip():
    _test_vcar_to_cyl_roundtrip()

def test_vcyl_to_car_roundtrip():
    _test_vcyl_to_car_roundtrip()
