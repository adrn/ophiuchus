# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
from gary.potential.tests.helpers import PotentialTestBase, CompositePotentialTestBase
from gary.potential.io import load

# Project
from .. import WangZhaoBarPotential, OphiuchusPotential
from ... import potential as op

class TestWangZhao(PotentialTestBase):
    potential = WangZhaoBarPotential(1E10, 1., 27*u.degree, 60*u.km/u.s/u.kpc)
    w0 = [4.0,0.7,-0.9,0.0352238,0.1579493,0.02]

    def test_save_load(self, tmpdir):
        fn = str(tmpdir.join("{}.yml".format(self.name)))
        self.potential.save(fn)
        p = load(fn, module=op)
        p.value(self.w0[:self.w0.size//2])

class TestOphiuchus(CompositePotentialTestBase):
    potential = OphiuchusPotential()
    w0 = [8.0,0.7,-0.9,0.0352238,0.1579493,0.02]

    def test_save_load(self, tmpdir):
        fn = str(tmpdir.join("{}.yml".format(self.name)))
        self.potential.save(fn)
        p = load(fn, module=op)
        p.value(self.w0[:self.w0.size//2])
