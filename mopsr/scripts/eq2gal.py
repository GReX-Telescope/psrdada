#!/usr/bin/env python

from __future__ import print_function
import sys
import astropy.units as u
from astropy.coordinates import SkyCoord

ra = str(sys.argv[1])
dec = str(sys.argv[2])

coords = SkyCoord(ra, dec, unit = (u.degree, u.degree))
gl = coords.galactic.l.degree
gb = coords.galactic.b.degree

print(gl, gb)

