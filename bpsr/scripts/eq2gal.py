#!/usr/bin/python

import sys, ephem
from math import degrees,radians
ra = sys.argv[1]
dec= sys.argv[2]

radec=ephem.Equatorial(radians(float(ra)),radians(float(dec)))
g = ephem.Galactic(radec)

ver = ephem.__version__

if ver == '3.7.5.2' or ver == '3.7.5.1' or ver == '3.7.4.1':
  gl = g.lon
else:
  gl = g.long

gb = g.lat

print degrees(gl),degrees(gb)
