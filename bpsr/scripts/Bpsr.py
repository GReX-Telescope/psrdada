#!/usr/bin/env python

#
# BPSR python module
#

import Dada, struct, math, median_smooth, time, matplotlib, pylab, os
import sys, cStringIO, traceback
import numpy, fnmatch

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def getConfig():
    
  config_file = Dada.DADA_ROOT + "/share/bpsr.cfg"
  config = Dada.readCFGFileIntoDict(config_file)
  return config

def getROACHConfig():

  roach_config_file = Dada.DADA_ROOT + "/share/roach.cfg"
  roach_config = Dada.readCFGFileIntoDict(roach_config_file)
  return roach_config

def getActivePolConfig(beam_name):
  active_pol_config_file = Dada.DADA_ROOT + "/share/bpsr_active_beams.cfg"
  active_pol_config = Dada.readCFGFileIntoDict(active_pol_config_file)
  return active_pol_config["BEAM_" + beam_name + "_p0"] == 'on', \
         active_pol_config["BEAM_" + beam_name + "_p1"] == 'on',

def convertIPToInteger(ipaddr):
  parts = ipaddr.split(".",3)
  ip_int = (int(parts[0]) * (2**24)) + \
           (int(parts[1]) * (2**16)) + \
           (int(parts[2]) * (2**8)) + \
            int(parts[3])
  return ip_int

####################################################################
#
# open connection to ROACH
def connectRoach(dl, rid):

  roach_cfg = getROACHConfig()
  roach_ip   = roach_cfg["ROACH_"+rid]
  roach_port = int(roach_cfg["ROACH_PORT"])

  connected = False
  attempts = 0

  Dada.logMsg(2, dl, "["+rid+"] connectRoach: connecting to " + roach_ip + ":" + str(roach_port))

  return ("ok", fpga)
