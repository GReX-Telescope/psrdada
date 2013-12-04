#!/usr/bin/env python26

# sn_hist2.py
#
# 8/8/12
#
# The script that takes a list of snr values and creates a histogram for
# a given pulsar/frequency.

import sys, datetime, os
import numpy as np
import pylab as P
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import pyplot
from matplotlib import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sn_mods import *

###################################################################
# constants defined in $DADA_ROOT/share/caspsr.cfg
DB_DIR        = "/lfs/data0/caspsr/snr_database"
RESULTS_DIR   = "/lfs/data0/results/caspsr"

def usage():
  sys.stdout.write ("Usage: realtime_hist.py UTC_START PSR_NAME CFREQ\n")

# creates a figure of the specified size
def createFigure(xdim, ydim):

  fig = P.Figure(facecolor='white')
  dpi = fig.get_dpi()
  curr_size = fig.get_size_inches()
  xinches = float(xdim) / float(dpi)
  yinches = float(ydim) / float(dpi)
  fig.set_size_inches((xinches, yinches))
  return fig

def psr_sn_hist(utc_start, jname, cfreq, xres, yres):

  fname1382 = DB_DIR + "/J"+jname+"."+cfreq+".5minSNRs.dat"
  fname_rt = RESULTS_DIR + "/" + utc_start + "/" + jname+"_64.log"
  data = loadtxt(fname1382,dtype='string')
  rt_data = loadtxt(fname_rt,dtype='string')
  snr = []
  length = []
  snr_norm = []
  for i in range(len(data[:,0])):
    snr.append(float(data[i,0].split("=")[1]))
    length.append(float(data[i,1].split("=")[1]))
    if length[i] > 250:
      snr_norm.append(snr[i]/sqrt(length[i]/300))   # Normalize by 5 minute length.
  
  # Since real time subints are 64 seconds long, must account for this when comparing
  # to the 5 min subints plotted in this histogram.
  last_snr_scaled = 0
  if (len(rt_data) > 0):
    last_row = rt_data[-1]
    if (len(last_row) == 4):
      last_snr = float(last_row[0].split("=")[1])
      last_length = float(last_row[1].split("=")[1])
      last_snr_scaled = last_snr/sqrt(last_length/300)

  percentile(snr_norm,last_snr_scaled)

  now = datetime.datetime.today()
  now_str = now.strftime("%Y-%m-%d-%H:%M:%S")
  file_name = RESULTS_DIR + "/" + utc_start + "/snr_hist_" + jname + "_" + now_str + "_" +  str(xres) + "x" + str(yres) + ".png";

  print_axes = 1
  if (xres < 300):
    print_axes = 0
    
  fig = createFigure(xres, yres)
  ax = []
  if (print_axes):
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('SNR (5 min)')
    ax.set_ylabel('# Events')
  else:
    ax = fig.add_axes((0,0,1,1))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  ax.grid(False)

  n, bins = np.histogram(snr_norm,30)
  width = 0.7*(bins[1]-bins[0])
  center=(bins[:-1]+bins[1:])/2

  ax.bar(center,n,align='center',width=width)
  ax.plot([last_snr_scaled,last_snr_scaled],[min(n),max(n)+10],c='red',linestyle='--',alpha=0.7)

  ax.set_xlim((0,max(snr_norm)+10))
  ax.set_ylim((0,max(n)+10))

  FigureCanvas(fig).print_png(file_name)
  fig.delaxes(ax)

def sn_track(utc_start, jname, xres, yres):

  fname = RESULTS_DIR + "/" + utc_start + "/" + jname+"_64.log"
  data = loadtxt(fname,dtype='string')
  snr = []
  length = []
  snr_norm = []
  if (data.ndim > 1):
    for i in range(len(data[:,0])):
      snr.append(float(data[i,0].split("=")[1]))
      length.append(float(data[i,1].split("=")[1]))
      if length[i] > 32:
        snr_norm.append(snr[i]/sqrt(length[i]/64))   # Normalize by 64 sec length.

    now = datetime.datetime.today()
    now_str = now.strftime("%Y-%m-%d-%H:%M:%S")
    file_name = RESULTS_DIR + "/" + utc_start + "/snr_track_" + jname + "_" + now_str + "_" + str(xres) + "x" + str(yres) + ".png";

    print_axes = 1
    if (xres < 300):
      print_axes = 0

    fig = createFigure(xres, yres)
    ax = []
    if (print_axes):
      ax = fig.add_subplot(1,1,1)
      ax.set_xlabel('Time')
      ax.set_ylabel('SNR (from previous 64s)')
    else:
      ax = fig.add_axes((0,0,1,1))
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

    ax.grid(False)

    xvals = arange(len(snr_norm))
    xvals[:] = [x*8 for x in xvals]
    ax.plot(xvals,snr_norm)

    ax.set_ylim((0,max(snr_norm)*1.1))

    FigureCanvas(fig).print_png(file_name)
    fig.delaxes(ax)

###############################################################################
#
# MAIN
#

if len(sys.argv) != 4:
  sys.stderr.write ("Error: 3 arguments required \n")
  usage()
  sys.exit(1)

utc_start = sys.argv[1]
psr_name = sys.argv[2]
cfreq = sys.argv[3]

results_dir = "/lfs/data0/results/caspsr/" + utc_start
db_dir = "/lfs/data0/caspsr/snr_database"

# check that the <UTC_START> dir exists
if not os.path.exists( results_dir ):
  sys.stderr.write ("Error: UTC_START dir did not exist: " +  results_dir + "\n")
  sys.exit(1)

# check if the <PSR_NAME>_64.log file exists
log_file = results_dir + "/" + psr_name + "_64.log"
if not os.path.exists( log_file ):
  sys.stderr.write ("Error: log_file did not exist: " +  log_file + "\n")
  sys.exit(1)

# check that a database corresponding to this PSR exists
db_file = db_dir + "/J" + psr_name + "." + cfreq + ".5minSNRs.dat"
if not os.path.exists ( db_file ):
  sys.stderr.write ("Warning: db_file did not exist: " +  db_file  + "\n")
else:
  psr_sn_hist(utc_start, psr_name, cfreq, 1024, 768)
  psr_sn_hist(utc_start, psr_name, cfreq, 200, 75)

sn_track(utc_start, psr_name, 1024, 768)
sn_track(utc_start, psr_name, 200, 75)
