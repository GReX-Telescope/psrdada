#!/usr/bin/env python

# auxiliary_plotter.py
#
# 8/8/12
#
# The script that takes a list of snr values and creates a histogram for
# a given pulsar/frequency.

import Dada, Mopsr

import sys, datetime, os
import numpy as np
import pylab as P
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import pyplot
from matplotlib import transforms
from matplotlib.dates import strpdate2num
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sn_mods import *

###################################################################

def usage():
  sys.stdout.write ("Usage: auxiliary_plotter.py UTC_START/ANT\n")

# creates a figure of the specified size
def createFigure(xdim, ydim):

  fig = P.Figure(facecolor='white')
  dpi = fig.get_dpi()
  curr_size = fig.get_size_inches()
  xinches = float(xdim) / float(dpi)
  yinches = float(ydim) / float(dpi)
  fig.set_size_inches((xinches, yinches))
  return fig

def file_len(fname):
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def sn_track(utc_start, ant, xres, yres):

  fname = cfg["SERVER_RESULTS_DIR"] + "/" + utc_start + "/" + ant + "/" + "snr_60.log"
  data = loadtxt(fname,dtype='string')
  snr = []
  length = []
  snr_norm = []
  if (data.ndim > 1):
    for i in range(len(data[:,0])):
      snr.append(float(data[i,0].split("=")[1]))
      length.append(float(data[i,1].split("=")[1]))
      if length[i] > 32:
        snr_norm.append(snr[i]/sqrt(length[i]/60))   # Normalize by 60 sec length.

    now = datetime.datetime.today()
    now_str = now.strftime("%Y-%m-%d-%H:%M:%S")
    file_name = cfg["SERVER_RESULTS_DIR"] + "/" + utc_start + "/" + now_str + "." + str(ant) + ".sn." + str(xres) + "x" + str(yres) + ".png";

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

def pm_track(utc_start, ant, xres, yres):

  fname = cfg["SERVER_RESULTS_DIR"] + "/" + utc_start + "/" + ant + "/power_monitor.log"
  if file_len(fname) < 3:
    return

  data = np.genfromtxt(fname, delimiter=',',names=True, \
                   dtype=('i4', 'f4', 'f4', 'f4','f4', 'f4', 'f4', 'f4', 'f4'))

  now = datetime.datetime.today()
  now_str = now.strftime("%Y-%m-%d-%H:%M:%S")
  file_name = cfg["SERVER_RESULTS_DIR"] + "/" + utc_start + "/" + now_str + "." + str(ant) + ".pm." + str(xres) + "x" + str(yres) + ".png";

  print_axes = 1
  if (xres < 500):
    print_axes = 0

  fig = createFigure(xres, yres)
  fig.subplots_adjust(hspace=0,wspace=0)
  cols = data.dtype.names
  ncols = len(cols)
  ax = []
  for i in range(1, ncols):

    if print_axes:
      ax = fig.add_subplot(ncols,1,i, label=cols[i])
      ax.yaxis.set_label_position("right")
      ax.set_ylabel(cols[i] + " MHz", rotation='horizontal')
      if i == ncols - 1:
        ax.set_xlabel('Time (seconds)')
      else:
        ax.get_xaxis().set_ticklabels([])
      if i == 1:
        ax.set_title('Total Power Monitor')
      ax.yaxis.set_major_locator(MaxNLocator(5, prune='both'))
      #for tick in ax.yaxis.get_major_ticks():
      #  tick.label.set_fontsize(10) 
      #ax.yaxis.label.set_fontsize(10)
    else:
      if i == 1:
        ax = fig.add_axes((0,0,1,1))
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

    ax.grid(True)
    ax.plot(data['UTC'], data[cols[i]], label=cols[i])
    ax.set_xlim((0,max(data['UTC'])))

  FigureCanvas(fig).print_png(file_name)
  fig.delaxes(ax)


###############################################################################
#
# MAIN
#

if len(sys.argv) != 2:
  sys.stderr.write ("Error: 1 argument required \n")
  usage()
  sys.exit(1)

path = sys.argv[1]

utc_start, ant = path.split('/', 2)

# get the backend configuration file
cfg = Mopsr.getConfig()

results_dir = cfg["SERVER_RESULTS_DIR"] + "/" + utc_start

# check that the <UTC_START> dir exists
if not os.path.exists( results_dir ):
  sys.stderr.write ("Error: UTC_START dir did not exist: " +  results_dir + "\n")
  sys.exit(1)

# see what antenna's exist in the <UTC_START> dir
ants = []
for o in os.listdir(results_dir):
  if os.path.isdir( results_dir + "/" + o):
    ants.append(o)

if len(ants) == 0:
  sys.stderr.write ("Warning: no antenna subdirs existed in " +  results_dir + "\n")
  sys.exit(0)


# check if the power_monitor.log file exists
#for ant in ants:
power_monitor = results_dir + "/" + ant + "/power_monitor.log"
if os.path.exists (power_monitor):
  pm_track(utc_start, ant, 200, 150)
  #pm_track(utc_start, ant, 1024, 768)

#log_file = results_dir + "/" + ant + "/snr_60.log"
#if os.path.exists( log_file ):
#  for ant in ants:
#    sn_track(utc_start, ant, 1024, 768)
#    sn_track(utc_start, ant, 200, 150)

sys.exit(0)
