#!/usr/bin/env python

# Dual Polarisation (PP, QQ) Spectrometer

import Dada, Bpsr, struct, math, median_smooth, time, matplotlib, pylab, os
import sys, cStringIO, traceback
import corr, numpy, fnmatch

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def programRoach(dl, fpga, rid, boffile='default'):

  roach_cfg = Bpsr.getROACHConfig()
  roach_bof = roach_cfg["ROACH_BOF"]
  if (not boffile == 'default'):
    roach_bof = boffile

  # program bit stream
  programmed = False
  attempts = 0
  while (not programmed and attempts < 5):

    Dada.logMsg(2, dl, "["+rid+"] programming FPGA with " + roach_bof)
    Dada.logMsg(3, dl, "["+rid+"] programRoach: programming FPGA with " + roach_bof)
    prog_result = fpga.progdev(roach_bof)
    if (prog_result):
      Dada.logMsg(2, dl, "["+rid+"] programRoach: programming done")
    else:
      Dada.logMsg(0, dl, "["+rid+"] programRoach: programming FAILED")

    time.sleep(0.1)

    try:
      # try to read from a register
      #enable = fpga.read_int('enable')
      #Dada.logMsg(3, dl, "["+rid+"] enable = " + str(enable))
      port = fpga.read_int('reg_10GbE_destport0')
      Dada.logMsg(3, dl, "["+rid+"] reg_10GbE_destport0 = " + str(port))

      # try to write to a register
      #enable = 0
      #Dada.logMsg(3, dl, "["+rid+"] enable" + str(enable))
      #fpga.write_int('enable', enable)
      port = 8000
      Dada.logMsg(3, dl, "["+rid+"] reg_10GbE_destport0 " + str(port))
      fpga.write_int('reg_10GbE_destport0', port)

      # if we got this far without throwing an exception, we are programmed!
      programmed = True

    except:

      # give it a chance to be ready for commands
      Dada.logMsg(1, dl, "["+rid+"] exception when trying to read/write")
      time.sleep(1.0)
      attempts += 1

  if (not programmed):
    Dada.logMsg(0, dl, "["+rid+"] failed to program FPGA with " + roach_bof)
    return ("fail", "could not program FPGA with " + roach_bof)

  return ("ok", "ROACH programmed with " + roach_bof)


def waitForARP(dl, fpga, tengbe_device, ip_addr, rid):

  # set packet rate to be extremely low [1-2 MB/s]
  ip_str = "0.0.0.0"
  ip   = Bpsr.convertIPToInteger(ip_str)
  port = 0
  acc_len = 1024
  sync_period = 100 * acc_len * 2048

  Dada.logMsg(2, dl, "["+rid+"] reg_ip " + ip_str)
  fpga.write_int('reg_ip', ip)

  Dada.logMsg(2, dl, "["+rid+"] reg_10GbE_destport0 " + str(port))
  fpga.write_int('reg_10GbE_destport0', port)

  Dada.logMsg(2, dl, "["+rid+"] acc_len " + str(acc_len-1))
  fpga.write_int('reg_acclen', acc_len-1)

  Dada.logMsg(2, dl, "["+rid+"] sync_period " + str(sync_period))
  fpga.write_int('reg_sync_period', sync_period)

  # setup the string to look for in the ARP table
  parts = ip_addr.split(".")
  arp_line = "IP: " + parts[0].rjust(3) + "." + parts[1].rjust(3) + "." + parts[2].rjust(3) + "." + parts[3].rjust(3) + ": MAC: FF FF FF FF FF FF "
  Dada.logMsg(3, dl, "["+rid+"] waitForARP arp_line = " + arp_line)

  arp_valid = False
  attempts = 0
  while ((not arp_valid) and attempts < 60):

    arp_table = fpga.return_10gbe_arp_table(tengbe_device)
    arp_valid = True
    lines = arp_table.split("\n")
    for line in lines:
      Dada.logMsg(3, dl, "["+rid+"] testing line '" + line + "' == '" + arp_line + "'")
      if (line == arp_line):
        arp_valid = False

    if (not arp_valid):
      Dada.logMsg(2, dl, "["+rid+"] arp not yet valid")
      time.sleep(1.0)
      attempts += 1

  if (attempts < 60):
    Dada.logMsg(3, dl, "["+rid+"] ARP now valid")
    arp_table = fpga.return_10gbe_arp_table(tengbe_device)
    return ("ok", "")

  else:
    return ("fail", "")

#
# Configure the programmed ROACH board
#
def configureRoach(dl, fpga, rid, cfg):

  roach_cfg = Bpsr.getROACHConfig()
  roach_beam = roach_cfg["BEAM_"+rid]

  roach_10gbe_src_ip    = Bpsr.convertIPToInteger(roach_cfg["ROACH_10GbE_SRC_"+rid])
  roach_10gbe_dest_ip   = Bpsr.convertIPToInteger(roach_cfg["ROACH_10GbE_DEST_"+rid])

  Dada.logMsg(2, dl, "["+rid+"] configureRoach: " + roach_cfg["ROACH_10GbE_SRC_"+rid] + " -> " + str(roach_10gbe_src_ip))
  Dada.logMsg(2, dl, "["+rid+"] configureRoach: " + roach_cfg["ROACH_10GbE_DEST_"+rid] + " -> " + str(roach_10gbe_dest_ip))

  # Same port in src and dest
  roach_10gbe_src_port  = int(cfg["PWC_UDP_PORT_"+rid])
  roach_10gbe_dest_port = int(cfg["PWC_UDP_PORT_"+rid])

  # use a designated private MAC address [02:xx:xx:xx:xx:xx]
  roach_10gbe_src_mac   = (2<<40) + (2<<32) + roach_10gbe_src_ip

  # hardware device
  tengbe_device = roach_cfg["ROACH_10GBE_DEVNAME"]
  # linux device
  tengbe_tapdev = tengbe_device

  # start tgtap, which configures the 10Gbe port and begins an ARP process 
  Dada.logMsg(2, dl, "["+rid+"] fpga.tap_start()")
  fpga.tap_start(tengbe_tapdev,tengbe_device,roach_10gbe_src_mac,roach_10gbe_src_ip,roach_10gbe_src_port)
  time.sleep(0.5)
  gbe0_link = bool(fpga.read_int(tengbe_device))
  if gbe0_link:
    Dada.logMsg(2, dl, "["+rid+"] configureRoach: 10GbE device now active")
  else:
    Dada.logMsg(-1, dl, "["+rid+"] configureRoach: 10GbE device NOT active")

  result, response = waitForARP(dl, fpga, tengbe_device, roach_cfg["ROACH_10GbE_DEST_"+rid], rid)
  if (not result == "ok"):
    Dada.logMsg(0, dl, "["+rid+"] configureRoach: ARP did not become valid after 60 seconds")
    return ("fail", fpga)

  # now configure device properly
  Dada.logMsg(2, dl, "["+rid+"] reg_ip " + roach_cfg["ROACH_10GbE_DEST_"+rid])
  fpga.write_int('reg_ip', roach_10gbe_dest_ip)

  Dada.logMsg(2, dl, "["+rid+"] reg_10GbE_destport0 " + str(roach_10gbe_dest_port))
  fpga.write_int('reg_10GbE_destport0', roach_10gbe_dest_port)

  Dada.logMsg(1, dl, "["+rid+"] programmed " + roach_beam + ", UDP -> " + roach_cfg["ROACH_10GbE_DEST_"+rid] + ":" + str(roach_10gbe_dest_port))
  Dada.logMsg(2, dl, "["+rid+"] configureRoach: returning ok")

  return ("ok", fpga)

def accLenRoach(dl, fpga, acc_len, rid):

  sync_period = 100 * acc_len * 2048

  Dada.logMsg(2, dl, "["+rid+"] acc_len " + str(acc_len-1))
  fpga.write_int('reg_acclen', acc_len-1)

  Dada.logMsg(2, dl, "["+rid+"] sync_period " + str(sync_period))
  fpga.write_int('reg_sync_period', sync_period)

  return ("ok", "")

###############################################################################
#
# arm roach to begin start of data
#
def rearm(dl, fpga):
  if (fpga != []):
    fpga.write_int('reg_arm',0)
    fpga.write_int('reg_arm',1)
    return "ok"
  else:
    return "fail"

###############################################################################
#
# start flow of 10GbE packets
#
def startTX(dl, fpga):
  if (fpga != []):
    #fpga.write_int('enable', 1)
    return "ok"
  else:
    return "fail"


###############################################################################
#
# stop flow of 10GbE packets
def stopTX(dl, fpga):
  if (fpga != []):
    #fpga.write_int('enable', 0)
    return "ok"
  else:
    return "fail"

def setGains(dl, fpga, rid, value):

  Dada.logMsg(1, dl, "["+rid+"]  setting gain levels to="+str(value))
  fpga.write_int('reg_coeff_pol1', value)
  fpga.write_int('reg_coeff_pol2', value)
  return ("ok", "")


def setLevels(dl, fpga, pol0_flag, pol1_flag, rid):

  n_attempts = 3

  # start with sensible first guess?
  fpga.write_int('reg_output_bitselect', 0)
  fpga.write_int('reg_coeff_pol1', 4096)
  fpga.write_int('reg_coeff_pol2', 4096)

  # intial guess at gain values
  pol1_coeff = fpga.read_int('reg_coeff_pol1')
  pol2_coeff = fpga.read_int('reg_coeff_pol2')
  bit_window = 0

  for i in range (n_attempts):

    fpga.write_int('reg_output_bitselect', bit_window)
    fpga.write_int('reg_coeff_pol1', pol1_coeff)
    fpga.write_int('reg_coeff_pol2', pol2_coeff)

    time.sleep(0.1)

    Dada.logMsg(2, dl, "["+rid+"] setLevels: setting coeffs ["+str(pol1_coeff)+", "+str(pol2_coeff)+"]")

    pol1, pol2 = bramdumpRoach(dl, fpga, 1)

    # do not count the first 150 channels out of 1024
    pol1_med = numpy.median( numpy.array(pol1)[75:] )
    pol2_med = numpy.median( numpy.array(pol2)[75:] )

    Dada.logMsg(2, dl, '['+rid+'] setLevels: pol1_med='+str(pol1_med)+' pol2_med='+str(pol2_med))

    # find the right bit window and coeffs
    bit_window, pol1_coeff, pol2_coeff = findBitWindow(dl, pol1_med, pol2_med, pol1_coeff, pol2_coeff)

    Dada.logMsg(2, dl, '['+rid+'] setLevels: bit_window='+str(bit_window)+' pol1_coeff='+str(pol1_coeff)+' pol2_coeff='+str(pol2_coeff))

  # now get a more sensitive estimate
  pol1, pol2 = bramdumpRoach(dl, fpga, 16)

  # do not count the first 150 channels out of 1024
  pol1_med = numpy.median( numpy.array(pol1)[75:] )
  pol2_med = numpy.median( numpy.array(pol2)[75:] )

  Dada.logMsg(2, dl, '['+rid+'] setLevels: pol1_med='+str(pol1_med)+' pol2_med='+str(pol2_med))

  # find the right bit window and coeffs
  bit_window, pol1_coeff, pol2_coeff = findBitWindow(dl, pol1_med, pol2_med, pol1_coeff, pol2_coeff)

  Dada.logMsg(2, dl, '['+rid+'] setLevels: bit_window='+str(bit_window)+' pol1_coeff='+str(pol1_coeff)+' pol2_coeff='+str(pol2_coeff))

  fpga.write_int('reg_output_bitselect', bit_window)
  fpga.write_int('reg_coeff_pol1', pol1_coeff)
  fpga.write_int('reg_coeff_pol2', pol2_coeff)

  Dada.logMsg(2, dl, '['+rid+'] bit_window='+str(bit_window)+' pol1='+str(pol1_coeff)+' pol2='+str(pol2_coeff))

  return ("ok", "pol1="+str(pol1_coeff)+",pol2="+str(pol2_coeff))


def bramdumpRoach(dl, fpga, n_dumps=1):

  # every second channel of the 1024 channel FB, interpet as Big Endian, 32 bit unsigned integer
  pol1 = numpy.array(struct.unpack('>512I',fpga.read('scope_output1_bram',512*4,0)))
  pol2 = numpy.array(struct.unpack('>512I',fpga.read('scope_output2_bram',512*4,0)))

  if (n_dumps <= 1):
    return pol1, pol2

  for i in range(n_dumps-1):
    tmp1 = numpy.array(struct.unpack('>512I',fpga.read('scope_output1_bram',512*4,0)))
    tmp2 = numpy.array(struct.unpack('>512I',fpga.read('scope_output2_bram',512*4,0)))
    for j in range(512):
      pol1[j] = pol1[j] + tmp1[j]
      pol2[j] = pol2[j] + tmp2[j]

  for i in range(512):
    pol1[i] = int(float(pol1[i]) / float(n_dumps))
    pol2[i] = int(float(pol2[i]) / float(n_dumps))
  
  return pol1, pol2 

###############################################################################
#
# dump the BRAM contents to a file for asynchronous plotting
#
def bramdiskRoach(dl, fpga, timestamp, roach_name):

  Dada.logMsg(2, dl, "[" + roach_name + "] bramdiskRoach: dumping BRAM");

  file_prefix = timestamp + "_" + roach_name

  # get the current values for pol1 and pol2
  pol1, pol2 = bramdumpRoach(dl, fpga, 3)

  # get the current bit window
  bit_window = fpga.read_int('reg_output_bitselect')

  # file to read is 
  bram_file = file_prefix + ".bram"
  fptr = open(bram_file, "wb")

  bit_window_bin = struct.pack("I1",bit_window)
  pol1_binary = struct.pack("512f",*pol1)
  pol2_binary = struct.pack("512f",*pol2)

  fptr.write(bit_window_bin)
  fptr.write(pol1_binary)
  fptr.write(pol2_binary)
  fptr.close()
  
  return 'ok'


###############################################################################
#
# create 3 plot files from a bramdump of the ROACH board
#
def bramplotRoach(dl, fpga, timestamp, roach_name):

  Dada.logMsg(2, dl, "[" + roach_name + "] bramplotRoach: dumping BRAM");

  file_prefix = timestamp + "_" + roach_name

  # get the current values for pol1 and pol2
  pol1, pol2 = bramdumpRoach(dl, fpga, 3)

  # get the current bit window
  bit_window = fpga.read_int('reg_output_bitselect')

  nvals = len(pol1)
  xaxis = []
  pol1_float = []
  pol2_float = []

  for i in range(nvals):
    xaxis.append(i)
    if (pol1[i] > 0):
      pol1_float.append(math.log(pol1[i],2))
    else:
      pol1_float.append(0)
    if (pol2[i] > 0):
      pol2_float.append(math.log(pol2[i],2))
    else:
      pol2_float.append(0)

  # reverse the elements in the list so the plots match the mon plots
  # xaxis.reverse()
  pol1_float.reverse()
  pol2_float.reverse()

  ymin = 0
  ymax = 32
  xmin = 0
  xmax = 511

  xres_list = [1024, 400,   112]
  yres_list = [768,  300,   84]
  pp_list   = [True, True, False]

  Dada.logMsg(2, dl, "[" + roach_name + "] bramplotRoach: starting plot");

  for i in range(3):

    xres = xres_list[i]
    yres = yres_list[i]
    pp   = pp_list[i]

    filename = file_prefix + "_" + str(xres) + "x" + str(yres) + ".png"
    
    # creates a figure of the specified resolution, white on black
    fig = createFigure(xres, yres)
    ax = []
    if (pp):
      ax = fig.add_subplot(1,1,1)
    else:
      ax = fig.add_axes((0,0,1,1))
    set_foregroundcolor(ax, 'white')
    set_backgroundcolor(ax, 'black')

    if (pp):
      ax.set_title('Pre Bit-Selected Bandpass')
      ax.set_xlabel('Channel')
      ax.set_ylabel('Activated Bits')
    else:
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

    ax.grid(False)

    # plot pol1 as red, pol2 as greeb
    ax.plot(xaxis, pol1_float, 'r-', label='pol 1')
    ax.plot(xaxis, pol2_float, 'g-', label='pol 2')

    if (pp):
      ax.legend()

    # print grey frames for the inactive parts of the 32 bit number
    if (bit_window == 0):
      ax.axhspan(8, 32, facecolor='grey', alpha=0.75)
    elif (bit_window == 1):
      ax.axhspan(0, 8, facecolor='grey', alpha=0.75)
      ax.axhspan(16, 32, facecolor='grey', alpha=0.75)
    elif (bit_window == 2):
      ax.axhspan(0, 16, facecolor='grey', alpha=0.75)
      ax.axhspan(24, 32, facecolor='grey', alpha=0.75)
    else:
      ax.axhspan(0, 24, facecolor='grey', alpha=0.75)

    # hard set the x,y limits
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    FigureCanvas(fig).print_png(filename)
    fig.delaxes(ax)
    Dada.logMsg(2, dl, "[" + roach_name + "] bramplotRoach: plotting complete");

    # now cleanup any excess plotting files
    filematch = "*"+roach_name + "_" + str(xres) + "x" + str(yres) + ".png"
    Dada.logMsg(3, dl, "bramplotRoach: checking " + os.getcwd() + " for matching " + filematch)
    filelist = os.listdir(os.getcwd())
    filelist.sort(reverse=True)
    count = 0
    for filename in filelist:
      if (fnmatch.fnmatch(filename, filematch)):
        count += 1
        if (count > 3):
          Dada.logMsg(3, dl, "bramplotRoach: cleaning file: "+filename)
          os.remove(filename)

  Dada.logMsg(2, dl, "[" + roach_name + "] bramplotRoach: cleanup complete");
  return "ok"

def calculate_max (data):
  maxval = 0
  for i, value in enumerate(data):
    if (i < 400):
      if (value > maxval):
        maxval = value

  return maxval

def findBitWindow(dl, pol1_med, pol2_med, gain1, gain2):

  bit_window = 0

  bitsel_min = [0, 256, 65535, 16777216]
  bitsel_mid = [64, 8192, 2097152, 536870912]
  bitsel_max = [255, 65535, 16777215, 4294967295]

  val = (pol1_med + pol2_med) / 2.0

  Dada.logMsg(3, dl, "findBitWindow: max value = "+str(val))
  Dada.logMsg(3, dl, "findBitWindow: initial gains ["+str(gain1)+", "+str(gain2)+"]")

  current_window = 0
  desired_window = 0

  for i in range(4):

    # If average (max) value is in the lower half
    if ((val > bitsel_min[i]) and (val <= bitsel_mid[i])):
      current_window = i

      if (i == 0):
        desired_window = 0
      else:
        desired_window = i-1

    # If average (max)n value is in the upper half, simply raise to
    # the top of this window
    if ((val > bitsel_mid[i]) and (val <= bitsel_max[i])):
      current_window = i
      desired_window = i

  if (desired_window == 3):
    desired_window = 2

  Dada.logMsg(3, dl, "findBitWindow: current_window="+str(current_window)+" desired_window="+str(desired_window))

  if ((pol1_med == 0) or (pol2_med == 0)):
    pol1_med = 1
    pol2_med = 1

  # If current gains are too high (2^16) then shift down a window
  if ((gain1 > 32000) or (gain2 > 32000)):
    if (desired_window > 0):
      desired_window -= 1

  # If current gains are too low (2^3) then shift down a window
  if ((gain1 < 8) or (gain2 < 8)):
    if (desired_window < 3):
      desired_window += 1

  desired_val = ((float((bitsel_max[desired_window]+1)) / 4.0))

  Dada.logMsg(3, dl, "findBitWindow: desired_val="+str(desired_val))

  gain_factor1 = desired_val / float(pol1_med)
  gain_factor2 = desired_val / float(pol2_med)

  Dada.logMsg(3, dl, "findBitWindow: 1 new gains ["+str(gain_factor1)+", "+str(gain_factor2)+"]")

  gain_factor1 = math.sqrt(gain_factor1)
  gain_factor2 = math.sqrt(gain_factor2)

  Dada.logMsg(3, dl, "findBitWindow: 2 new gains ["+str(gain_factor1)+", "+str(gain_factor2)+"]")

  gain_factor1 *= gain1
  gain_factor2 *= gain2

  Dada.logMsg(3, dl, "findBitWindow: 3 new gains ["+str(gain_factor1)+", "+str(gain_factor2)+"]")

  gain1 = int(gain_factor1)
  gain2 = int(gain_factor2)

  Dada.logMsg(3, dl, "findBitWindow: gains "+str(gain_factor1)+" -> "+str(gain1))
  Dada.logMsg(3, dl, "findBitWindow: gains "+str(gain_factor2)+" -> "+str(gain2))

  if (gain1 > 262143):
    gain1 = 262143

  if (gain2 > 262143):
    gain2 = 262143 

  return desired_window, gain1, gain2

def plotLog2List(bit_window, pol1, pol2):

  nvals = len(pol1)
  xaxis = []
  pol1_float = []
  pol2_float = []

  for i in range(nvals):
    xaxis.append(i)
    if (pol1[i] > 0):
      pol1_float.append(math.log(pol1[i],2))
    else:
      pol1_float.append(0)
    if (pol2[i] > 0):
      pol2_float.append(math.log(pol2[i],2))
    else:
      pol2_float.append(0)

  ymin = 0
  ymax = 32
  xmin = 0
  xmax = 511

  # creates a figure of the specified resolution, white on black
  fig = createFigure(1024, 768)
  ax = fig.add_subplot(1,1,1)
  set_foregroundcolor(ax, 'white')
  set_backgroundcolor(ax, 'black')
  ax.set_title('Pre Decimation Bandpass')
  ax.set_xlabel('Channel')
  ax.set_ylabel('Activated Bits')
  ax.grid(False)

  # plot pol1 as red, pol2 as greeb
  ax.plot(xaxis, pol1_float, 'r-', label='pol 1')
  ax.plot(xaxis, pol2_float, 'g-', label='pol 2')

  # add a legend
  ax.legend()

  # print grey frames for the inactive parts of the 32 bit number
  if (bit_window == 0):
    ax.axhspan(8, 32, facecolor='grey', alpha=0.75)
  elif (bit_window == 1):
    ax.axhspan(0, 8, facecolor='grey', alpha=0.75)
    ax.axhspan(16, 32, facecolor='grey', alpha=0.75)
  elif (bit_window == 2):
    ax.axhspan(0, 16, facecolor='grey', alpha=0.75)
    ax.axhspan(24, 32, facecolor='grey', alpha=0.75)
  else:
    ax.axhspan(0, 24, facecolor='grey', alpha=0.75)

  # hard set the x,y limits
  matplotlib.pylab.xlim((xmin, xmax))
  matplotlib.pylab.ylim((ymin, ymax))

  # set filename
  #filename = "plot_" + str(1024) + "x" + str(768) + ".png"
  #print 'creating ' + filename
  #dpi = fig.get_dpi()
  #fig.savefig(filename, dpi=(dpi), facecolor='black')

  matplotlib.pylab.show()
  matplotlib.pylab.clf()

def plotList(data):

  fig = pylab.figure()
  ax = fig.add_subplot(1,1,1)

  nvals = len(data)
  ymax = 0.0

  xaxis = []
  for i in range(nvals):
    xaxis.append(i)
    if (data[i] > ymax):
      ymax = data[i]

  ax.set_ylim(0,ymax)
  ax.set_xlim(0,nvals)
  ax.set_title('Misx Plot')
  ax.set_xlabel('Channel')
  ax.set_ylabel('Activated Bits')
  ax.grid(True)

  ax.plot(xaxis, data, '-')
  ax.axhspan(8, 32, facecolor='grey', alpha=0.5)
  matplotlib.pylab.show()

# creates a figure of the specified size
def createFigure(xdim, ydim):

  fig = matplotlib.figure.Figure(facecolor='black')
  dpi = fig.get_dpi()
  curr_size = fig.get_size_inches()
  xinches = float(xdim) / float(dpi)
  yinches = float(ydim) / float(dpi)
  fig.set_size_inches((xinches, yinches))
  return fig

def set_foregroundcolor(ax, color):
     '''For the specified axes, sets the color of the frame, major ticks,                                                             
         tick labels, axis labels, title and legend                                                                                   
     '''
     for tl in ax.get_xticklines() + ax.get_yticklines():
         tl.set_color(color)
     for spine in ax.spines:
         ax.spines[spine].set_edgecolor(color)
     for tick in ax.xaxis.get_major_ticks():
         tick.label1.set_color(color)
     for tick in ax.yaxis.get_major_ticks():
         tick.label1.set_color(color)
     ax.axes.xaxis.label.set_color(color)
     ax.axes.yaxis.label.set_color(color)
     ax.axes.xaxis.get_offset_text().set_color(color)
     ax.axes.yaxis.get_offset_text().set_color(color)
     ax.axes.title.set_color(color)
     lh = ax.get_legend()
     if lh != None:
         lh.get_title().set_color(color)
         lh.legendPatch.set_edgecolor('none')
         labels = lh.get_texts()
         for lab in labels:
             lab.set_color(color)
     for tl in ax.get_xticklabels():
         tl.set_color(color)
     for tl in ax.get_yticklabels():
         tl.set_color(color)


def set_backgroundcolor(ax, color):
     '''Sets the background color of the current axes (and legend).                                                                   
         Use 'None' (with quotes) for transparent. To get transparent                                                                 
         background on saved figures, use:                                                                                            
         pp.savefig("fig1.svg", transparent=True)                                                                                     
     '''
     ax.patch.set_facecolor(color)
     ax.set_axis_bgcolor(color)
     lh = ax.get_legend()
     if lh != None:
         lh.legendPatch.set_facecolor(color)

def capture(func, *args, **kwargs):
    """Capture the output of func when called with the given arguments.

    The function output includes any exception raised. capture returns
    a tuple of (function result, standard output, standard error).
    """
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = c1 = cStringIO.StringIO()
    sys.stderr = c2 = cStringIO.StringIO()
    result = None
    try:
        result = func(*args, **kwargs)
    except:
        traceback.print_exc()
    sys.stdout = stdout
    sys.stderr = stderr
    return (result, c1.getvalue(), c2.getvalue())



