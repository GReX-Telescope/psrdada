#!/usr/bin/env python26

#
# BPSR python module
#

import Dada, struct, math, median_smooth, time, matplotlib, pylab, os
import corr, numpy, fnmatch


def getConfig():
    
  config_file = Dada.DADA_ROOT + "/share/bpsr.cfg"
  config = Dada.readCFGFileIntoDict(config_file)

  pwc_config_file = Dada.DADA_ROOT + "/share/bpsr_pwcs.cfg"
  pwc_config = Dada.readCFGFileIntoDict(pwc_config_file)

  config.update(pwc_config)
  return config


def getROACHConfig():

  roach_config_file = Dada.DADA_ROOT + "/share/roach.cfg"
  roach_config = Dada.readCFGFileIntoDict(roach_config_file)
  return roach_config

def convertIPToInteger(ipaddr):
  parts = ipaddr.split(".",3)
  ip_int = (int(parts[0]) * (2**24)) + \
           (int(parts[1]) * (2**16)) + \
           (int(parts[2]) * (2**8)) + \
            int(parts[3])
  return ip_int

###############################################################################
#
# create and configure the specified ROACH [rid] based on the roach_cfg
#
def configureRoach(dl, acc_len, rid, cfg):

  roach_cfg = getROACHConfig()
  roach_ip   = roach_cfg["ROACH_IP_"+rid]
  roach_port = int(roach_cfg["ROACH_PORT"])
  roach_bof  = roach_cfg["ROACH_BOF"]

  roach_10gbe_src_ip    = convertIPToInteger(roach_cfg["ROACH_10GbE_SRC_IP_"+rid])
  roach_10gbe_dest_ip   = convertIPToInteger(roach_cfg["ROACH_10GbE_DEST_IP_"+rid])

  Dada.logMsg(2, dl, "configureRoach["+rid+"]: " + roach_cfg["ROACH_10GbE_SRC_IP_"+rid] + " -> " + str(roach_10gbe_src_ip))
  Dada.logMsg(2, dl, "configureRoach["+rid+"]: " + roach_cfg["ROACH_10GbE_DEST_IP_"+rid] + " -> " + str(roach_10gbe_dest_ip))

  # Same port in src and dest
  roach_10gbe_src_port  = int(cfg["CLIENT_UDPDB_PORT"])
  roach_10gbe_dest_port = int(cfg["CLIENT_UDPDB_PORT"])

  # use a designated private MAC address [02:xx:xx:xx:xx:xx]
  roach_10gbe_src_mac   = (2<<40) + (2<<32) + roach_10gbe_src_ip

  sync_period = 100 * acc_len * 2048
  tengbe_device = roach_cfg["ROACH_10GBE_DEVNAME"]

  # connect to ROACH FPGA
  Dada.logMsg(2, dl, "configureRoach["+rid+"]: connecting to "+roach_ip+":"+str(roach_port))
  fpga = corr.katcp_wrapper.FpgaClient(roach_ip, roach_port)
  time.sleep(0.5)
  if (fpga.is_connected()):
    Dada.logMsg(2, dl, "configureRoach["+rid+"]: connected")
  else:
    Dada.logMsg(-2, dl, "configureRoach["+rid+"]: connection failed")
    return ("fail", 0)

  # program bit stream
  Dada.logMsg(1, dl, "programming FPGA with " + roach_bof)
  Dada.logMsg(2, dl, "configureRoach["+rid+"]: programming FPGA with " + roach_bof)
  fpga.progdev(roach_bof)
  Dada.logMsg(2, dl, "configureRoach["+rid+"]: programming done")

  time.sleep(2.0)
  fpga.tap_start(tengbe_device,tengbe_device,roach_10gbe_src_mac,roach_10gbe_src_ip,roach_10gbe_src_port)
  time.sleep(0.5)
  gbe0_link = bool(fpga.read_int(tengbe_device))
  if gbe0_link:
    Dada.logMsg(2, dl, "configureRoach["+rid+"]: 10GbE device now active")
  else:
    Dada.logMsg(-1, dl, "configureRoach["+rid+"]: 10GbE device not active")
    
  # configure 10 GbE device
  Dada.logMsg(2, dl, "configureRoach["+rid+"]: configuring 10GbE")
  fpga.write_int('reg_ip', roach_10gbe_dest_ip)
  fpga.write_int('reg_10GbE_destport0', roach_10gbe_dest_port)
  Dada.logMsg(1, dl, "configureRoach["+rid+"]: acc_len <- " + str(acc_len-1))
  fpga.write_int('reg_acclen', acc_len-1)
  Dada.logMsg(1, dl, "configureRoach["+rid+"]: sync_period <- " + str(sync_period))
  fpga.write_int('reg_sync_period', sync_period)

  fpga.write_int('reg_coeff_pol1', 16384)
  fpga.write_int('reg_coeff_pol2', 16384)


  Dada.logMsg(2, dl, "configureRoach["+rid+"]: returning ok")
  return ("ok", fpga)

###############################################################################
#
# arm roach to begin start of data
#
def armRoach(dl, fpga):
  fpga.write_int('reg_arm',0)
  fpga.write_int('reg_arm',1)
  return "ok"

def setLevels (dl, fpga):

  n_attempts = 3
  
  # intial guess at gain values
  pol1_coeff = fpga.read_int('reg_coeff_pol1')
  pol2_coeff = fpga.read_int('reg_coeff_pol2')
  bit_window = 0

  for i in range (n_attempts):

    fpga.write_int('reg_coeff_pol1', pol1_coeff)
    fpga.write_int('reg_coeff_pol2', pol2_coeff)

    time.sleep(0.1)

    Dada.logMsg(2, dl, "setLevels: setting coeffs ["+str(pol1_coeff)+", "+str(pol2_coeff)+"]")

    pol1, pol2 = bramdumpRoach(dl, fpga, 1)

    # perform median smoothing on data to remove spikes
    smoothed_pol1 = median_smooth.smoothList(pol1, 11, 'flat')
    smoothed_pol2 = median_smooth.smoothList(pol2, 11, 'flat')

    # get the maximum value in each list, excluding certain channels...
    pol1_max = calculate_max(smoothed_pol1)
    pol2_max = calculate_max(smoothed_pol2)

    Dada.logMsg(2, dl, 'setLevels: pol1_max='+str(pol1_max)+' pol2_max='+str(pol2_max))

    # find the right bit window and coeffs
    bit_window, pol1_coeff, pol2_coeff = findBitWindow(dl, pol1_max, pol2_max, pol1_coeff, pol2_coeff)

    Dada.logMsg(2, dl, 'setLevels: bit_window='+str(bit_window)+' pol1_coeff='+str(pol1_coeff)+' pol2_coeff='+str(pol2_coeff))

  # now get a more sensitive estimate
  pol1, pol2 = bramdumpRoach(dl, fpga, 32)

  smoothed_pol1 = median_smooth.smoothList(pol1, 11, 'flat')
  smoothed_pol2 = median_smooth.smoothList(pol2, 11, 'flat')

  # get the maximum value in each list, excluding certain channels...
  pol1_max = calculate_max(smoothed_pol1)
  pol2_max = calculate_max(smoothed_pol2)

  Dada.logMsg(2, dl, 'setLevels: pol1_max='+str(pol1_max)+' pol2_max='+str(pol2_max))

  # find the right bit window and coeffs
  bit_window, pol1_coeff, pol2_coeff = findBitWindow(dl, pol1_max, pol2_max, pol1_coeff, pol2_coeff)

  Dada.logMsg(2, dl, 'setLevels: bit_window='+str(bit_window)+' pol1_coeff='+str(pol1_coeff)+' pol2_coeff='+str(pol2_coeff))

  # plotLog2List(bit_window, pol1, pol2)

  return "ok"


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
# create 3 plot files from a bramdump of the ROACH board
#
def bramplotRoach(dl, fpga, timestamp, roach_name):

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

  ymin = 0
  ymax = 32
  xmin = 0
  xmax = 511

  xres_list = [1024, 400,   112]
  yres_list = [768,  300,   84]
  pp_list   = [True, True, False]

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
      #ax.set_axis_off()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

    ax.grid(False)

    # plot pol1 as red, pol2 as greeb
    ax.plot(xaxis, pol1_float, 'r-', label='pol 1')
    ax.plot(xaxis, pol2_float, 'g-', label='pol 2')

    if (pp):
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

    dpi = fig.get_dpi()
    Dada.logMsg(2, dl, "bramplotRoach: creating " + filename)
    fig.savefig(filename, dpi=(dpi), facecolor='black')

    # now cleanup any excess plotting files
    filematch = "*"+roach_name + "_" + str(xres) + "x" + str(yres) + ".png"
    Dada.logMsg(2, dl, "bramplotRoach: checking " + os.getcwd() + " for matching " + filematch)
    filelist = os.listdir(os.getcwd())
    filelist.sort(reverse=True)
    count = 0
    for filename in filelist:
      if (fnmatch.fnmatch(filename, filematch)):
        count += 1
        if (count > 3):
          Dada.logMsg(2, dl, "bramplotRoach: cleaning file: "+filename)
          os.remove(filename)

    matplotlib.pylab.clf()

  Dada.logMsg(2, dl, "bramplotRoach: plotting complete");
  return "ok"

def calculate_max (data):
  maxval = 0
  for i, value in enumerate(data):
    if ((i<430) or (i > 480)):
      if (value > maxval):
        maxval = value

  return maxval

def findBitWindow(dl, pol1_max, pol2_max, gain1, gain2):

  bit_window = 0

  bitsel_min = [0, 256, 65535, 16777216]
  bitsel_mid = [64, 8192, 2097152, 536870912]
  bitsel_max = [255, 65535, 16777215, 4294967295]

  val = (pol1_max + pol2_max) / 2.0

  Dada.logMsg(2, dl, "findBitWindow: max value = "+str(val))
  Dada.logMsg(2, dl, "findBitWindow: initial gains ["+str(gain1)+", "+str(gain2)+"]")

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

  Dada.logMsg(2, dl, "findBitWindow: current_window="+str(current_window)+" desired_window="+str(desired_window))

  if ((pol1_max == 0) or (pol2_max == 0)):
    pol1_max = 1
    pol2_max = 1

  # If current gains are too high (2^16) then shift down a window
  if ((gain1 > 32000) or (gain2 > 32000)):
    if (desired_window > 0):
      desired_window -= 1

  # If current gains are too low (2^3) then shift down a window
  if ((gain1 < 8) or (gain2 < 8)):
    if (desired_window < 3):
      desired_window += 1

  desired_val = ((float((bitsel_max[desired_window]+1)) / 4.0))

  Dada.logMsg(2, dl, "findBitWindow: desired_val="+str(desired_val))

  gain_factor1 = desired_val / float(pol1_max)
  gain_factor2 = desired_val / float(pol2_max)

  Dada.logMsg(2, dl, "findBitWindow: 1 new gains ["+str(gain_factor1)+", "+str(gain_factor2)+"]")

  gain_factor1 = math.sqrt(gain_factor1)
  gain_factor2 = math.sqrt(gain_factor2)

  Dada.logMsg(2, dl, "findBitWindow: 2 new gains ["+str(gain_factor1)+", "+str(gain_factor2)+"]")

  gain_factor1 *= gain1
  gain_factor2 *= gain2

  Dada.logMsg(2, dl, "findBitWindow: 3 new gains ["+str(gain_factor1)+", "+str(gain_factor2)+"]")

  gain1 = int(gain_factor1)
  gain2 = int(gain_factor2)

  Dada.logMsg(2, dl, "findBitWindow: gains "+str(gain_factor1)+" -> "+str(gain1))
  Dada.logMsg(2, dl, "findBitWindow: gains "+str(gain_factor2)+" -> "+str(gain2))

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

  fig = pylab.figure(1, facecolor='black')
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



