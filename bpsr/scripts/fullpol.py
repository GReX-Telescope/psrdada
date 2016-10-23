#!/usr/bin/env python

# Full Polarization Spectrometer

import corr,time,numpy,struct,sys,matplotlib, pylab,os,math
import Dada,Bpsr
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def programRoach (dl, fpga, rid, boffile='default'):

  roach_cfg  = Bpsr.getROACHConfig()
  roach_beam = roach_cfg["BEAM_"+rid]
  roach_bof  = 'parspec_fullpol_2013_Sep_13_1726.bof'
  roach_bof  = 'parspec_fullpol_2015_Mar_30_2045.bof'
  if (not boffile == 'default'):
    roach_bof = boffile

  # program bit stream
  programmed = False
  attempts = 0
  while (not programmed and attempts < 10):

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
      port = fpga.read_int('reg_10GbE_destport0')
      Dada.logMsg(3, dl, "["+rid+"] reg_10GbE_destport0 = " + str(port))

      # try to write to a register
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
    Dada.logMsg(0, dl, "["+rid+"] failed to program FPGA")
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
  while ((not arp_valid) and attempts < 300):

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

  if (attempts < 300):
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
  tengbe_device = "ten_Gbe_v2"
  # linux device
  tengbe_tapdev = "ten_Gbe"

  # ensure 10GbE output is disabled
  enable = 1
  Dada.logMsg(2, dl, "["+rid+"] enable " + str(enable))
  fpga.write_int('enable', enable)

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

def initializeRoach(dl, acc_len, rid, cfg):

  result, fpga = Bpsr.connectRoach(dl, rid)
  if (not result == "ok"):
    Dada.logMsg(0, dl, "["+rid+"] initializeRoach: could not connect to ROACH")
    return ("fail", 0)

  result, response = programRoach(dl, fpga, rid)
  if (not result == "ok"):
    Dada.logMsg(0, dl, "["+rid+"] initializeRoach: could not program ROACH")
    return ("fail", 0)

  result, response = configureRoach(dl, fpga, rid, cfg)
  if (not result == "ok"):
    Dada.logMsg(0, dl, "["+rid+"] initializeRoach: could not configure ROACH")
    return ("fail", fpga)

  result, response = accLenRoach(dl, fpga, acc_len, rid)
  if (not result == "ok"):
    Dada.logMsg(0, dl, "["+rid+"] initializeRoach: could not set acc_len on roach")
    return ("fail", fpga)

  result, response = levelSetRoach(dl, fpga, rid)
  if (not result == "ok"):
    Dada.logMsg(0, dl, "["+rid+"] initializeRoach: could not set levels on ROACH")
    return ("fail", fpga)

  return ("ok", fpga)


def setGains(dl, fpga, rid, value):

  Dada.logMsg(1, dl, "["+rid+"]  setting gain levels to="+str(value))
  fpga.write_int('reg_coeff_pol1', value)
  fpga.write_int('reg_coeff_pol2', value)
  return ("ok", "")

def setComplexGains(dl, fpga, rid, value, bit_window):

  Dada.logMsg(1, dl, "["+rid+"]  setting complex gain levels to="+str(value)+" window="+str(bit_window))
  fpga.write_int('reg_output_bitselectR', bit_window)
  fpga.write_int('reg_output_bitselectC', bit_window)
  fpga.write_int('reg_coeff_XYI1', value)
  fpga.write_int('reg_coeff_XYI2', value)
  fpga.write_int('reg_coeff_XYR1', value)
  fpga.write_int('reg_coeff_XYR2', value)
  return ("ok", "")

#
#  need to set gains for the PP, QQ and the XPOL
#
def setLevels (dl, fpga, pol1, pol2, rid):

  Dada.logMsg(3, dl, "setLevels: levelSetRoach(pol1="+str(pol1)+" pol2="+str(pol2)+")")
  result, ppqq_response = levelSetRoach(dl, fpga, pol1, pol2, rid)
  Dada.logMsg(3, dl, "setLevels: " + result + " " + ppqq_response)

  Dada.logMsg(2, dl, "setLevels: crossLevelSetRoach()")
  result, xpol_response = crossLevelSetRoach(dl, fpga, rid)
  Dada.logMsg(3, dl, "setLevels: " + result + " " + xpol_response)

  return ("ok", ppqq_response + "," + xpol_response)
  

def levelSetRoach(dl, fpga, pol1_flag, pol2_flag, rid):

  n_attempts = 3

  # start with sensible first guess, need high gain due to cross poln 3rd window requirement
  pol1_coeff = 110000
  pol2_coeff = 110000
  total_power_bit_window = 1

  if not pol1_flag:
    pol1_coeff = 0
  if not pol2_flag:
    pol2_coeff = 0

  for i in range (n_attempts):

    fpga.write_int('reg_output_bitselect', total_power_bit_window)
    fpga.write_int('reg_coeff_pol1', pol1_coeff)
    fpga.write_int('reg_coeff_pol2', pol2_coeff)

    time.sleep(0.01)

    pol1 = dumpRoach(dl, fpga, 'scope_pol1_Shared_BRAM', 'I', 1)
    pol2 = dumpRoach(dl, fpga, 'scope_pol2_Shared_BRAM', 'I', 1)

    # do not count the first 150 channels out of 1024
    pol1_med = numpy.median( numpy.array(pol1)[75:] )
    pol2_med = numpy.median( numpy.array(pol2)[75:] )

    if not pol1_flag:
      pol1_med = -1
    if not pol2_flag:
      pol2_med = -1

    Dada.logMsg(2, dl, '['+rid+'] levelSetRoach: pol1_med='+str(pol1_med)+' pol2_med='+str(pol2_med))

    # find the right bit window and coeffs
    total_power_bit_window, pol1_coeff, pol2_coeff = findBitWindow(dl, pol1_med, pol2_med, pol1_coeff, pol2_coeff)

    Dada.logMsg(2, dl, '['+rid+'] levelSetRoach: bit_window='+str(total_power_bit_window)+' pol1_coeff='+str(pol1_coeff)+' pol2_coeff='+str(pol2_coeff))

    if not pol1_flag:
      pol1_coeff = 0
    if not pol2_flag:
      pol2_coeff = 0

  # now get a more sensitive estimate
  pol1 = dumpRoach(dl, fpga, 'scope_pol1_Shared_BRAM', 'I', 16)
  pol2 = dumpRoach(dl, fpga, 'scope_pol2_Shared_BRAM', 'I', 16)

  # do not count the first 150 channels out of 1024
  pol1_med = numpy.median( numpy.array(pol1)[75:] )
  pol2_med = numpy.median( numpy.array(pol2)[75:] )

  Dada.logMsg(2, dl, '['+rid+'] levelSetRoach: pol1_med='+str(pol1_med)+' pol2_med='+str(pol2_med))

  # find the right bit window and coeffs
  total_power_bit_window, pol1_coeff, pol2_coeff = findBitWindow(dl, pol1_med, pol2_med, pol1_coeff, pol2_coeff)

  Dada.logMsg(2, dl, '['+rid+'] levelSetRoach: bit_window='+str(total_power_bit_window)+' pol1_coeff='+str(pol1_coeff)+' pol2_coeff='+str(pol2_coeff))

  fpga.write_int('reg_output_bitselect', total_power_bit_window)
  fpga.write_int('reg_coeff_pol1', pol1_coeff)
  fpga.write_int('reg_coeff_pol2', pol2_coeff)

  Dada.logMsg(2, dl, '['+rid+'] bit_window='+str(total_power_bit_window)+' pol1='+str(pol1_coeff)+' pol2='+str(pol2_coeff))

  Dada.logMsg(2, dl, '['+rid+'] levelSetRoach: pol1='+str(pol1_coeff)+',pol2='+str(pol2_coeff) + ",bit_window="+str(total_power_bit_window))

  return ("ok", "pol1="+str(pol1_coeff)+",pol2="+str(pol2_coeff)+",bw="+str(total_power_bit_window))

def crossLevelSetRoach(dl, fpga, rid):

  n_attempts = 10

  # this is constant and a requirement due to two's complement representation
  # of the 32-bit numbers inside ROACH prior to bit selection
  bit_window = 3
  fpga.write_int('reg_output_bitselectR', bit_window)
  fpga.write_int('reg_output_bitselectC', bit_window)

  gain = 128

  for i in range (n_attempts):

    Dada.logMsg(3, dl, "["+rid+"] crossLevelSetRoach: setting all cross gains=" + str(gain)) 

    fpga.write_int('reg_coeff_XYI1', gain)
    fpga.write_int('reg_coeff_XYI2', gain)
    fpga.write_int('reg_coeff_XYR1', gain)
    fpga.write_int('reg_coeff_XYR2', gain)

    time.sleep(0.01)

    # get the levels from the BRAMS for all cross pols
    r1 = dumpRoach(dl, fpga, 'scope_crossR1_Shared_BRAM', 'i', 1)
    r2 = dumpRoach(dl, fpga, 'scope_crossR2_Shared_BRAM', 'i', 1)
    c1 = dumpRoach(dl, fpga, 'scope_crossC1_Shared_BRAM', 'i', 1)
    c2 = dumpRoach(dl, fpga, 'scope_crossC2_Shared_BRAM', 'i', 1)

    r = (r1 + r2) / 2 
    c = (c1 + c2) / 2

    # get the median absolute value for each
    r_med = numpy.median( numpy.absolute(numpy.array(r)[80:] ))
    c_med = numpy.median( numpy.absolute(numpy.array(c)[80:] ))

    Dada.logMsg(3, dl, "["+rid+"] crossLevelSetRoach: r_med="+str(r_med)+" c_med="+str(c_med))

    # average the median between the 2
    avg_med = (r_med + c_med) / 2
    Dada.logMsg(3, dl, "["+rid+"] crossLevelSetRoach: avg_med="+str(avg_med))

    # we want to be averaging around the 6th bit in 3rd bit window [2^(31-2)]
    # the previuos value used here - was 2 ** 29
    desired_val = 2 ** 29
    desired_val = 1.5 * (2 ** 27)
    Dada.logMsg(3, dl, "["+rid+"] crossLevelSetRoach: desired_val="+str(desired_val))

    # Jkocz advised all cross poln scale factors should be same
    if avg_med == 0:
      avg_med = 1
    gain_factor = desired_val / (float(avg_med))
    Dada.logMsg(3, dl, "["+rid+"] crossLevelSetRoach: [1] gain_factor="+str(gain_factor))
    gain_factor *= gain
    Dada.logMsg(3, dl, "["+rid+"] crossLevelSetRoach: [2] gain_factor="+str(gain_factor))

    gain = int(gain_factor)
    Dada.logMsg(3, dl, "["+rid+"] crossLevelSetRoach: gains " + str(gain_factor) + " -> " + str(gain))

    if (gain > 262143):
      gain = 262143

  return ("ok", "polx="+str(gain)+",bwx="+str(bit_window))

def findBitWindow(dl, pol1_med, pol2_med, gain1, gain2):

  bit_window = 1

  bitsel_min = [0, 256, 65535, 16777216]
  bitsel_mid = [64, 8192, 2097152, 536870912]
  bitsel_max = [255, 65535, 16777215, 4294967295]

  val = 0
  count = 0
  if pol1_med >= 0:
    val += pol1_med
    count += 1
    pold1_med = 0
  if pol2_med >= 0:
    val += pol2_med
    count +=1

  if count > 0:
    val /= count;

  # val = (pol1_med + pol2_med) / 2.0

  Dada.logMsg(3, dl, "findBitWindow: max value = "+str(val))
  Dada.logMsg(3, dl, "findBitWindow: initial gains ["+str(gain1)+", "+str(gain2)+"]")

  current_window = 1
  desired_window = 1
  minimum_window = 1

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

  if (desired_window < minimum_window):
    desired_window = minimum_window

  Dada.logMsg(3, dl, "findBitWindow: current_window="+str(current_window)+" desired_window="+str(desired_window))

  if pol1_med <= 0:
    pol1_med = 1
  if pol2_med <= 0:
    pol2_med = 1

  Dada.logMsg(2, dl, "findBitWindow: current_window="+str(current_window)+" desired_window="+str(desired_window))

  desired_val = ((float((bitsel_max[desired_window]+1)) / 4.0))

  Dada.logMsg(2, dl, "findBitWindow: desired_val="+str(desired_val))

  gain_factor1 = desired_val / float(pol1_med)
  gain_factor2 = desired_val / float(pol2_med)

  Dada.logMsg(2, dl, "findBitWindow: 1 new gains ["+str(gain_factor1)+", "+str(gain_factor2)+"]")

  gain_factor1 = math.sqrt(gain_factor1)
  gain_factor2 = math.sqrt(gain_factor2)

  Dada.logMsg(2, dl, "findBitWindow: 2 new gains ["+str(gain_factor1)+", "+str(gain_factor2)+"]")

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


def bramdumpRoach(dl, fpga, n_dumps=1):

  pol1 = dumpRoach (dl, fpga, "scope_pol1_Shared_BRAM", "I", n_dumps)
  pol2 = dumpRoach (dl, fpga, "scope_pol2_Shared_BRAM", "I", n_dumps)
  return pol1, pol2

def bramdumpRoachCross(dl, fpga, n_dumps=1):

  # get the levels from the BRAMS for all cross pols
  r1 = dumpRoach(dl, fpga, 'scope_crossR1_Shared_BRAM', 'i', n_dumps)
  r2 = dumpRoach(dl, fpga, 'scope_crossR2_Shared_BRAM', 'i', n_dumps)
  c1 = dumpRoach(dl, fpga, 'scope_crossC1_Shared_BRAM', 'i', n_dumps)
  c2 = dumpRoach(dl, fpga, 'scope_crossC2_Shared_BRAM', 'i', n_dumps)
  
  r = (r1 + r2)/2
  c = (c1 + c2)/2

  return r, c

#
# dump the BRAM contents to a file for asynchronous plotting
#
def bramdiskRoach(dl, fpga, timestamp, roach_name):

  Dada.logMsg(2, dl, "[" + roach_name + "] bramdiskRoach: dumping BRAM");

  file_prefix = timestamp + "_" + roach_name
  fpga.write_int ('snapshot1_ctrl', 1)
  fpga.write_int ('snapshot_ctrl', 1)
  fpga.write_int ('adc_snap', 1)

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

  r, c = bramdumpRoachCross (dl, fpga, 1)

  bram_file_cross = file_prefix + ".bram_cross"
  fptr = open(bram_file_cross, "wb")

  # always must be in bit window 3
  bit_window_bin = struct.pack("I1",3)
  r_binary = struct.pack("512f",*r)
  c_binary = struct.pack("512f",*c)

  fptr.write(bit_window_bin)
  fptr.write(r_binary)
  fptr.write(c_binary)
  fptr.close()

  # now form some histograms
  a = dumpADCRoach(dl, fpga, 'snapshot_bram')
  b = dumpADCRoach(dl, fpga, 'snapshot1_bram')

  # disable these for next time
  fpga.write_int ('snapshot_ctrl', 0)
  fpga.write_int ('snapshot1_ctrl', 0)
  fpga.write_int ('adc_snap', 0)

  hist_a, bin_edges = numpy.histogram (a, bins=256, range=(-128,127))
  hist_b, bin_edges = numpy.histogram (b, bins=256, range=(-128,127))

  bram_file_hist = file_prefix + ".bram_hist"
  fptr = open(bram_file_hist, "wb")
  hist_a.astype(numpy.uint32).tofile(fptr)
  hist_b.astype(numpy.uint32).tofile(fptr)
  fptr.close()

  return 'ok'

def dumpRoach(dl, fpga, reg, unpack_type, n_dumps=1):

  data = numpy.array(struct.unpack('>512'+unpack_type,fpga.read(reg,512*4,0)))

  if (n_dumps <= 1):
    return data

  for i in range(n_dumps-1):
    tmp1 = numpy.array(struct.unpack('>512'+unpack_type,fpga.read(reg,512*4,0)))
    for j in range(512):
      data[j] = data[j] + tmp1[j]

  for i in range(512):
    data[i] = int(float(data[i]) / float(n_dumps))

  return data

def dumpADCRoach (dl, fpga, reg, n_dumps=1):

  data = numpy.array(struct.unpack('>4096b', fpga.read(reg,1024*4,0)))

  if (n_dumps <= 1):
    return data

  for i in range(n_dumps-1):
    new = numpy.array(struct.unpack('>4096b', fpga.read(reg,1024*4,0)))
    data = numpy.concatenate ((data, new), axis=0)

  return data

def formHistograms (A, B):

  histA = numpy.histogram(A, bins=256, range=(-128,127))
  histB = numpy.histogram(B, bins=256, range=(-128,127))

  return histA, histB

def plotHistograms (A, B, nbin):

  fig = pylab.figure()
  ax = fig.add_subplot(1,1,1)

  custombins=numpy.arange(-128, 128, 1)
  n, bins, patches = ax.hist (A, custombins, histtype='step')
  n, bins, patches = ax.hist (B, custombins, histtype='step')

  matplotlib.pylab.show()

def plot (y1,y2,bit_window,log,cross):

  fig = pylab.figure()
  ax = fig.add_subplot(1,1,1)

  nvals = len(y1)
  ymin = 100000.0
  ymax = -100000.0

  Dada.logMsg(1, 1, "plot: nvals=" + str(nvals))

  # +ve / -ve signs
  y1_sign = y1 < 0
  y2_sign = y2 < 0

  f1 = numpy.absolute(y1)
  f2 = numpy.absolute(y2)

  xaxis = []
  for i in range(nvals):
    xaxis.append(i)

  if log:
    for i in range(nvals):
      if f1[i] == 0:
        f1[i] = 1
      if f2[i] == 0:
        f2[i] = 1
    f1 = numpy.log2(f1) 
    f2 = numpy.log2(f2) 

  mult_y1 = numpy.ones(nvals)
  mult_y2 = numpy.ones(nvals)
  for i in range(nvals):
    if y1_sign[i]:
      mult_y1[i] = -1
    if y2_sign[i]:
      mult_y2[i] = -1

  f1 = numpy.multiply(f1, mult_y1)
  f2 = numpy.multiply(f2, mult_y2)

  if cross:
    f1 = f1 / 2
    f2 = f2 / 2

  ymin = 0
  for i in range(nvals):
    if (f1[i] < ymin):
      ymin = f1[i]
    if (f2[i] < ymin):
      ymin = f2[i]
    if (f1[i] > ymax):
      ymax = f1[i]
    if (f2[i] > ymax):
      ymax = f2[i]

  if log and cross:
    ymin = -16
    ymax = 16
  if log and not cross:
    ymin = 0 
    ymax = 32

  ax.set_ylim(ymin,ymax)
  ax.set_xlim(0,nvals)
  ax.set_title('Misc Plot')
  ax.set_xlabel('Channel')
  ax.set_ylabel('Activated Bits')
  ax.grid(True)

  if log and not cross:
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

  if log and cross:
    if (bit_window == 0):
      ax.axhspan(4, 8, facecolor='grey', alpha=0.75)
      ax.axhspan(-4, -8, facecolor='grey', alpha=0.75)
    elif (bit_window == 1):
      ax.axhspan(-4, 4, facecolor='grey', alpha=0.75)
      ax.axhspan(-8, -16, facecolor='grey', alpha=0.75)
      ax.axhspan(8, 16, facecolor='grey', alpha=0.75)
    elif (bit_window == 2):
      ax.axhspan(-8, 8, facecolor='grey', alpha=0.75)
      ax.axhspan(-12, -16, facecolor='grey', alpha=0.75)
      ax.axhspan(12, 16, facecolor='grey', alpha=0.75)
    else:
      ax.axhspan(-12, 12, facecolor='grey', alpha=0.75)

  print "ylimits = [" + str(ymin) + ", " + str(ymax) + "]"

  ax.plot(xaxis, f1, '-')
  ax.plot(xaxis, f2, '-')
  matplotlib.pylab.show()

def plotLog2List(bit_window, pol1, pol2):

  nvals = len(pol1)
  xaxis = []
  pol1_float = []
  pol2_float = []

  Dada.logMsg(1, 1, "pol1")
  print pol1
  Dada.logMsg(1, 1, "pol2")
  print pol2

  for i in range(nvals):
    xaxis.append(i)
    pol1_float.append(pol1[i])
    pol2_float.append(pol2[i])

  ymin = numpy.amin(numpy.minimum(pol1_float, pol2_float))
  ymax = numpy.amax(numpy.maximum(pol1_float, pol2_float))
  if ymin == ymax:
    ymin = ymin - 1
    ymax = ymax + 1

  Dada.logMsg(1, 1, "ymin="+str(ymin)+" ymax="+str(ymax))

  xmin = 0
  xmax = 511

  xres = 768
  yres = 1024

  fig = matplotlib.figure.Figure(facecolor='black')
  dpi = fig.get_dpi()
  ax = []

  # set resolution
  xinches = float(xres) / float(dpi)
  yinches = float(yres) / float(dpi)
  fig.set_size_inches((xinches, yinches))

  ax = fig.add_subplot(1,1,1)
  set_foregroundcolor(ax, 'white')
  set_backgroundcolor(ax, 'black')

  ax.set_title('Full Pol Test')
  ax.set_xlabel('Channel')
  ax.set_ylabel('Power')

  ax.grid(False)

  # plot pol1 as red, pol2 as greeb
  ax.plot(xaxis, pol1_float, 'r-', label='pol 1')
  ax.plot(xaxis, pol2_float, 'g-', label='pol 2')

  # add a legend
  ax.legend()

  # hard set the x,y limits
  ax.set_xlim((xmin, xmax))
  ax.set_ylim((ymin, ymax))

  Dada.logMsg(1, 1, "ax.show")
  matplotlib.pyplot.show()

  Dada.logMsg(1, 1, "pylab done")

def set_foregroundcolor(ax, color):
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
     ax.patch.set_facecolor(color)
     ax.set_axis_bgcolor(color)
     lh = ax.get_legend()
     if lh != None:
         lh.legendPatch.set_facecolor(color)

def stopTX (dl, fpga):
  if (fpga != []):
    #fpga.write_int('enable', 1)
    return "ok"
  else:
    return "fail"


def startTX (dl, fpga):
  if (fpga != []):
    #fpga.write_int('enable', 0)
    return "ok"
  else:
    return "fail"

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

