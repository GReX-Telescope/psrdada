#!/usr/bin/env python

import Dada, Bpsr, sys, time, socket, select, signal, traceback
import corr, numpy, math, os

DL = 2

###############################################################################
#
# Globals
#
roach_bof       = 'testge_2011_Dec_11_1829.bof'
tengbe_device   = 'gbe0'
payload_len     = 1026
period          = 2048
# not that the smaller the period the higher the data rate. min period is 1324 -> 9.9 Gbps
#period          = 2652
period          = 1360

def usage():
  sys.stdout.write("Usage: " + sys.argv[0] + " roach_id dest_ip dest_port\n")
  sys.stdout.write("  roach_id    defined in $\DADA_ROOT/share/roach.cfg\n")
  sys.stdout.write("  dest_ip     defined in $\DADA_ROOT/share/roach.cfg\n")

def convertIPToInteger(ipaddr):
  parts = ipaddr.split(".",3)
  ip_int = (int(parts[0]) * (2**24)) + \
           (int(parts[1]) * (2**16)) + \
           (int(parts[2]) * (2**8)) + \
            int(parts[3])
  return ip_int

###############################################################################
#
# Main
#

if len(sys.argv) != 4:
  sys.stderr.write("ERROR: 3 command line arguments required\n")
  usage()
  sys.exit(1)

roach_id    = int(sys.argv[1])
dest_ip_str = sys.argv[2]
dest_port   = int(sys.argv[3])

cfg = Bpsr.getConfig()
roach_cfg = Bpsr.getROACHConfig()
num_roach = int(roach_cfg["NUM_ROACH"])

if (roach_id < 0 or roach_id >= num_roach):
  sys.stderr.write("ERROR: roach_id not in valid range\n")

rid = str(roach_id)

roach_ip    = roach_cfg["ROACH_"+rid]
roach_port  = int(roach_cfg["ROACH_PORT"])

roach_10gbe_src_ip  = Bpsr.convertIPToInteger(roach_cfg["ROACH_10GbE_SRC_"+rid])
roach_10gbe_dest_ip = Bpsr.convertIPToInteger(dest_ip_str)

Dada.logMsg(2, DL, "["+rid+"] " + roach_cfg["ROACH_10GbE_SRC_"+rid] + " -> " + str(roach_10gbe_src_ip))
Dada.logMsg(2, DL, "["+rid+"] " + dest_ip_str + " -> " + str(roach_10gbe_dest_ip))

# same port in src and dest
roach_10gbe_src_port  = dest_port
roach_10gbe_dest_port = dest_port

# use a designated private MAC address [02:xx:xx:xx:xx:xx]
roach_10gbe_src_mac   = (2<<40) + (2<<32) + roach_10gbe_src_ip

connected = False
attempts = 0

Dada.logMsg(2, DL, "["+rid+"] connecting to " + roach_ip + ":" + str(roach_port))

# connect to ROACH FPGA
while (not connected and attempts < 5):

  Dada.logMsg(3, DL, "["+rid+"] connection attempt " + str(attempts) + " for " + roach_ip + ":" + str(roach_port))
  fpga = corr.katcp_wrapper.FpgaClient(roach_ip, roach_port)
  time.sleep(0.1)
  if (fpga.is_connected()):
    Dada.logMsg(3, DL, "["+rid+"] connected to " + roach_ip + ":" + str(roach_port))

  connected = fpga.is_connected()

  if (not connected):
    Dada.logMsg(0, DL, "["+rid+"] connection to " + roach_ip + " failed, retrying")
    time.sleep(1.0)
    attempts += 1

if (not connected):
  Dada.logMsg(-2, DL, "["+rid+"]  connection failed")
  sys.exit(1)

# program bit stream
programmed = False
attempts = 0
while (not programmed and attempts < 5):

  Dada.logMsg(2, DL, "["+rid+"] programming FPGA with " + roach_bof)
  prog_result = fpga.progdev(roach_bof)
  if (prog_result):
    Dada.logMsg(2, DL, "["+rid+"] programming done")
  else:
    Dada.logMsg(0, DL, "["+rid+"] programming FAILED")

  time.sleep(0.1)

  try:
    # try to read from a register
    port = fpga.read_int('dest_port')
    Dada.logMsg(2, DL, "["+rid+"] dest_port = " + str(port))

    # try to write to a register
    port = 8000
    Dada.logMsg(2, DL, "["+rid+"] dest_port " + str(port))
    fpga.write_int('dest_port', port)

    # if we got this far without throwing an exception, we are programmed!
    programmed = True

  except:

    # give it a chance to be ready for commands
    Dada.logMsg(1, DL, "["+rid+"] exception when trying to read/write")
    time.sleep(1.0)
    attempts += 1

if (not programmed):
  Dada.logMsg(0, DL, "["+rid+"] failed to program FPGA")
  sys.exit(1)

null_ip_str = "0.0.0.0"
null_ip   = convertIPToInteger(null_ip_str)
null_port = 0
null_payload_len = payload_len
null_period = 16384

Dada.logMsg(2, DL, "["+rid+"] dest_ip " + null_ip_str)
fpga.write_int('dest_ip', null_ip)

Dada.logMsg(2, DL, "["+rid+"] dest_port " + str(null_port))
fpga.write_int('dest_port', null_port)

Dada.logMsg(2, DL, "["+rid+"] pkt_sim_payload_len " + str(null_payload_len))
fpga.write_int('pkt_sim_payload_len', null_payload_len)

Dada.logMsg(2, DL, "["+rid+"] pkt_sim_period " + str(null_period))
fpga.write_int('pkt_sim_period', null_period)

# start tgtap, which configures the 10Gbe port and begins an ARP process 
Dada.logMsg(2, DL, "["+rid+"] fpga.tap_start()")
fpga.tap_start(tengbe_device,tengbe_device,roach_10gbe_src_mac,roach_10gbe_src_ip,roach_10gbe_src_port)
time.sleep(0.5)
gbe0_link = bool(fpga.read_int(tengbe_device))
if gbe0_link:
  Dada.logMsg(2, DL, "["+rid+"] 10GbE device now active")
else:
  Dada.logMsg(-1, DL, "["+rid+"] 10GbE device NOT active")

# now wait until out destination IP Address has appeared in tgtaps ARP table
parts = dest_ip_str.split(".")
arp_line = "IP: " + parts[0].rjust(3) + "." + parts[1].rjust(3) + "." + parts[2].rjust(3) + "." + parts[3].rjust(3) + ": MAC: FF FF FF FF FF FF "
Dada.logMsg(3, DL, "["+rid+"] arpLine = " + arp_line)

arp_valid = False
attempts = 0
while ((not arp_valid) and attempts < 60):

  arp_table = fpga.return_10gbe_arp_table(tengbe_device)
  arp_valid = True
  lines = arp_table.split("\n")
  for line in lines:
    Dada.logMsg(3, DL, "["+rid+"] testing line '" + line + "' == '" + arp_line + "'")
    if (line == arp_line):
      arp_valid = False

  if (not arp_valid):
    Dada.logMsg(2, DL, "["+rid+"] arp not yet valid")
    time.sleep(1.0)
    attempts += 1

# we have got an IP -> MAC address mapping, and can proceed
if (attempts < 60):

  Dada.logMsg(2, DL, "["+rid+"] dest_ip " + dest_ip_str)
  fpga.write_int('dest_ip', roach_10gbe_dest_ip)

  Dada.logMsg(2, DL, "["+rid+"] dest_port " + str(roach_10gbe_dest_port))
  fpga.write_int('dest_port', roach_10gbe_dest_port)

  Dada.logMsg(2, DL, "["+rid+"] pkt_sim_payload_len " + str(payload_len))
  fpga.write_int('pkt_sim_payload_len', payload_len)

  Dada.logMsg(2, DL, "["+rid+"] pkt_sim_period "  + str(period))
  fpga.write_int('pkt_sim_period', period)

  Dada.logMsg(1, DL, "["+rid+"] programmed UDP -> " + dest_ip_str + ":" + str(roach_10gbe_dest_port))

# the ARP process did not work, fail
else:
  Dada.logMsg(1, DL, "["+rid+"] arp not valid after 60 seconds");
  sys.exit(1)


Dada.logMsg(1, DL, "["+rid+"] sleeping 2 seconds")
time.sleep(2)

Dada.logMsg(1, DL, "["+rid+"] pkt_sim_enable 0")
fpga.write_int('pkt_sim_enable', 0)

Dada.logMsg(1, DL, "["+rid+"] rst 1")
fpga.write_int('rst', 1)

Dada.logMsg(1, DL, "["+rid+"] rst 0")
fpga.write_int('rst', 0)

Dada.logMsg(1, DL, "["+rid+"] pkt_sim_enable 1")
fpga.write_int('pkt_sim_enable', 1)

Dada.logMsg(1, DL, "["+rid+"] DONE!")

sys.exit(0)

