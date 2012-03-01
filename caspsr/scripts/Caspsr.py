#!/usr/bin/env python26

#
# BPSR python module
#

import Dada, time, corr

#from corr import katcp_wrapper

def getConfig():
    
  config_file = Dada.DADA_ROOT + "/share/caspsr.cfg"
  config = Dada.readCFGFileIntoDict(config_file)
  return config

def convertIPToInteger(ipaddr):
  parts = ipaddr.split(".",3)
  ip_int = (int(parts[0]) * (2**24)) + \
           (int(parts[1]) * (2**16)) + \
           (int(parts[2]) * (2**8)) + \
            int(parts[3])
  return ip_int

###############################################################################
#
# create and configure the roach based on the config
#
def configureRoach(dl, cfg):

  roach_ip   = cfg["ROACH_IP"]
  roach_port = int(cfg["ROACH_PORT"])
  roach_bof  = cfg["ROACH_BOF"]

  roach_10gbe_src_ip_0    = convertIPToInteger(cfg["ROACH_10GbE_SRC_IP_0"])
  roach_10gbe_src_ip_1    = convertIPToInteger(cfg["ROACH_10GbE_SRC_IP_1"])

  Dada.logMsg(2, dl, "configureRoach: " + cfg["ROACH_10GbE_SRC_IP_0"] + " -> " + str(roach_10gbe_src_ip_0))
  Dada.logMsg(2, dl, "configureRoach: " + cfg["ROACH_10GbE_SRC_IP_1"] + " -> " + str(roach_10gbe_src_ip_1))

  roach_10gbe_dest_ip_0   = convertIPToInteger(cfg["ROACH_10GbE_DEST_IP_0"])
  roach_10gbe_dest_ip_1   = convertIPToInteger(cfg["ROACH_10GbE_DEST_IP_1"])

  Dada.logMsg(2, dl, "configureRoach: " + cfg["ROACH_10GbE_DEST_IP_0"] + " -> " + str(roach_10gbe_dest_ip_0))
  Dada.logMsg(2, dl, "configureRoach: " + cfg["ROACH_10GbE_DEST_IP_1"] + " -> " + str(roach_10gbe_dest_ip_1))

  # Same port in src and dest
  roach_10gbe_src_port_0  = int(cfg["DEMUX_UDP_PORT_0"])
  roach_10gbe_src_port_1  = int(cfg["DEMUX_UDP_PORT_1"])

  roach_10gbe_dest_port_0 = int(cfg["DEMUX_UDP_PORT_0"])
  roach_10gbe_dest_port_1 = int(cfg["DEMUX_UDP_PORT_1"])

  # use a designated private MAC address [02:xx:xx:xx:xx:xx]
  roach_10gbe_src_mac_0   = (2<<40) + (2<<32) + roach_10gbe_src_ip_0
  roach_10gbe_src_mac_1   = (2<<40) + (2<<32) + roach_10gbe_src_ip_1

  # FPGA device names
  tengbe_device_0 = cfg["ROACH_10GbE_DEVNAME_0"]
  tengbe_device_1 = cfg["ROACH_10GbE_DEVNAME_1"]

  # something jkocz said
  sync_period = int(cfg["ROACH_SYNC_PERIOD"])

  # connect to ROACH FPGA
  Dada.logMsg(2, dl, "configureRoach: connecting to "+roach_ip+":"+str(roach_port))
  fpga = corr.katcp_wrapper.FpgaClient(roach_ip, roach_port)
  time.sleep(0.5)
  if (fpga.is_connected()):
    Dada.logMsg(2, dl, "configureRoach: connected")
  else:
    Dada.logMsg(-2, dl, "configureRoach: connection failed")
    return ("fail", 0)

  # program bit stream
  Dada.logMsg(1, dl, "programming FPGA with " + roach_bof)
  Dada.logMsg(2, dl, "configureRoach: programming FPGA with " + roach_bof)
  fpga.progdev(roach_bof)
  Dada.logMsg(2, dl, "configureRoach: programming done")

  time.sleep(2.0)

  # start a TGTAP device for both 10GbE ports
  Dada.logMsg(2, dl, "configureRoach: configuing 10GbE device 0: ")
  fpga.tap_start(tengbe_device_0,tengbe_device_0,roach_10gbe_src_mac_0,roach_10gbe_src_ip_0,roach_10gbe_src_port_0)
  time.sleep(0.5)
  gbe0_link = bool(fpga.read_int(tengbe_device_0))
  if gbe0_link:
    Dada.logMsg(2, dl, "configureRoach: 10GbE device 0 now active")
  else:
    Dada.logMsg(-1, dl, "configureRoach: 10GbE device 0 not active")
    
  Dada.logMsg(2, dl, "configureRoach: configuing 10GbE device 1: ")
  fpga.tap_start(tengbe_device_1,tengbe_device_1,roach_10gbe_src_mac_1,roach_10gbe_src_ip_1,roach_10gbe_src_port_1)
  time.sleep(0.5)
  gbe1_link = bool(fpga.read_int(tengbe_device_1))
  if gbe0_link:
    Dada.logMsg(2, dl, "configureRoach: 10GbE device 1 now active")
  else:
    Dada.logMsg(-1, dl, "configureRoach: 10GbE device 1 not active")

 
  fpga.write_int('ip_ctr_reg_num_ips', 2); 
  fpga.write_int('ip_ctr_reg_ip1', roach_10gbe_dest_ip_0); 
  fpga.write_int('ip_ctr_reg_ip2', roach_10gbe_dest_ip_1); 
  fpga.write_int('ip_ctr_reg_port1', roach_10gbe_dest_port_0); 
  fpga.write_int('ip_ctr_reg_port2', roach_10gbe_dest_port_1); 
  fpga.write_int('reg_sync_period', sync_period)
  
  Dada.logMsg(2, dl, "configureRoach: returning ok")
  return ("ok", fpga)

###############################################################################
#
# arm roach to begin start of data
#
def armRoach(dl, fpga):
  fpga.write_int('reg_arm',0)
  fpga.write_int('reg_arm',1)
  return "ok"

