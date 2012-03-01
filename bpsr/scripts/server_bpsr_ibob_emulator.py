#!/usr/bin/python
#
# Filename: server_bpsr_ibob_emulator.py
#
#   * masqueradges a roach as an ibob, providing a "tinyshell" interface

import Dada, Bpsr, threading, sys, time, socket, select, signal, traceback
import corr, time, numpy, math, os, struct, datetime

PIDFILE  = "bpsr_ibob_emulator.pid"
LOGFILE  = "bpsr_ibob_emulator.log"
QUITFILE = "bpsr_ibob_emulator.quit"
DL = 2

QUIT = 1
IDLE = 2
ERROR = 3
TASK_STATE = 4
TASK_CONFIG = 5
TASK_ARM = 6
TASK_SET_LEVELS = 7
TASK_BRAMPLOT = 8

###########################################################################

def signal_handler(signal, frame):
  print 'You pressed Ctrl+C!'
  global quit_event
  quit_event.set()
  
def send_greeting(sock):
  sock.send("Design name : parkes_200mhz_FINAL_lwip\r\n")
  sock.send("Compiled on : 01-Apr-2008 20:41:09\r\n")
  sock.send("\r\n")
  sock.send("\r\n")

def send_prompt(sock):
  sock.send("IBOB % ")

def send_reply(sock, message, terminator):
  Dada.logMsg(2, DL, "-> " + message)
  sock.send(message + terminator)

def list_dev(sock):
  sock.send("Address map :\r\n");
  sock.send("        <NO ADDR>  -> XSG core config\r\n");
  sock.send("        <NO ADDR>  -> adc\r\n");
  sock.send("        0xD0000000 -> ibob_lwip/ethlite\r\n");
  sock.send("        <NO ADDR>  -> ibob_lwip/lwip\r\n");
  sock.send("        0xD0002000 -> ibob_lwip/macbits\r\n");
  sock.send("        <NO ADDR>  -> ibob_lwip/machdr\r\n");
  sock.send("        <NO ADDR>  -> led0_10gbe_up\r\n");
  sock.send("        <NO ADDR>  -> led1_10gbe_tx\r\n");
  sock.send("        <NO ADDR>  -> led2_pps\r\n");
  sock.send("        0xD0002100 -> reg_10GbE_destport0\r\n");
  sock.send("        0xD0002200 -> reg_acclen\r\n");
  sock.send("        0xD0002300 -> reg_adcscope_1\r\n");
  sock.send("        0xD0002400 -> reg_adcscope_2\r\n");
  sock.send("        0xD0002500 -> reg_arm\r\n");
  sock.send("        0xD0002600 -> reg_coeff_pol1\r\n");
  sock.send("        0xD0002700 -> reg_coeff_pol2\r\n");
  sock.send("        0xD0002800 -> reg_ip\r\n");
  sock.send("        0xD0002900 -> reg_output_bitselect\r\n");
  sock.send("        0xD0002A00 -> reg_sync_period\r\n");
  sock.send("        0xD0004000 -> scope_output1/bram\r\n");
  sock.send("        0xD1000000 -> scope_output3/bram\r\n");
  sock.send("        0x40000000 -> ten_GbE0\r\n");


############################################################################### 
#
# main
#

try:

  # get the BPSR configuration
  cfg = Bpsr.getConfig()
  roach_cfg = Bpsr.getROACHConfig()

  log_file = cfg["SERVER_LOG_DIR"] + "/" + LOGFILE;
  pid_file = cfg["SERVER_CONTROL_DIR"] + "/" + PIDFILE;
  quit_file = cfg["SERVER_CONTROL_DIR"] + "/"  + QUITFILE;
  quit_event = threading.Event()

  signal.signal(signal.SIGINT, signal_handler)

  # start a control thread to handle quit requests
  control_thread = Dada.controlThread(quit_file, pid_file, quit_event, DL);
  control_thread.start()

  if (len(sys.argv) != 4):
    Dada.logMsg(-1, DL, "expecting 3 command line arguements")
    sys.exit(1)

  listen_ip   = sys.argv[1]
  listen_port = sys.argv[2]
  roach       = sys.argv[3]

  # get the roach ID from the roach_config
  n_roach = int(roach_cfg["NUM_ROACH"])
  i_roach = -1
  for i in range(n_roach):
    if (roach_cfg["ROACH_IP_"+str(i)] == roach):
      i_roach = i

  if (i_roach == -1):
    Dada.logMsg(-1, DL, "could not find a roach that matched '"+roach+"'")
    sys.exit(1)

  rid = str(i_roach)

  roach_ip   = roach_cfg["ROACH_IP_"+rid]
  roach_port = int(roach_cfg["ROACH_PORT"])
  roach_bof  = roach_cfg["ROACH_BOF"]

  fpga = []

  # connect to ROACH FPGA
  Dada.logMsg(2, DL, "main: connecting to "+roach_ip+":"+str(roach_port))
  fpga = corr.katcp_wrapper.FpgaClient(roach_ip, roach_port)
  time.sleep(0.5)
  if (fpga.is_connected()):
    Dada.logMsg(2, DL, "main: connected")
  else:
    Dada.logMsg(-2, DL, "main: connection failed")
    sys.exit(1)

  # program bit stream
  Dada.logMsg(2, DL, "main: programming FPGA with " + roach_bof)
  fpga.progdev(roach_bof)
  Dada.logMsg(2, DL, "main: programming done")

  # open a listening socket
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

  Dada.logMsg(2, DL, "main: binding to "+listen_ip+":"+listen_port)
  sock.bind((listen_ip, int(listen_port)))

  # listen for at most 2 connections at a time (TCS and self)
  Dada.logMsg(2, DL, "main: sock.listen(1)")
  sock.listen(1)

  can_read = [sock]
  can_write = []
  can_error = []
  timeout = 1
  acc_len = 25

  message1 = chr(255) + chr(251) + chr(37) + \
             chr(255) + chr(253) + chr(38) + \
             chr(255) + chr(251) + chr(38) + \
             chr(255) + chr(253) + chr(3)  + \
             chr(255) + chr(251) + chr(24) + \
             chr(255) + chr(251) + chr(31) + \
             chr(255) + chr(251) + chr(32) + \
             chr(255) + chr(251) + chr(33) + \
             chr(255) + chr(251) + chr(34) + \
             chr(255) + chr(251) + chr(39) + \
             chr(255) + chr(253) + chr(5)  + \
             chr(255) + chr(251) + chr(35) 

  response = chr(255) + chr(251) + chr(1) + chr(255) + chr(251) + chr(3)

  message2 = chr(255) + chr(253) + chr(1)

  count = 0
  testing = False
  while (testing and not quit_event.isSet()):

    t1 = datetime.datetime.today()
    rawdata = fpga.read('scope_output1_bram', 512*4, 0)
    t2 = datetime.datetime.today()
    rawdata = fpga.read('scope_output2_bram', 512*4, 0)
    t3 = datetime.datetime.today()

    d1 = t2 - t1
    d2 = t3 - t2
    Dada.logMsg(2, DL, "pol1=" + str(d1) + " pol2="+str(d2))

    time.sleep(1)

    count += 1
    if (count > 10000):
      quit_event.set()

  bram = []
  tcpborph_cold = True

  # now simply wait for socket connections / commands
  while (not quit_event.isSet()):

    Dada.logMsg(3, DL, "main: calling select")
    did_read, did_write, did_error = select.select(can_read, can_write, can_error, timeout)
    Dada.logMsg(3, DL, "main: read="+str(len(did_read))+" write="+str(len(did_write))+" error="+str(len(did_error)))

    config_done = 0
    arp = {}

    # if we did_read
    if (len(did_read) > 0):
      for handle in did_read:
        if (handle == sock):
          (new_conn, addr) = sock.accept()
          Dada.logMsg(2, DL, "main: accept connection from "+repr(addr))
          # add the accepted connection to can_read
          can_read.append(new_conn)

          # immediately check if there is anything to read on the new_conn
          handshake = 0
          did_read, did_write, did_error = select.select(can_read, can_write, can_error, timeout)
          if (len(did_read) > 0):
            for handle in did_read:
              if (handle == new_conn):
                Dada.logMsg(2, DL, "main: data on new_conn")
                handshake = 1

          if (handshake):
            raw = new_conn.recv(4096)
            if (not raw == message1):
              Dada.logMsg(-1, DL, "main: handshake err: not message1")
            else:
              Dada.logMsg(2, DL, "main: handshake: sending response")
              new_conn.send(response)
              raw = new_conn.recv(4096)
              if (not raw == message2):
                Dada.logMsg(-1, DL, "main: handshake err: not message2")
              else:
                Dada.logMsg(2, DL, "main: handshake: sending greeting and prompt")
                send_greeting(new_conn)
                send_prompt(new_conn)
          else:
            Dada.logMsg(2, DL, "main: NO handshake: sending greeting and prompt")
            send_greeting(new_conn)
            send_prompt(new_conn)

        # an accepted connection must have generated some data
        else:

          raw = handle.recv(4096)
          message = raw.strip()
          Dada.logMsg(3, DL, "main: message='" + message+"' len="+str(len(message)) + " rawlen="+str(len(raw)))

          if (len(message) == 0):
            Dada.logMsg(2, DL, "main: closing connection")
            handle.close()
            for i, x in enumerate(can_read):
              if (x == handle):
                del can_read[i]
          else:

            if (message.find("bramdump") != 0):
              Dada.logMsg(1, DL, "<- " + message)
            parts = message.split(" ")

            if (message == "listdev"):
              handle.send(raw)
              list_dev(handle)

            # set the dest IP address (hex)
            elif (message.find("regwrite reg_ip") >= 0):
              dest_ip_hex = parts[2].replace("0x", "", 1).upper()
              send_reply(handle, raw, "\r")

            # set the dest port
            elif (message.find("regwrite reg_10GbE_destport0") >= 0):
              dest_port = parts[2]
              send_reply(handle, raw, "\r")

            # can be ignored
            elif (message == "write l xd0000000 xffffffff"):
              send_reply(handle, raw, "\r")
              send_reply(handle, "0xD0000000 : 0xFFFFFFFF", "\r\n")

            # can be ignored
            elif (message == "setb x40000000"):
              send_reply(handle, raw, '\r')
              send_reply(handle, "base address set to 0x40000000", "\r\n")

            # sets first part of source MAC address
            elif (message.find("writeb l 0") >= 0):
              first_part_mac = parts[3].lstrip("x").upper()
              send_reply(handle, raw, '\r')
              send_reply(handle, "base address set to 0x40000000", "\r\n")
              send_reply(handle, "0x40000000 : 0x" + first_part_mac, "\r\n")

            # sets second part of source MAC address
            elif (message.find("writeb l 4") >= 0):
              second_part_mac = parts[3].lstrip("x").upper()
              send_reply(handle, raw, '\r')
              send_reply(handle, "base address set to 0x40000000", "\r\n")
              send_reply(handle, "0x40000004 : 0x" + second_part_mac, "\r\n")

            elif (message.find("writeb l 8") >= 0):
              gateway_ip = parts[3].lstrip("x").upper()
              send_reply(handle, raw, '\r')
              send_reply(handle, "base address set to 0x40000000", "\r\n")
              send_reply(handle, "0x40000008 : 0x" + gateway_ip, "\r\n")
              
            elif (message.find("writeb l 12") >= 0):
              source_ip = parts[3].lstrip("x").upper() 
              send_reply(handle, raw, '\r')
              send_reply(handle, "base address set to 0x40000000", "\r\n")
              send_reply(handle, "0x40000012 : 0x" + source_ip, "\r\n")

            elif (message.find("writeb b x16") >= 0):
              src_port1 = parts[3].lstrip("x").upper()
              send_reply(handle, raw, '\r')
              send_reply(handle, "base address set to 0x40000000", "\r\n")
              send_reply(handle, "0x40000016 : 0x" + src_port1 + "A00000", "\r\n")

            elif (message.find("writeb b x17") >= 0):
              src_port2 = parts[3].lstrip("x").upper()
              send_reply(handle, raw, '\r')
              send_reply(handle, "base address set to 0x40000000", "\r\n")
              send_reply(handle, "0x40000017 : 0x" + src_port2 + "000000", "\r\n")

            # ignored I think
            elif (message.find("writeb b x15 xff") >= 0):
              send_reply(handle, raw, '\r')
              send_reply(handle, "base address set to 0x40000000", "\r\n")
              send_reply(handle, "0x40000015 : 0x010FA000", "\r\n")

            # ignored I think
            elif (message.find("write l xd0000000 x0") >= 0):
              send_reply(handle, raw, '\r')
              send_reply(handle, "base address set to 0x40000000", "\r\n")
              send_reply(handle, "0xD0000000 : 0x00000000", "\r\n")
              config_done = 1
          
            # ARP Entries
            elif (message.find("writeb l x") >= 0):
              memaddr = parts[2].lstrip("x")
              arp[memaddr] = parts[3].lstrip("x").upper()
              send_reply(handle, raw, '\r')
              send_reply(handle, "base address set to 0x40000000", "\r\n")
              send_reply(handle, "0x4000" + memaddr + " : 0x" + arp[memaddr], "\r\n")

            elif (message.find("regwrite reg_acclen") >= 0):
              acclen = parts[2]
              fpga.write_int('reg_acclen', int(acclen))
              send_reply(handle, raw, '\r')

            elif (message.find("regwrite reg_sync_period") >= 0):
              syncperiod = parts[2]
              fpga.write_int('reg_sync_period', int(syncperiod))
              send_reply(handle, raw, '\r')

            elif (message.find("regwrite reg_coeff_pol1") >= 0):
              coeff = parts[2]
              fpga.write_int('reg_coeff_pol1', int(coeff))
              send_reply(handle, raw, '\r')

            elif (message.find("regwrite reg_output_bitselect") >= 0):
              bitwindow = parts[2]
              fpga.write_int('reg_output_bitselect', int(bitwindow))
              send_reply(handle, raw, '\r')

            elif (message.find("regwrite reg_coeff_pol2") >= 0):
              coeff = parts[2]
              fpga.write_int('reg_coeff_pol2', int(coeff))
              send_reply(handle, raw, '\r')

            elif (message.find("regwrite reg_arm") >= 0):
              arm = parts[2]
              fpga.write_int('reg_arm', int(arm))
              send_reply(handle, raw, '\r')

            elif (message.find("bramdump scope_output") >= 0):

              Dada.logMsg(3, DL,  "main: bram: -> " + raw)

              bram_name = "scope_output1_bram"
              if (parts[1] == "scope_output3/bram"):
                bram_name = "scope_output2_bram"
          
              # try to get the tcpborphserver running hot
              t1 = datetime.datetime.today()
              bram = numpy.array(struct.unpack('>512I', fpga.read(bram_name, 512*4, 0)))
              t2 = datetime.datetime.today()
              delta = t2 - t1
              if (delta.microseconds < 100000):
                if (tcpborph_cold):
                  Dada.logMsg(2, DL, "main: TCPBORPH NOW HOT: "+str(delta))
                tcpborph_cold = False
              else:
                ntries = 3
                if (not tcpborph_cold):
                  Dada.logMsg(2, DL, "main: TCPBORPH NOW COLD: "+str(delta))
                tcpborph_cold = True
                while (ntries > 0):
                  bram = numpy.array(struct.unpack('>512I', fpga.read(bram_name, 512*4, 0)))
                  ntries -= 1

              bram_response = raw + "\r"

              for i in range(512):
                hex_bram = "%X" % bram[i]
                bram_response = bram_response + "0x" + hex_bram.rjust(8,'0') + "\r\n"

              handle.send(bram_response)

                #if (i == 255):
                #  handle.send(bram_response)
                #  bram_response = ""
                #if (i == 511):
                #  handle.send(bram_response)
                
                #Dada.logMsg(3, DL,  "main: bram["+str(i)+"]="+str(bram[i]) + " hex=" + hex_bram.rjust(8,'0'))
                #handle.send("0x" + hex_bram.rjust(8,'0') + "\r\n")

            elif (message == "quit"):
              send_reply(handle, raw, '\r')
              quit_event.set()

            else:
              Dada.logMsg(-1, DL, "Unrecognised command: '" + message + "'")

            if (raw != message1):
              Dada.logMsg(3, DL, "main: sending prompt")
              send_prompt(handle)

            if (config_done): 
              Dada.logMsg(2, DL, "main: config commands complete")

              # need to use the dest_ip, dest_port and src_port
              # use ROACH convention for src_mac and src_port

              Dada.logMsg(2, DL,  "main: dest_ip_hex="+dest_ip_hex)

              dest_ip = str(int(dest_ip_hex[0:2], 16)) + "." + \
                        str(int(dest_ip_hex[2:4], 16)) + "." + \
                        str(int(dest_ip_hex[4:6], 16)) + "." + \
                        str(int(dest_ip_hex[6:8], 16))

              roach_10gbe_dest_port = int(dest_port)

              src_port = str(int((src_port1 + src_port2), 16))
              roach_10gbe_src_port = int(src_port)

              roach_10gbe_dest_ip = Bpsr.convertIPToInteger(dest_ip)
              roach_10gbe_src_ip = Bpsr.convertIPToInteger(roach_cfg["ROACH_10GbE_SRC_IP_"+rid])
              roach_10gbe_src_mac   = (2<<40) + (2<<32) + roach_10gbe_src_ip

              Dada.logMsg(2, DL, "main: SRC: IP=" + roach_cfg["ROACH_10GbE_SRC_IP_"+rid] + " PORT=" + src_port + " MAC=" + str(roach_10gbe_src_mac))
              Dada.logMsg(2, DL, "main: DST: IP=" + dest_ip + " PORT=" + dest_port)
      
              tengbe_device = "ten_GbE"  

              fpga.tap_start(tengbe_device,tengbe_device,roach_10gbe_src_mac,roach_10gbe_src_ip,roach_10gbe_src_port)
              time.sleep(0.5)
              gbe0_link = bool(fpga.read_int(tengbe_device))
              if gbe0_link:
                Dada.logMsg(2, DL, "main: 10GbE device now active")
              else:
                Dada.logMsg(-1, DL, "main: 10GbE device not active")

              fpga.write_int('reg_ip', roach_10gbe_dest_ip)
              fpga.write_int('reg_10GbE_destport0', roach_10gbe_dest_port)
              fpga.write_int('reg_coeff_pol1', 16384)
              fpga.write_int('reg_coeff_pol2', 16384)
              config_done = 0


except:
  Dada.logMsg(-2, DL, "main: exception caught: " + str(sys.exc_info()[0]))
  print '-'*60
  traceback.print_exc(file=sys.stdout)
  print '-'*60
  quit_event.set()

# join threads
Dada.logMsg(2, DL, "main: joining control thread")
control_thread.join()

Dada.logMsg(1, DL, "STOPPING SCRIPT")
# exit
sys.exit(0)


