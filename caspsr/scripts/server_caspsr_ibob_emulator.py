#!/usr/bin/env python26
#
# Filename: server_caspsr_ibob_emulator.py
#
#   * masqueradges a roach as an ibob, providing a "tinyshell" interface

import Dada, Caspsr, threading, sys, time, socket, select, signal, traceback, struct
from corr import katcp_wrapper

PIDFILE  = "caspsr_ibob_emulator.pid"
LOGFILE  = "caspsr_ibob_emulator.log"
QUITFILE = "caspsr_ibob_emulator.quit"
DL = 2

###########################################################################

def signal_handler(signal, frame):
  print 'You pressed Ctrl+C!'
  global quit_event
  quit_event.set()

def send_reply (handle, addr, hdr, message):
  Dada.logMsg(2, DL, "send_reply: addr="+addr[0]+" port="+str(addr[1]) + " message=" +message)

  raw = struct.pack("8c", hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7]) + message 

  Dada.logMsg(3, DL, "send_reply: len(raw)="+str(len(raw)))
  Dada.logMsg(3, DL, "send_reply: raw="+raw)

  Dada.logMsg(1, DL, "-> " + message)
  handle.sendto(raw, addr)

  

  
############################################################################### 
#
# main
#

try:

  # get the BPSR configuration
  cfg = Caspsr.getConfig()

  log_file = cfg["SERVER_LOG_DIR"] + "/" + LOGFILE;
  pid_file = cfg["SERVER_CONTROL_DIR"] + "/" + PIDFILE;
  quit_file = cfg["SERVER_CONTROL_DIR"] + "/"  + QUITFILE;
  quit_event = threading.Event()

  signal.signal(signal.SIGINT, signal_handler)

  # start a control thread to handle quit requests
  control_thread = Dada.controlThread(quit_file, pid_file, quit_event, DL);
  control_thread.start()

  listen_ip   = cfg["IBOB_CONTROL_IP"]
  listen_port = cfg["IBOB_CONTROL_PORT"]
  fpga = []

  # open a listening socket
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

  Dada.logMsg(2, DL, "main: binding to "+listen_ip+":"+listen_port)
  sock.bind((listen_ip, int(listen_port)))

  can_read = [sock]
  can_write = []
  can_error = []
  timeout = 1

  # configure the ROACH with based on the default configuration
  Dada.logMsg(2, DL, "main: configureRoach()")
  (result, fpga) = Caspsr.configureRoach(DL, cfg)
  if (result == "fail"):
    Dada.logMsg(2, dl, "main: could not configure roach")
    sys.exit(1)
  Dada.logMsg(2, DL, "main: " + result)

  # now simply wait for socket connections / commands
  while (not quit_event.isSet()):

    Dada.logMsg(3, DL, "main: select CAN read="+str(len(can_read))+" write="+str(len(can_write))+" error="+str(len(can_error)))
    did_read, did_write, did_error = select.select(can_read, can_write, can_error, timeout)
    Dada.logMsg(3, DL, "main: select DID read="+str(len(did_read))+" write="+str(len(did_write))+" error="+str(len(did_error)))

    # if we did_read
    if (len(did_read) > 0):
      for handle in did_read:
        if (handle == sock):
          (raw, addr) = handle.recvfrom(1032)
          Dada.logMsg(3, DL, "main: recvfrom "+repr(addr))
          Dada.logMsg(3, DL, "main: raw="+raw)

          # first 8 bytes are a custom header
          Dada.logMsg(3, DL, "main: len(raw[0:8]) = "+str(len(raw[0:8])))
          header = struct.unpack("8c", raw[0:8])
          message = raw[8:].strip()

          Dada.logMsg(3, DL, "main: message='" + message+"' len="+str(len(message)) + " rawlen="+str(len(raw)))

          parts = message.split(" ")

          Dada.logMsg(1, DL, "<- " + message)

          # arms the roach
          if (message.find("regwrite reg_arm") >= 0):
            arm = parts[2]
            fpga.write_int('reg_arm', int(arm))
            send_reply(handle, addr, header, "ok")

          elif (message.find("quit") >= 0):
            send_reply(handle, addr, header, "ok")

          # ignore all messages that start with setb or write
          elif ((message.find("setb") == 0) or (message.find("write") == 0) or \
                (message.find("regwrite ") == 0) or (message.find("regread") == 0)):
            Dada.logMsg(2, DL, "main: ignoring " + message)
            message = "junk: " + ("x" * 57)
            send_reply(handle, addr, header, message)

          elif (message.find("help") == 0):
            Dada.logMsg(2, DL, "main: sending junk help reply")
            help_message = "junk help: " + ("h" * 310);
            send_reply(handle, addr, header, help_message)

          else:
            Dada.logMsg(-1, DL, "Unrecognised command: '" + message + "'")

  Dada.logMsg(2, DL, "main: closing socket")
  sock.close()

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

