#!/usr/bin/env python
#
# Filename: server_mopsr_input_monitor2.py
#

import Dada, Mopsr, threading, sys, time, socket, select, signal, traceback
import time, numpy, math, os, fnmatch

PIDFILE  = "mopsr_input_monitor.pid"
LOGFILE  = "mopsr_input_monitor.log"
QUITFILE = "mopsr_input_monitor.quit"
DL = 2
NCHAN = 128

###########################################################################

def signal_handler(signal, frame):
  print 'You pressed Ctrl+C!'
  global quit_event
  quit_event.set()

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
  
# Thread to collect data from all mopsr_dbstats instances
class monitorThread(threading.Thread):

  def __init__(self, quit_event, cond, lock, raw, images, data, cfg, nsamps):
    threading.Thread.__init__(self)
    self.quit_event = quit_event
    self.cond = cond
    self.lock = lock
    self.raw = raw
    self.images = images
    self.data = data
    self.cfg = cfg
    self.nsamps = nsamps 

  def run(self):
    self.quit_event = quit_event
    cond = self.cond
    lock = self.lock
    raw = self.raw
    images = self.images
    data = self.data
    cfg = self.cfg

    bytes = 128 * 2 * self.nsamps
    raw_bytes = []

    count = 1

    try:
      Dada.logMsg(2, DL, "monitorThread: starting")

      while (not quit_event.isSet()):

        # open a socket to each instance of mopsr_dbstats
        for i in range(int(cfg["NSEND"])):

          host = cfg["SEND_" + str(i)]
          port = 54321

          Dada.logMsg(3, DL, "monitorThread: openSocket(" + host + ", " + str(port) + ")")
          sock = Dada.openSocket(DL, host, port)

          Dada.logMsg(3, DL, "monitorThread: " + host + " <- 'nsamp'")
          result, response = Dada.sendTelnetCommand(sock, 'nsamp')
          Dada.logMsg(3, DL, "monitorThread: " + host + " -> " + result + " " + response)
          self.nsamps = int(response)

          for j in range (2):

            ant = 2*i + j

            bytes = 128 * 2 * self.nsamps

            Dada.logMsg(3, DL, "monitorThread: " + host + " <- 'dump " + str(ant) + "'")
            sock.send ("dump " + str(ant) + "\r\n")

            lock.acquire()
            Dada.logMsg(3, DL, "monitorThread: [lock acquired]")

            raw_bytes = ''
            recvd = 0
            while (len(raw_bytes) < bytes):
              chunk = sock.recv (bytes - recvd)
              if (chunk == ''):
                raise RuntimeError("socket connection broken")
              raw_bytes = raw_bytes + chunk

            Dada.logMsg(3, DL, "monitorThread: len(raw_bytes)=" + str(len(raw_bytes)))

            #raw[ant] = numpy.zeros ((128, self.nsamps), dtype=numpy.complex64)
            raw[ant] = Mopsr.unpack (NCHAN, self.nsamps, 2, raw_bytes)

            # delete all buffered images pertaining to this antenna
            images[ant].clear()
            data[ant].clear()

            Dada.logMsg(3, DL, "monitorThread: " + host + " -> binary [" + str(bytes) + "]")
            lock.release()
            Dada.logMsg(3, DL, "monitorThread: [lock released]")

          Dada.logMsg(3, DL, "monitorThread: sock.close()")
          sock.close()

        Dada.logMsg(3, DL, "monitorThread: sleep(2)")
        time.sleep(5)

    except:
      Dada.logMsg(-2, DL, "monitorThread: exception caught: " + str(sys.exc_info()[0]))
      print '-'*60
      traceback.print_exc(file=sys.stdout)
      print '-'*60
      quit_event.set()
      lock.release()

    if (not sock == []):
      Dada.logMsg(2, DL, "monitorThread: closing server socket")
      sock.close()

    Dada.logMsg(2, DL, "monitorThread: exiting")


# Thread to handle commands
class commandThread(threading.Thread):

  def __init__(self, quit_event, cond, lock, raw, images, data, cfg):
    threading.Thread.__init__(self)
    self.quit_event = quit_event
    self.cond = cond
    self.lock = lock
    self.raw = raw
    self.images = images
    self.data = data
    self.cfg = cfg

  def run(self):
    self.quit_event = quit_event
    cond = self.cond
    lock = self.lock
    raw = self.raw
    images = self.images
    data = self.data
    cfg = self.cfg

    try:

      # allocate some classes for each type of plot
      bandpass   = Mopsr.BandpassPlot ()
      histogram  = Mopsr.HistogramPlot ()
      timeseries = Mopsr.TimeseriesPlot ()
      freqtime   = Mopsr.FreqTimePlot ()

      nchan = 128

      # for bandpass output
      # spectrum = numpy.empty(nchan);

      # for waterfall output
      # spectra  = numpy.empty(nchan * nsamps);
      
      # for timeseries output
      # timeseries = numpy.empty(nsamps);
      
      Dada.logMsg(2, DL, "commandThread: starting")

      # open a socket to receive commands via socket
      hostname = cfg["SERVER_HOST"]
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      port = 33334
      Dada.logMsg(2, DL, "commandThread: binding to "+hostname+":"+str(port))
      sock.bind((hostname, port))

      # listen for at most 2 connections at a time (TMC and self)
      Dada.logMsg(3, DL, "commandThread: sock.listen(2)")
      sock.listen(2)

      can_read = [sock]
      can_write = []
      can_error = []

      timeout = 1
      bandpass_max = 0
      bandpass_min = 0

      # keep listening
      # while ((not quit_event.isSet()) or (len(can_read) > 1)):
      while (not quit_event.isSet()):

        Dada.logMsg(3, DL, "commandThread: calling select len(can_read)="+str(len(can_read)))
        timeout = 1
        did_read, did_write, did_error = select.select(can_read, can_write, can_error, timeout)
        Dada.logMsg(3, DL, "commandThread: read="+str(len(did_read))+" write="+str(len(did_write))+" error="+str(len(did_error)))  

        # if we did_read
        if (len(did_read) > 0):
          for handle in did_read:
            if (handle == sock):
              (new_conn, addr) = sock.accept()
              Dada.logMsg(3, DL, "commandThread: accept connection from "+repr(addr))
              # add the accepted connection to can_read
              can_read.append(new_conn)
              # new_conn.send("Welcome to the MOPSR Input Monitor\r\n")

            # an accepted connection must have generated some data
            else:
              message = handle.recv(4096)
              message = message.strip()
              Dada.logMsg(3, DL, "commandThread: message='" + message+"'")
              if (len(message) == 0):
                Dada.logMsg(1, DL, "commandThread: closing connection")
                handle.close()
                for i, x in enumerate(can_read):
                  if (x == handle):
                    del can_read[i]
              else:

                qdl = 3
                Dada.logMsg(qdl, DL, "<- " + message)
                args = message.split('&')
                command = args.pop(0)

                ant = 0
                chan = -1
                log = False
                transpose = False
                plain = False
                type = 'both'
                xres = 1024
                yres = 768
                nchan = 128
                ndim = 2
                zap = False
                limits = 'unique'
                nsamps = raw[ant].shape[1]
                Dada.logMsg(3, DL, "commandThread: nsamps="+str(nsamps))

                for i in range(len(args)):
                  Dada.logMsg(3, DL, "commandThread: parsing "+str(args[i]))
                  (key, val) = args[i].split('=')
                  if key == 'ant':
                    ant = int(val)
                  if key == 'chan':
                    chan = int(val)
                  if key == 'log':
                    log = str2bool(val)
                  if key == 'plain':
                    plain = str2bool(val)
                  if key == 'transpose':
                    transpose = str2bool(val)
                  if key == 'type':
                    type = val
                  if key == 'limits':
                    limits = val
                  if key == 'zap':
                    zap = str2bool(val) 
                  if key == 'size':
                    dims = val.split('x')
                    xres = dims[0]
                    yres = dims[1]

                Dada.logMsg(3, DL, "commandThread: ant="+str(ant)+" chan="+str(chan)+" log="+str(log)+
                                   " type="+type+" size="+str(xres)+"x"+str(yres)+
                                   " transpose="+str(transpose)+" plain="+str(plain)+
                                   " limits="+limits+" zap="+str(zap))

                if (command == "quit"):
                  quit_event.set()

                  # remove the server socket from the list of can_reads
                  for i, x in enumerate(can_read):
                    if (x == sock):
                      Dada.logMsg(2, DL, "commandThread: removed sock from can_read")
                      del can_read[i]

                  Dada.logMsg(2, DL, "commandThread: closing server socket [1]")
                  sock.close()
                  sock = []

                # request to close the socket
                elif (command == "exit"):
                  for i, x in enumerate(can_read):
                    if (x == handle):
                      Dada.logMsg(3, DL, "commandThread: removed curent handle from can_read")
                      del can_read[i]
                      handle.close()
                      handle = []

                elif (command == "help"):
                  handle.send("Available commands:\r\n")
                  handle.send("  help                      print these commands\r\n")
                  handle.send("  exit                      close current connection\r\n")
                  handle.send("  quit                      stop the server script\r\n")
                  handle.send("  dump <ant> <file>         dump raw binary data as int8_t from ant to file\r\n")
                  handle.send("  bandpass <options>        plot bandpass\r\n")
                  handle.send("  waterfall <options>       plot waterfall plot\r\n")
                  handle.send("  hist <options>            plot histogram\r\n")
                  handle.send("  timeseries <options>      plot histogram\r\n")
                  handle.send("  <options>                 space separated options as below\r\n")
                  handle.send("    ant=<antenna index>     optional [default=0]\r\n")
                  handle.send("    chan=<channel>          optional\r\n")
                  handle.send("    log=[no,yes]            optional\r\n")
                  handle.send("    type=[real,image,both]  optional\r\n")
                  handle.send("    size=<xres>x<yres>      optional\r\n")
                  handle.send("    plain=[no,yes]          optional\r\n")
                  handle.send("    zap=[no,yes]            optional\r\n")
                  handle.send("    transpose=[no,yes]      optional\r\n")
                  handle.send("    limits=[common,unique]  optional\r\n")
 
                elif (command == "dump"):
                  Dada.logMsg(3, DL, "commandThread: dump command received")
                  lock.acquire()
                  Dada.logMsg(3, DL, "commandThread: dump [lock acquired]")
                  lock.release()
                  Dada.logMsg(3, DL, "commandThread: dump [lock released]")
                  handle.send("ok\r\n")

                elif (command == "bandpass"):
                  lock.acquire()
                  key = 'bp_'+xres+'_'+yres+'_'+str(log)+'_'+str(transpose)+'_'+str(plain)+"_"+str(zap)
                  if key not in images[ant].keys():
                    Dada.logMsg(2, DL, key);
                    if 'spectrum' not in data[ant].keys():
                      data[ant]['spectrum'] = Mopsr.detectAndIntegrate (nchan, nsamps, ndim, raw[ant])
                    else:
                      Dada.logMsg(2, DL, 'skipping bandpass detection!')
                    bandpass.configure (log, zap, transpose, nchan)
                    bandpass.plot (xres, yres, plain, nchan, data[ant]['spectrum'])
                    images[ant][key] = bandpass.getRawImage()
                  handle.send (images[ant][key])
                  lock.release()
                  for i, x in enumerate(can_read):
                    if (x == handle):
                      Dada.logMsg(3, DL, "commandThread: removed curent handle from can_read")
                      del can_read[i]
                      handle.close()
                      handle = []

                elif (command == "hist"):
                  lock.acquire()
                  Dada.logMsg(3, DL, "commandThread: hist command received")
                  key = 'hg_'+xres+'_'+yres+'_'+str(chan)+'_'+str(type)+'_'+str(plain)
                  if key not in images[ant].keys():
                    Dada.logMsg(2, DL, key);
                    dkey1 = 'real_'+str(chan)
                    dkey2 = 'imag_'+str(chan)
                    if dkey1 not in data[ant].keys():
                      (data[ant][dkey1], data[ant][dkey2]) = Mopsr.extractChannel(chan, nchan, nsamps, type, raw[ant])
                    histogram.configure (chan)
                    histogram.plot (xres, yres, plain, data[ant][dkey1], data[ant][dkey2], 256)
                    images[ant][key] = histogram.getRawImage()
                  handle.send (images[ant][key])
                  lock.release()
                  for i, x in enumerate(can_read):
                    if (x == handle):
                      Dada.logMsg(3, DL, "commandThread: removed curent handle from can_read")
                      del can_read[i]
                      handle.close()
                      handle = []

                elif (command == "timeseries"):
                  lock.acquire()
                  Dada.logMsg(3, DL, "commandThread: timeseries command received")
                  key = 'ts_'+xres+'_'+yres+'_'+str(chan)+'_'+str(type)+'_'+str(plain)
                  if key not in images[ant].keys():
                    Dada.logMsg(2, DL, key);
                    if chan == -1:
                      chan = 64
                    dkey1 = 'real_'+str(chan)
                    dkey2 = 'imag_'+str(chan)
                    if dkey1 not in data[ant].keys():
                      (data[ant][dkey1], data[ant][dkey2]) = Mopsr.extractChannel(chan, nchan, nsamps, type, raw[ant])
                    timeseries.configure (chan, nsamps)
                    timeseries.plot(xres, yres, plain, data[ant][dkey1], data[ant][dkey2])
                    images[ant][key] = timeseries.getRawImage()
                  handle.send (images[ant][key])
                  lock.release()
                  for i, x in enumerate(can_read):
                    if (x == handle):
                      Dada.logMsg(3, DL, "commandThread: removed curent handle from can_read")
                      del can_read[i]
                      handle.close()
                      handle = []

                elif (command == "waterfall"):
                  lock.acquire()
                  key = 'wf_'+xres+'_'+yres+'_'+str(log)+'_'+str(plain)+"_"+str(zap)
                  if key not in images[ant].keys():
                    Dada.logMsg(2, DL, key);
                    if 'spectra' not in data[ant].keys():
                      data[ant]['spectra'] = Mopsr.detectTranspose (nchan, nsamps, ndim, raw[ant])
                    freqtime.configure (log, zap, transpose)
                    freqtime.plot (xres, yres, plain, data[ant]['spectra'], nchan, nsamps)
                    images[ant][key] = freqtime.getRawImage()
                  handle.send (images[ant][key])
                  lock.release()
                  for i, x in enumerate(can_read):
                    if (x == handle):
                      Dada.logMsg(3, DL, "commandThread: removed curent handle from can_read")
                      del can_read[i]
                      handle.close()
                      handle = []

                else:
                  Dada.logMsg(2, DL, "commandThread: unrecognised command ["+command+"]")
                  Dada.logMsg(1, DL, " -> fail")
                  handle.send("fail\r\n")

    except:
      Dada.logMsg(-2, DL, "commandThread: exception caught: " + str(sys.exc_info()[0]))
      print '-'*60
      traceback.print_exc(file=sys.stdout)
      print '-'*60
      lock.release()
      quit_event.set()

    for i, handle in enumerate(can_read):
      Dada.logMsg(2, DL, "commandThread: closing can_read["+str(i)+"]")
      handle.close
      del can_read[i]

    if (not sock == []): 
      Dada.logMsg(2, DL, "commandThread: closing server socket [2]")
      sock.close()

    Dada.logMsg(2, DL, "commandThread: exiting")     

############################################################################### 
#
# main
#

# get the BPSR configuration
cfg = Mopsr.getConfig()

lock = threading.Lock()
cond = threading.Condition(lock)

control_thread = []
monitor_thread = []
raw = []
images = []
data = []
command_sock = 0
nsamps = 0

log_file  = cfg["SERVER_LOG_DIR"] + "/" + LOGFILE
pid_file  = cfg["SERVER_CONTROL_DIR"] + "/" + PIDFILE
quit_file = cfg["SERVER_CONTROL_DIR"] + "/"  + QUITFILE

if os.path.exists(quit_file):
  sys.stderr.write("quit file existed at launch: " + quit_file)
  sys.exit(1)

# become a daemon
# Dada.daemonize(pid_file, log_file)

try:

  Dada.logMsg(1, DL, "STARTING SCRIPT")

  quit_event = threading.Event()

  signal.signal(signal.SIGINT, signal_handler)

  # start a control thread to handle quit requests
  control_thread = Dada.controlThread(quit_file, pid_file, quit_event, DL)
  control_thread.start()

  # start a thread for each ROACH board
  for i in range(int(cfg["NANT"])):
    raw.append(0)
    images.append({})
    data.append({})

  # start the monitor thread to handle collection of data from inputs
  Dada.logMsg(2, DL, "main: starting input monitor thread")
  monitor_thread = monitorThread(quit_event, cond, lock, raw, images, data, cfg, nsamps)
  monitor_thread.start()

  # start a thread to handle socket commands that will server plot/raw data 
  Dada.logMsg(2, DL, "main: starting command thread")
  command_thread = commandThread(quit_event, cond, lock, raw, images, data, cfg)
  command_thread.start()

  # allow some time for commandThread to open listening socket
  Dada.logMsg(2, DL, "main: sleep 3")
  time.sleep (3)

  hostname = cfg["SERVER_HOST"]
  port = 33334

  while (not quit_event.isSet()):

    Dada.logMsg(3, DL, "main: while loop NANT="+cfg["NANT"])

    if (command_sock == 0):
      Dada.logMsg(3, DL, "main: openSocket("+hostname+", "+str(port)+")")
      command_sock = Dada.openSocket(DL, hostname, port)
      Dada.logMsg(3, DL, "main: command_sock="+repr(command_sock))

      # receive the welcome message [junk]
      response = command_sock.recv(4096)

    # result, response = Dada.sendTelnetCommand (command_sock, 'bandpass')

    # open a socket to each instance of mopsr_dbstats
#    for i in range(int(cfg["NANT"])):

#      timestamp = Dada.getCurrentDadaTime()

#      file = "/data/mopsr/monitor/" + str(i) + "/" + timestamp + ".raw"

#      cmd = "dump " + str(i) + " " + file
#      Dada.logMsg (3, DL, "main: <- '" + cmd + "'")
#      result, response = Dada.sendTelnetCommand (command_sock, cmd)
#      Dada.logMsg (3, DL, "main: -> " + result + " " + response)

    # every 10 seconds, we will dump raw data for all antenna to disk for long term storage
    counter = 10
    while ((not quit_event.isSet()) and (counter > 0)):
      counter -= 1
      Dada.logMsg(3, DL, "main: sleeping")
      time.sleep(1)

  Dada.logMsg(2, DL, "main: command_sock.close()")
  command_sock.close()

  # join the command thread
  Dada.logMsg(2, DL, "main: joining command thread")
  command_thread.join()
  Dada.logMsg(2, DL, "main: command thread joined")

except:
  Dada.logMsg(-2, DL, "main: exception caught: " + str(sys.exc_info()[0]))
  print '-'*60
  traceback.print_exc(file=sys.stdout)
  print '-'*60
  quit_event.set()

Dada.logMsg(2, DL, "main: lock.acquire()")
lock.acquire()
Dada.logMsg(3, DL, "main: lock acquired")

Dada.logMsg(2, DL, "main: cond.notifyAll()")
cond.notifyAll()
Dada.logMsg(3, DL, "main: lock.release()")
lock.release()

# join threads
if (control_thread):
  Dada.logMsg(1, DL, "main: joining control thread")
  control_thread.join()
  Dada.logMsg(2, DL, "main: control thread joined")

if (monitor_thread):
  Dada.logMsg(1, DL, "main: joining monitor thread")
  monitor_thread.join()
  Dada.logMsg(2, DL, "main: monitor thread joined")

Dada.logMsg(2, DL, "main: exiting")

Dada.logMsg(1, DL, "STOPPING SCRIPT")
# exit
sys.exit(0)


