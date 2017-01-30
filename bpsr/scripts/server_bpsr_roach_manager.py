#!/usr/bin/env python
#
# Filename: server_bpsr_roach_manager.py
#
#   * load firmware and config roaches for use with BPSR
#   * allow level setting
#   * allow rearming
#   * retrieve bram data

import Dada, Bpsr, threading, sys, time, socket, select, signal, traceback
import corr, time, numpy, math, os, fnmatch
import fullpol as spec
#import dualpol as spec

PIDFILE  = "bpsr_roach_manager.pid"
LOGFILE  = "bpsr_roach_manager.log"
QUITFILE = "bpsr_roach_manager.quit"
DL = 1

QUIT = 1
IDLE = 2
ERROR = 3
TASK_STATE = 4
TASK_EXIT = 5
TASK_HELP = 6
TASK_CONFIG = 7
TASK_ARM = 8
TASK_START_TX= 9
TASK_STOP_TX = 10
TASK_LEVELS = 11
TASK_SETGAINS = 12
TASK_BRAMPLOT = 13
TASK_BRAMDISK = 14
TASK_ACCLEN = 15

###########################################################################

def signal_handler(signal, frame):
  print 'You pressed Ctrl+C!'
  global quit_event
  quit_event.set()
  
# Thread to operate on Roach
class roachThread(threading.Thread):

  def __init__(self, roach_num, quit_event, cond, lock, states, args, results, responses, cfg):
    threading.Thread.__init__(self)
    self.roach_num = roach_num
    self.quit_event = quit_event
    self.cond = cond
    self.lock = lock
    self.states = states
    self.args = args
    self.results = results
    self.responses = responses
    self.cfg = cfg

  def run(self):
    rid = str(self.roach_num)
    ithread = self.roach_num
    cfg = self.cfg
    lock = self.lock
    cond = self.cond
    states = self.states
    args = self.args
    results = self.results
    responses = self.responses
    fpga = []
    locked = False
    acc_len = 25
    roach_cfg = Bpsr.getROACHConfig()
    roach_name = roach_cfg["ROACH_"+rid]
    beam_name = roach_cfg["BEAM_"+rid]
    os.chdir(cfg["SERVER_STATS_DIR"])

    try:

      Dada.logMsg(1, DL, "["+rid+"] roachThread: starting")

      Dada.logMsg(2, DL, "["+rid+"] roachThread: connectRoach("+rid+")")
      result, fpga = Bpsr.connectRoach(DL, rid)
      if (result != "ok"):
        raise NameError('Could not connect to Roach')

      # acquire lock to wait for commands setup
      locked = lock.acquire()
      Dada.logMsg(3, DL, "["+rid+"] roachThread: lock acquired")

      while (states[ithread] != QUIT):
        while (states[ithread] == IDLE):
          Dada.logMsg(3, DL, "["+rid+"] roachThread: waiting for not IDLE")
          locked = False
          cond.wait()
          locked = True
        if (states[ithread] == QUIT):
          Dada.logMsg(1, DL, "["+rid+"] roachThread: quit requested")
          spec.stopTX(DL, fpga)
          lock.release()
          locked = False
          return

        # we have been given a command to perform (i.e. not QUIT or IDLE)
        task = states[ithread]
        arg  = args[ithread]
        Dada.logMsg(3, DL, "["+rid+"] roachThread: TASK="+str(task)+" ARG="+str(arg))

        Dada.logMsg(3, DL, "["+rid+"] roachThread: lock.release()")
        lock.release()
        locked = False
  
        result = "fail"
        response = ""

        if (task == TASK_STATE):
          Dada.logMsg(3, DL, "["+rid+"] roachThread: state request")
          if (fpga != []):
            result = "ok"

        elif (task == TASK_CONFIG):
          Dada.logMsg(3, DL, "["+rid+"] roachThread: perform config")
          
          result, response = spec.programRoach(DL, fpga, rid)
          if (result == "ok"):
            result, response = spec.configureRoach(DL, fpga, rid, cfg)
            if (result == "ok"):
              result, response = spec.accLenRoach(DL, fpga, acc_len, rid)
              if (result == "ok"):
                Dada.logMsg(2, DL, "["+rid+"] roachThread: config finished")
              else:
                Dada.logMsg(-2, DL, "["+rid+"] roachThread: accLenRoach failed " + response)
            else:
              Dada.logMsg(-2, DL, "["+rid+"] roachThread: configureRoach failed " + response)
          else:
            Dada.logMsg(-2, DL, "["+rid+"] roachThread: programRoach failed " + response)

        elif (task == TASK_ACCLEN):
          acc_len = int(arg)
          Dada.logMsg(3, DL, "["+rid+"] roachThread: perform accLen")
          result, response = spec.accLenRoach(DL, fpga, acc_len, rid)

        elif (task == TASK_ARM):
          Dada.logMsg(3, DL, "["+rid+"] roachThread: perform arm")
          result = spec.rearm(DL, fpga)

        elif (task == TASK_START_TX):
          Dada.logMsg(3, DL, "["+rid+"] roachThread: perform start tx")
          result = spec.startTX(DL, fpga)

        elif (task == TASK_STOP_TX):
          Dada.logMsg(3, DL, "["+rid+"] roachThread: perform stop tx")
          result = spec.stopTX(DL, fpga)

        elif (task == TASK_LEVELS):
          Dada.logMsg(3, DL, "["+rid+"] roachThread: perform set levels")
          if (fpga != []):
            # test if all our beam/pols are active first
            pol1, pol2 = Bpsr.getActivePolConfig(beam_name)

            Dada.logMsg(2, DL, "["+rid+"] roachThread: setLevels("+str(pol1)+", " + str(pol2)+")")
            result, response = spec.setLevels(DL, fpga, pol1, pol2, rid)
            Dada.logMsg(2, DL, "["+rid+"] roachThread: result=" + result + " response=" + response)

        elif (task == TASK_SETGAINS):
          new_cross_gain = int(arg)
          new_bit_window = 3
          Dada.logMsg(3, DL, "["+rid+"] roachThread: perform setGains")
          result, response = spec.setComplexGains(DL, fpga, rid, new_cross_gain, new_bit_window)

        elif (task == TASK_BRAMPLOT):
          if (fpga != []):
            Dada.logMsg(3, DL, "["+rid+"] roachThread: perform bramplot")
            time_str = Dada.getCurrentDadaTime()
            result = spec.bramplotRoach(DL, fpga, time_str, beam_name)
          else:
            Dada.logMsg(-1, DL, "["+rid+"] roachThread: not connected to FPGA")

        elif (task == TASK_BRAMDISK):
          if (fpga != []):
            Dada.logMsg(3, DL, "["+rid+"] roachThread: perform bramdisk")
            time_str = Dada.getCurrentDadaTime()
            result = spec.bramdiskRoach(DL, fpga, time_str, beam_name)
          else:
            Dada.logMsg(-1, DL, "["+rid+"] roachThread: not connected to FPGA")

        else:
          Dada.logMsg(2, DL, "["+rid+"] roachThread: unrecognised task!!!")

        # now that task is done, re-acquire lock
        Dada.logMsg(3, DL, "["+rid+"] roachThread: lock.acquire()")
        locked = lock.acquire()

        Dada.logMsg(3, DL, "["+rid+"] roachThread: setting state = IDLE")
        states[ithread] = IDLE
        results[ithread] = result
        responses[ithread] = response
        Dada.logMsg(3, DL, "["+rid+"] roachThread: cond.notifyAll()")
        cond.notifyAll()
    
    except:

      print '-'*60
      traceback.print_exc(file=sys.stdout)
      print '-'*60

      quit_event.set()
      if (not locked):
        Dada.logMsg(0, DL, "["+rid+"] roachThread: except: lock.acquire()")
        locked = lock.acquire()
      Dada.logMsg(0, DL, "["+rid+"] roachThread: except: setting state = ERROR")
      states[ithread] = ERROR
      results[ithread] = "fail"
      responses[ithread] = "exception ocurred in roachThread " + roach_name + ":" + beam_name
      spec.stopTX(DL, fpga)
      Dada.logMsg(2, DL, "["+rid+"] roachThread: except: cond.notifyAll()")
      cond.notifyAll()
      Dada.logMsg(2, DL, "["+rid+"] roachThread: except: lock.release()")
      lock.release()
      Dada.logMsg(2, DL, "["+rid+"] roachThread: except: exiting")
      return

     # we have been asked to exit the rthread
    Dada.logMsg(1, DL, "["+rid+"] roachThread: end of thread")
    spec.stopTX(DL, fpga)

# Thread to handle commands
class commandThread(threading.Thread):

  def __init__(self, quit_event, cond, lock, states, args, results, responses, cfg):
    threading.Thread.__init__(self)
    self.quit_event = quit_event
    self.cond = cond
    self.lock = lock
    self.states = states
    self.args = args 
    self.results = results
    self.responses = responses
    self.cfg = cfg

  def run(self):
    self.quit_event = quit_event
    cond = self.cond
    lock = self.lock
    states = self.states
    args = self.args
    results = self.results
    responses = self.responses
    cfg = self.cfg

    try:

      Dada.logMsg(2, DL, "commandThread: starting")

      # open a socket to receive commands via socket
      hostname = cfg["SERVER_HOST"]
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      Dada.logMsg(2, DL, "commandThread: binding to "+hostname+":"+cfg["IBOB_MANAGER_PORT"])
      sock.bind((hostname, int(cfg["IBOB_MANAGER_PORT"])))

      # listen for at most 2 connections at a time (TCS and self)
      Dada.logMsg(3, DL, "commandThread: sock.listen(2)")
      sock.listen(2)

      can_read = [sock]
      can_write = []
      can_error = []
      timeout = 1

      valid_commands = dict({ "QUIT":QUIT, \
                              "EXIT":TASK_EXIT, \
                              "HELP":TASK_HELP, \
                              "STATE":TASK_STATE, \
                              "CONFIG":TASK_CONFIG, \
                              "ARM":TASK_ARM, \
                              "START_TX":TASK_START_TX, \
                              "STOP_TX":TASK_STOP_TX, \
                              "LEVELS":TASK_LEVELS, \
                              "SETGAINS":TASK_SETGAINS, \
                              "BRAMPLOT":TASK_BRAMPLOT, \
                              "BRAMDISK":TASK_BRAMDISK, \
                              "ACCLEN":TASK_ACCLEN})

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
              Dada.logMsg(1, DL, "commandThread: accept connection from "+repr(addr))
              # add the accepted connection to can_read
              can_read.append(new_conn)
              new_conn.send("Welcome to the BPSR Roach Manager\r\n")

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

                message = message.upper()
                if ((message == "BRAMPLOT") or (message == "BRAMDISK") or (message == "STATE")):
                  qdl = 2
                else:
                  qdl = 1

                Dada.logMsg(qdl, DL, "<- " + message)
                message_parts = message.split(" ", 2)

                if (message_parts[0] in valid_commands.keys()):
                
                  command = valid_commands[message_parts[0]]
                  arg = ""
                  if (len(message_parts) == 2):
                   arg = message_parts[1] 

                  Dada.logMsg(3, DL, "commandThread: " + message + " was valid, index=" + str(command))

                  if (command == QUIT):
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
                  elif (command == TASK_EXIT):
                    for i, x in enumerate(can_read):
                      if (x == handle):
                        Dada.logMsg(2, DL, "commandThread: removed curent handle from can_read")
                        del can_read[i]
                        handle.close()
                        handle = []

                  elif (command == TASK_HELP):
                    handle.send("Available commands:\r\n")
                    handle.send("  help           print these commands\r\n")
                    handle.send("  exit           close current connection\r\n")
                    handle.send("  quit           stop the server script\r\n")
                    handle.send("  config         program all roaches\r\n")
                    handle.send("  state          report state of all roaches\r\n")
                    handle.send("  acclen <num>   set the accumlation length\r\n")
                    handle.send("  levels         perform level setting\r\n")
                    handle.send("  setgains <num> set cross pol gains to <num>\r\n")
                    handle.send("  arm            start 10GbE packets, reset seq no, returning UTC_START\r\n")
                    handle.send("  stop           stop 10GbE packets\r\n")
              
                  else:
                    # now process this message
                    Dada.logMsg(3, DL, "commandThread: lock.acquire()")
                    lock.acquire()
                    Dada.logMsg(3, DL, "commandThread: lock acquired")

                    for i in range(n_roach):
                      states[i] = command 
                      args[i] = arg
                    state = command 
                    Dada.logMsg(3, DL, "commandThread: states set to " + message)

                    # all commands should happen as soon as is practical, expect
                    # for the arm, which should ocurr very close to 0.5 seconds
                    # through a second. This command should also return the UTC time
                    # corresponding to the expected start
                    if (command == TASK_ARM):
                      # busy sleep until the next second ticks over
                      curr_time = int(time.time())
                      next_time = curr_time
                      Dada.logMsg(2, DL, "commandThread: waiting for 1 second boundary")
                      while (curr_time == next_time):
                        next_time = int(time.time())
                      Dada.logMsg(2, DL, "commandThread: sleeping 0.5 seconds")
                      time.sleep(0.5)
                      utc_start = Dada.getUTCDadaTime(1)
                      Dada.logMsg(2, DL, "commandThread: UTC_START=" + utc_start)

                    # activate threads
                    Dada.logMsg(3, DL, "commandThread: cond.notifyAll()")
                    cond.notifyAll()
                    Dada.logMsg(3, DL, "commandThread: lock.release()")
                    lock.release()

                    # wait for all roaches to finished the command
                    Dada.logMsg(3, DL, "commandThread: lock.acquire()")
                    lock.acquire()
                    Dada.logMsg(3, DL, "commandThread: lock acquired")

                    command_result = ""
                    command_response = ""

                    while (state == command):
                      Dada.logMsg(3, DL, "commandThread: checking all roaches for IDLE")

                      n_idle = 0
                      n_error = 0
                      n_running = 0
                      n_ok = 0
                      n_fail = 0

                      for i in range(n_roach):
                        Dada.logMsg(3, DL, "commandThread: testing roach["+str(i)+"]")

                        # check the states of each roach thread
                        if (states[i] == IDLE):
                          n_idle += 1
                        # check the return values of this roach
                          Dada.logMsg(3, DL, "commandThread: results["+str(i)+"] = " + results[i])
                          if (results[i] == "ok"):
                            n_ok += 1
                          else:
                            n_fail += 1
                        elif (states[i] == ERROR):
                          Dada.logMsg(-1, DL, "commandThread: roach["+str(i)+"] thread failed")
                          n_error += 1
                        else:
                          n_running += 1


                      # if all roach threads are idle, we are done - extract the results
                      if (n_idle == n_roach):

                        state = IDLE
                        if (n_ok == n_roach):
                          command_result = "ok"
                        else:
                          command_result= "fail"
                        for i in range(n_roach):
                          command_response = command_response + roach_cfg["BEAM_"+str(i)] + ":"
                          if (responses[i] != ""):
                            command_response += responses[i] + " "
                          else:
                            command_response += results[i] + " "

                      elif (n_error > 0):
                        Dada.logMsg(2, DL, "commandThread: roach thread error")
                        command_result = "fail"
                        command_response = str(n_error) + " roach threads failed"
                        state = ERROR

                      else:
                        Dada.logMsg(3, DL, "commandThread: NOT all IDLE, cond.wait()")
                        cond.wait()
  
                    if (command == TASK_ARM):
                      if (command_response != ""):
                        command_response = command_response + "\n" + utc_start
                      else:
                        command_response = utc_start

                    if (command_response != ""):
                      handle.send(command_response + "\r\n") 
                      Dada.logMsg(qdl, DL, "-> " + command_response)

                    handle.send(command_result + "\r\n")
                    Dada.logMsg(qdl, DL, "-> " + command_result)

                    Dada.logMsg(3, DL, "commandThread: lock.release()")
                    lock.release()

                else:
                  Dada.logMsg(2, DL, "commandThread: unrecognised command")
                  Dada.logMsg(1, DL, " -> fail")
                  handle.send("fail\r\n")

    except:
      Dada.logMsg(-2, DL, "commandThread: exception caught: " + str(sys.exc_info()[0]))
      print '-'*60
      traceback.print_exc(file=sys.stdout)
      print '-'*60

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
# Thread to plot any bram dump files that occur in the stats dir
#
class plotThread(threading.Thread):

  def __init__(self, quit_event, cfg, roach_cfg):
    threading.Thread.__init__(self)
    self.quit_event = quit_event
    self.cfg = cfg

  def run(self):
    cfg = self.cfg
    os.chdir(cfg["SERVER_STATS_DIR"])

    # generate list of current beams
    beams = []
    for i in range(int(roach_cfg["NUM_ROACH"])):
      beams.append(roach_cfg["BEAM_"+str(i)])

    # generate list of resolutions
    resolutions = ["1024x768", "400x300", "112x84"]

    Dada.logMsg(1, DL, "plotThread: starting")

    while (not quit_event.isSet()):
    
      # get dir listing of the stats dir
      directory = cfg["SERVER_STATS_DIR"]
      extension = ".bram"
      bram_files = [file for file in os.listdir(directory) if file.lower().endswith(extension)]

      for bram_file in bram_files:

        Dada.logMsg(2, DL, "plotThread: found " + bram_file)
        cmd = "bpsr_bramplot " + bram_file
        
        Dada.logMsg(2, DL, "plotThread: " + cmd)
        os.system(cmd)
        os.remove(bram_file)

      for beam in beams:
        for res in resolutions:
          filematch = "*_" + beam + "_" + res + ".png"
          Dada.logMsg(3, DL, "plotThread: checking " + os.getcwd() + " for matching " + filematch)
          filelist = os.listdir(os.getcwd())
          filelist.sort(reverse=True)
          count = 0
          for filename in filelist:
            if (fnmatch.fnmatch(filename, filematch)):
              count += 1
              if (count > 3):
                Dada.logMsg(3, DL, "plotThread: cleaning file: "+filename)
                os.remove(filename)

      extension = ".bram_cross"
      bram_files = [file for file in os.listdir(directory) if file.lower().endswith(extension)]

      for bram_file in bram_files:

        Dada.logMsg(2, DL, "plotThread: found " + bram_file)
        cmd = "bpsr_bramplot_cross " + bram_file

        Dada.logMsg(2, DL, "plotThread: " + cmd)
        os.system(cmd)
        os.remove(bram_file)

      for beam in beams:
        for res in resolutions:
          filematch = "*_" + beam + "_" + res + "_cross.png"
          Dada.logMsg(3, DL, "plotThread: checking " + os.getcwd() + " for matching " + filematch)
          filelist = os.listdir(os.getcwd())
          filelist.sort(reverse=True)
          count = 0
          for filename in filelist:
            if (fnmatch.fnmatch(filename, filematch)):
              count += 1
              if (count > 3):
                Dada.logMsg(3, DL, "plotThread: cleaning file: "+filename)
                os.remove(filename)


      extension = ".bram_hist"
      bram_files = [file for file in os.listdir(directory) if file.lower().endswith(extension)]

      for bram_file in bram_files:

        Dada.logMsg(2, DL, "plotThread: found " + bram_file)
        cmd = "bpsr_bramplot_hist " + bram_file

        Dada.logMsg(2, DL, "plotThread: " + cmd)
        os.system(cmd)
        os.remove(bram_file)

      for beam in beams:
        for res in resolutions:
          filematch = "*_" + beam + "_" + res + "_hist.png"
          Dada.logMsg(3, DL, "plotThread: checking " + os.getcwd() + " for matching " + filematch)
          filelist = os.listdir(os.getcwd())
          filelist.sort(reverse=True)
          count = 0
          for filename in filelist:
            if (fnmatch.fnmatch(filename, filematch)):
              count += 1
              if (count > 3):
                Dada.logMsg(3, DL, "plotThread: cleaning file: "+filename)
                os.remove(filename)

      # now sleep for a bit
      time.sleep (4.0)

    # end of thread 
    Dada.logMsg(1, DL, "plotThread: exiting")


############################################################################### 
#
# main
#

# get the BPSR configuration
cfg = Bpsr.getConfig()
roach_cfg = Bpsr.getROACHConfig()

lock = threading.Lock()
cond = threading.Condition(lock)

control_thread = []

n_roach = 0
roach_threads = []
roach_states = []
roach_args = []
roach_results = []
roach_responses = []

log_file = cfg["SERVER_LOG_DIR"] + "/" + LOGFILE
pid_file = cfg["SERVER_CONTROL_DIR"] + "/" + PIDFILE
quit_file = cfg["SERVER_CONTROL_DIR"] + "/"  + QUITFILE

if os.path.exists(quit_file):
  sys.stderr.write("quit file existed at launch: " + quit_file)
  sys.exit(1)

# become a daemon
Dada.daemonize(pid_file, log_file)

try:

  Dada.logMsg(1, DL, "STARTING SCRIPT")

  quit_event = threading.Event()

  signal.signal(signal.SIGINT, signal_handler)

  # start a control thread to handle quit requests
  control_thread = Dada.controlThread(quit_file, pid_file, quit_event, DL)
  control_thread.start()

  # start a thread for each ROACH board
  for i in range(int(roach_cfg["NUM_ROACH"])):
    roach_states.append(IDLE)
    roach_args.append("")
    roach_results.append("")
    roach_responses.append("")
    thr = roachThread(i, quit_event, cond, lock, roach_states, roach_args, roach_results, roach_responses, cfg)
    thr.start()
    roach_threads.append(thr)
    n_roach += 1

  # start a thread to handle socket commands that will interact with the ROACH threads
  Dada.logMsg(2, DL, "main: starting command thread")
  command_thread = commandThread(quit_event, cond, lock, roach_states, roach_args, roach_results, roach_responses, cfg)
  command_thread.start()

  # start a thread to handle plotting of bram dumps
  Dada.logMsg(2, DL, "main: starting plot thread")
  plot_thread = plotThread(quit_event, cfg, roach_cfg)
  plot_thread.start()

  # allow some time for commandThread to open listening socket
  time.sleep(2)

  # open a socket to the command thread
  hostname = cfg["SERVER_HOST"]
  port =  int(cfg["IBOB_MANAGER_PORT"])

  # wait for all roaches to be active, then start bramdumping till exit
  roaches_all_active = False
  command_sock = 0

  # connect to our own command thread
  Dada.logMsg(3, DL, "main: openSocket("+hostname+", "+str(port)+")")
  command_sock = Dada.openSocket(DL, hostname, port)
  Dada.logMsg(3, DL, "main: command_sock="+repr(command_sock))

  response = command_sock.recv(4096)

  Dada.logMsg(1, DL, 'Configuring ' + roach_cfg["NUM_ROACH"]+ ' ROACH boards with BPSR gateware')
  Dada.logMsg(3, DL, "main: <- 'config'")
  result, response = Dada.sendTelnetCommand(command_sock, 'config')
  Dada.logMsg(3, DL, "main: -> " + result + " " + response)

  time.sleep(0.1)

  Dada.logMsg(1, DL, 'Setting levels to nominal values')
  Dada.logMsg(3, DL, "main: <- 'levels'")
  result, response = Dada.sendTelnetCommand(command_sock, 'levels')
  Dada.logMsg(3, DL, "main: -> " + result + " " + response)

  while (not quit_event.isSet()):

    Dada.logMsg(3, DL, "main: while loop")

    if (command_sock == 0):
      Dada.logMsg(3, DL, "main: openSocket("+hostname+", "+str(port)+")")
      command_sock = Dada.openSocket(DL, hostname, port)
      Dada.logMsg(3, DL, "main: command_sock="+repr(command_sock))

      # receive the welcome message [junk]
      response = command_sock.recv(4096)

    # see if roaches are all active
    if ((not roaches_all_active) and (command_sock != 0)):

      Dada.logMsg(3, DL, "main: <- 'state'")
      result, response = Dada.sendTelnetCommand(command_sock, 'state')
      Dada.logMsg(3, DL, "main: -> " + result + " " + response)

      if (result == "ok"):
        roaches_all_active = True 
    
    # if roaches are all active, bramdisk 'em
    if (roaches_all_active):
      time.sleep(1)
      Dada.logMsg(3, DL, "main: <- 'bramdisk'")
      result, response = Dada.sendTelnetCommand(command_sock, 'bramdisk')
      Dada.logMsg(3, DL, "main: -> " + result + " " + response)
      
    counter = 8
    while ((not quit_event.isSet()) and (counter > 0)):
      counter -= 1
      Dada.logMsg(3, DL, "main: sleeping")
      time.sleep(1)

  Dada.logMsg(3, DL, "main: command_sock.close()")
  command_sock.close()

  # join the command therad
  Dada.logMsg(2, DL, "main: joining command thread")
  command_thread.join()
  Dada.logMsg(2, DL, "main: command thread joined")

except:
  Dada.logMsg(-2, DL, "main: exception caught: " + str(sys.exc_info()[0]))
  print '-'*60
  traceback.print_exc(file=sys.stdout)
  print '-'*60
  quit_event.set()

Dada.logMsg(3, DL, "main: lock.acquire()")
lock.acquire()
Dada.logMsg(3, DL, "main: lock acquired")

for i in range(n_roach):
  roach_states[i] = QUIT
#roach_state = QUIT

Dada.logMsg(3, DL, "main: cond.notifyAll()")
cond.notifyAll()
Dada.logMsg(3, DL, "main: lock.release()")
lock.release()

# join threads
Dada.logMsg(2, DL, "main: joining control thread")
if (control_thread):
  control_thread.join()

Dada.logMsg(2, DL, "main: joining roach threads")
for i in range(n_roach):
  Dada.logMsg(2, DL, "main: joining roach thread["+str(i)+"]")
  roach_threads[i].join()

Dada.logMsg(1, DL, "main: joining plot thread")
if (plot_thread):
  plot_thread.join()

Dada.logMsg(2, DL, "main: exiting")

Dada.logMsg(1, DL, "STOPPING SCRIPT")
# exit
sys.exit(0)


