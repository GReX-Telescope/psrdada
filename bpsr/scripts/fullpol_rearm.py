#!/usr/bin/env python

import corr,time,numpy,struct,sys
import Dada,Bpsr,fullpol

############################################################################### 
#
# main
#

cfg = Bpsr.getConfig()
rid = "0"
dl = 3
acc_len = 25

Dada.logMsg(1, dl, "main: connectRoach")
result, fpga = fullpol.connectRoach(dl, rid)
if (not result == "ok"):
  Dada.logMsg(0, dl, "["+rid+"] main: could not connect to ROACH")
  sys.exit(1)

Dada.logMsg(1, dl, "main: setLevels")
result, response = fullpol.levelSetRoach(dl, fpga, rid)
Dada.logMsg(1, dl, "main: " + result + " " + response)

Dada.logMsg(1, dl, "main: crossLevelSetRoach")
result, response = fullpol.crossLevelSetRoach(dl, fpga, rid)
Dada.logMsg(1, dl, "main: " + result + " " + response)

curr_time = int(time.time())
next_time = curr_time
Dada.logMsg(2, dl, "main: waiting for 1 second boundary")
while (curr_time == next_time):
  next_time = int(time.time())

Dada.logMsg(2, dl, "main: sleeping 0.5 seconds")
time.sleep(0.5)
utc_start = Dada.getUTCDadaTime(1)
Dada.logMsg(2, dl, "main: UTC_START=" + utc_start)

fpga.write_int('reg_arm',0)
fpga.write_int('reg_arm',1)

print utc_start
