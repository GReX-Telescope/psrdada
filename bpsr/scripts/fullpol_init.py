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
acc_len = 1024

result, fpga = fullpol.initializeRoach (dl, acc_len, rid, cfg)
if result == "ok":
  Dada.logMsg(1, dl, "main: initalizeRoach worked!")
else:
  Dada.logMsg(1, dl, "main: initalizeRoach failed!")

