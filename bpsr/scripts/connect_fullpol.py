#!/usr/bin/env python

import corr,time,numpy,struct,sys
import Dada,Bpsr,fullpol

############################################################################### 
#
# main
#

def bramNormal():
  p1 = fullpol.bramDump(dl, fpga, 'scope_pol1_bram', 'I', 1)
  p2 = fullpol.bramDump(dl, fpga, 'scope_pol2_bram', 'I', 1)
  return (p1, p2)

cfg = Bpsr.getConfig()
rid = "0"
dl = 2
acc_len = 25

# configure ROACH 
Dada.logMsg(1, dl, "main: configureRoach")
result, fpga = fullpol.configureRoach (dl, acc_len, rid, cfg)
Dada.logMsg(1, dl, "main: " + result)

Dada.logMsg(1, dl, "main: write_int('reg_arm', 0)")
fpga.write_int('reg_arm', 0)
Dada.logMsg(1, dl, "main: write_int('reg_arm', 1)")
fpga.write_int('reg_arm', 1)

# set Levels
Dada.logMsg(1, dl, "main: setLevels")
result, response = fullpol.setLevels (dl, fpga, rid)
Dada.logMsg(1, dl, "main: " + result + " " + response)

Dada.logMsg(1, dl, "main:  reg_coeff_pol1 2048")
fpga.write_int('reg_coeff_pol1', 2048)
Dada.logMsg(1, dl, "main:  reg_coeff_pol2 1536")
fpga.write_int('reg_coeff_pol2', 1536)

gain = 2048
Dada.logMsg(1, dl, "main: setComplexGains("+str(gain)+")")
result, response = fullpol.setComplexGains(dl, fpga, rid, gain)
Dada.logMsg(1, dl, "main: " + result + " " + response)

time.sleep(1)

p1 = fullpol.bramDump(dl, fpga, 'scope_pol1_bram', 'I', 1)
p2 = fullpol.bramDump(dl, fpga, 'scope_pol2_bram', 'I', 1)
print p1
print p2
fullpol.plot(p1, p2)

sys.exit(0)

time.sleep(10)
gain = 2
while gain < 65536:
  
  Dada.logMsg(1, dl, "main: setGains("+str(gain)+")")
  result, response = fullpol.setGains(dl, fpga, rid, gain)
  Dada.logMsg(1, dl, "main: " + result + " " + response)

  time.sleep(10)
  gain *= 2

#Dada.logMsg(1, dl, "main: plotting r1, c1")
r1 = fullpol.bramDump(dl, fpga, 'scope_crossR1_bram', 'i')
c1 = fullpol.bramDump(dl, fpga, 'scope_crossC1_bram', 'i')
fullpol.plot(r1, c1)

#Dada.logMsg(1, dl, "main: plotting r2, c2")
r2 = fullpol.bramDump(dl, fpga, 'scope_crossR2_bram', 'i')
c2 = fullpol.bramDump(dl, fpga, 'scope_crossC2_bram', 'i')
fullpol.plot(r2, c2)

# now do some bramp dumps
#Dada.logMsg(1, dl, "main: bramDumping")
#pol1 = fullpol.bramDump(dl, fpga, 'scope_pol1_bram', 1)
#pol2 = fullpol.bramDump(dl, fpga, 'scope_pol2_bram', 1)
#r1   = fullpol.bramDump(dl, fpga, 'scope_crossR1_bram', 1)
#r2   = fullpol.bramDump(dl, fpga, 'scope_crossR2_bram', 1)
#c1   = fullpol.bramDump(dl, fpga, 'scope_crossC1_bram', 1)
#c2   = fullpol.bramDump(dl, fpga, 'scope_crossC2_bram', 1)

# Dada.logMsg(1, dl, "main: pol1")
# print pol1

# Dada.logMsg(1, dl, "main: scope_crossR1_bram")
# print r1






