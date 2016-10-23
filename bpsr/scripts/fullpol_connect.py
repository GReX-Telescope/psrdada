#!/usr/bin/env python

import corr,time,numpy,struct,sys
import Dada,Bpsr
import fullpol as spec

############################################################################### 
#
# main
#

cfg = Bpsr.getConfig()
rid = "0"
dl = 3
acc_len = 25

Dada.logMsg(1, dl, "main: connectRoach")
result, fpga = Bpsr.connectRoach(dl, rid)
if (not result == "ok"):
  Dada.logMsg(0, dl, "["+rid+"] main: could not connect to ROACH")
  sys.exit(1)

result, response = spec.accLenRoach(dl, fpga, acc_len, rid)
if (not result == "ok"):
  Dada.logMsg(0, dl, "["+rid+"] main: could not set acc_len on ROACH")
  sys.exit(1)

Dada.logMsg(1, dl, "main: setLevels")
result, response = spec.levelSetRoach(dl, fpga, rid)
Dada.logMsg(1, dl, "main: " + result + " " + response)

real_bit_window = 1
p1 = spec.dumpRoach(dl, fpga, 'scope_pol1_Shared_BRAM', 'I', 1)
p2 = spec.dumpRoach(dl, fpga, 'scope_pol2_Shared_BRAM', 'I', 1)
spec.plot(p1, p2, real_bit_window,True, False)

Dada.logMsg(1, dl, "main: crossLevelSetRoach")
result, response = spec.crossLevelSetRoach(dl, fpga, rid)
Dada.logMsg(1, dl, "main: " + result + " " + response)

complex_bit_window = 3
r1 = spec.dumpRoach(dl, fpga, 'scope_crossR1_Shared_BRAM', 'i', 1)
c1 = spec.dumpRoach(dl, fpga, 'scope_crossC1_Shared_BRAM', 'i', 1)
spec.plot(r1, c1, complex_bit_window, True, True)

sys.exit(0)

Dada.logMsg(1, dl, "main:  reg_coeff_pol1 " + str(pol1_gain))
fpga.write_int('reg_coeff_pol1', pol1_gain)
Dada.logMsg(1, dl, "main:  reg_coeff_pol2 " + str(pol2_gain))
fpga.write_int('reg_coeff_pol2', pol2_gain)
Dada.logMsg(2, dl, "["+rid+"] reg_output_bitselect " + str(real_bit_window))
fpga.write_int('reg_output_bitselect', real_bit_window)

gain = 128
Dada.logMsg(1, dl, "main: setComplexGains("+str(gain)+", "+str(complex_bit_window)+")")
result, response = spec.setComplexGains(dl, fpga, rid, gain, complex_bit_window)
Dada.logMsg(1, dl, "main: " + result + " " + response)

time.sleep(1)

p1 = spec.dumpRoach(dl, fpga, 'scope_pol1_Shared_BRAM', 'I', 1)
p2 = spec.dumpRoach(dl, fpga, 'scope_pol2_Shared_BRAM', 'I', 1)
spec.plot(p1, p2, real_bit_window,True, False)

r1 = spec.dumpRoach(dl, fpga, 'scope_crossR1_Shared_BRAM', 'i', 1)
c1 = spec.dumpRoach(dl, fpga, 'scope_crossC1_Shared_BRAM', 'i', 1)
spec.plot(r1, c1, complex_bit_window, True, True)

#sys.exit(0)


