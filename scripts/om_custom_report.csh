#!/bin/tcsh

if ( -f /usr/sbin/racadm ) then
  set racadm = /usr/sbin/racadm
  set omreport = /usr/bin/omreport
else if ( -f /opt/dell/srvadmin/sbin/racadm ) then
  set racadm = /opt/dell/srvadmin/sbin/racadm
  set omreport = /opt/dell/srvadmin/bin/omreport
else
  set racadm = ""
  set omreport = ""
endif

set host = `hostname -s`
set os_arch = `uname -i`
set kernel_ver = `uname -r`
set os = `cat /etc/issue | head -n 1 | awk '{print $1" "$3}'`

if ( $omreport != "" ) then

  set raid_fw = `$omreport storage controller | grep "^Firmware Version" | awk '{print $4}'`
  set raid_drv = `$omreport storage controller | grep "^Driver Version" | awk '{print $4}'`
  set bios = `$omreport chassis bios | grep "^Version" | awk '{print $3}'`
  set service_tag = `$omreport chassis info | grep "^Chassis Service Tag" | awk '{print $NF}'`
  set rac_fw = `$omreport chassis info | grep "DRAC" | awk '{print $NF}'`
  set disk0_id = `$omreport storage pdisk controller=0 pdisk=0:0:0 | grep "^Product ID" | awk '{print $NF}'`
  set disk0_fw = `$omreport storage pdisk controller=0 pdisk=0:0:0 | grep "^Revision" | awk '{print $NF}'`
  set disk1_id = `$omreport storage pdisk controller=0 pdisk=0:0:1 | grep "^Product ID" | awk '{print $NF}'`
  set disk1_fw = `$omreport storage pdisk controller=0 pdisk=0:0:1 | grep "^Revision" | awk '{print $NF}'`
  set omsa_ver = `$racadm version |& grep "^RACADM" | awk '{print $3}'`

else

  set raid_fw = --
  set raid_drv = --
  set bios = --
  set service_tag = --
  set rac_fw = `ipmitool mc info | grep "Firmware Revision" | awk '{print $NF}'`
  set disk0_id = --
  set disk0_fw = --
  set disk1_id = --
  set disk1_fw = --
  set omsa_ver = --

endif

echo os $os
echo kernel "$kernel_ver ($os_arch)"
echo raid_fw $raid_fw
echo raid_drv $raid_drv
echo bios $bios
echo service_tag $service_tag
echo rac_fw $rac_fw
echo disk0_id $disk0_id
echo disk0_fw $disk0_fw
echo disk1_id $disk1_id
echo disk1_fw $disk1_fw
echo omsa_ver $omsa_ver

