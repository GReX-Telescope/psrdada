#!/bin/bash -f

#
# merges info from $2/$1=<UTC>_bpsr.log into $2/<beam>/obs.start
#

dir=$1
file=$2

cd $dir

for beam in `seq 1 13` ; do
  beam=`echo $beam | awk '{printf("%02d",$1)}'`
  grep -e 'beam_all' -e "beam_$beam" $file >! tcs.$beam
  cat tcs.$beam >> $beam/obs.start

  rm tcs.$beam
done

