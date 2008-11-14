#!/bin/csh -f

if ( "$1" == "" ) then

cat <<EOF

  apsr_rm.csh - use with loopssh to remove \$1* from /lfs/data0/apsr/scratch

  e.g. loopssh apsr 0 15 apsr_rm.csh 2008-10-17

EOF

  exit 0

endif

echo Removing /lfs/data0/apsr/scratch/$1*

rm -f /lfs/data0/apsr/scratch/$1*

