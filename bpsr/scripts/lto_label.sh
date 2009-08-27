#!/bin/bash

device="/dev/nst1"
project="P630"
prefix="HITRUN"

usage()
{
cat <<EOF

Usage: lto_label.sh [OPTION]

Known values for OPTION are:

  --device=TAPE      tape drive device name [default: $device]
  --project=PID      project ID [default: $project]
  --prefix=NAME      tape name prefix [default: $prefix]

EOF

exit $1
}

while test $# -gt 0; do

case "$1" in
    -*=*)
    arg=`echo "$1" | awk -F= '{print $1}'`
    optarg=`echo "$1" | awk -F= '{print $2}'`
      ;;

    *)
    arg=$1
    optarg=
    ;;
esac

case "$arg" in

    --device)
    device=$optarg
    ;;

    --project)
    project=$optarg
    ;;

    --prefix)
    prefix=$prefix
    ;;

    --help)
    usage 0
    ;;
    
    *)
    usage
    exit 1
    ;;

esac

shift

done

echo Tape device: $device
echo Project ID: $project
echo Label prefix: $prefix

number=$1
name=`echo $number | awk '{printf("HRE%03dS4",$1);}'`
mkdir -p /tmp/tapelableotron
cd /tmp/tapelableotron
touch $name
mt -f /dev/nst0 status
if [ $? -ne 0 ]; then
echo "Tape status check failed!"
exit
fi
mt -f /dev/nst0 rewind
echo "Labeling tape in /dev/nst0 with label $name"
echo "Press return to continue"
read ret
echo "Labeling tape"
tar -cf /dev/nst0 $name
echo "Ejecting tape"
mt -f /dev/nst0 rewoffl
rm $name
cd ..
rmdir tapelableotron
echo "Done"
exit
