#!/bin/csh

ipcs | grep $USER | awk '{print "ipcrm -s "$2}' | csh
ipcs | grep $USER | awk '{print "ipcrm -m "$2}' | csh

