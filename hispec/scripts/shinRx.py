#!/usr/bin/env python
# encoding: utf-8
"""
shinRx.py
---------

Receives a HIPSR Narrowband UDP packet on a given IP/port, and unpacks the
packet, then plots it. Only plots a single packet at a time. 

This script has a sister, shinTx.py, which packs and sends an 
HIPSR Narrowband UDP packet. 
"""

import socket, struct, sys, matplotlib

import pylab as plt
import numpy as np

# import HIPSR packet class
import shinPacket as pkt

# Simple option parsing so you can change the IP address via command line
from optparse import OptionParser
p = OptionParser()
opts, args = p.parse_args(sys.argv[1:])
if args ==[]:   
    UDP_IP = "127.0.0.1"
    UDP_PORT = 12345
else:           
    UDP_IP, UDP_PORT = args[0].split(":")
    UDP_PORT = int(UDP_PORT)

def main():
    # Create UDP socket
    print "Running on %s:%s"%(UDP_IP, UDP_PORT)
    
    sock = socket.socket( socket.AF_INET, socket.SOCK_DGRAM )
    sock.bind( (UDP_IP,UDP_PORT) )
    
    # Lie in waiting
    print "Waiting for packets..."
    count =0
    
    while True:
        # Set up to listen for UDP packets
        rx_data, addr = sock.recvfrom(8192) # buffer size is 8192 bytes
        count += 1
        print "Packet count: ",count
        
        header  = pkt.Header()
        payload = pkt.Payload()
        
        try:
            print len(rx_data)
            header.unpackHeader(rx_data[0:8])  
            payload.unpackPayload(rx_data[8:])
            print header
            print payload
        except:
            raise

    
        
    
if __name__ == '__main__':
    main()