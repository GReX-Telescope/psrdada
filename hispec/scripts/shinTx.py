#!/usr/bin/env python
# encoding: utf-8
"""
shinTx.py
---------

Sends a HIPSR Narrowband formatted UDP packet to a given IP/port. 
First, a headeris generated, then data is appended, then it is 
converted to a struct and sent over UDP.

This script has a sister, shinRx.py, which receives and unpacks a
HIPSR Narrowband  UDP packet. 
"""

import socket, sys, datetime, time
import numpy as np
from optparse import OptionParser

# import packet class
import shinPacket as pkt 

def main():
    # Simple option parsing so you can change the IP address via command line
    p = OptionParser()
    opts, args = p.parse_args(sys.argv[1:])
    if args ==[]:   
        UDP_IP = "127.0.0.1"
        UDP_PORT = 12345
    else:           
        UDP_IP, UDP_PORT = args[0].split(":")
        UDP_PORT = int(UDP_PORT)

    # Create packet header
    header = pkt.Header()
    header.version     = 1     # Packet structure version numnber
    header.beam_id     = 1     # Antenna ID, unsigned integer
    header.pkt_cnt     = 1     # Packet counter / ID
    header.diode_state = 0     # Noise diode ON / OFF
    header.freq_state  = 0     # Frequency switching state
    header.makeHeader()

    # Set up the packet as a UDP datagram
    
    print "UDP target IP:", UDP_IP
    print "UDP target port:", UDP_PORT

    # Create socket connection
    sock = socket.socket( socket.AF_INET, socket.SOCK_DGRAM ) # UDP

    # PACKET FLOOD LOOP
    # Loop through, sending data.
    # Nb: Too little sleep and the server will drop packets
    num_seconds = 5
    data_rate = 5000000     # 5 MiB/s
    payload = pkt.Payload();
    num_pkts_per_second = data_rate / payload.getSize()
    print "payload size=%i, num_pkts_per_second=%f"%(payload.getSize(), num_pkts_per_second)
    sleep_time = 1.0 / num_pkts_per_second
    num_pkts = num_seconds * num_pkts_per_second
    #num_pkts  = 260         # How many packets to send
    #sleep_time= 0.001       # How long to sleep inbetween packets

    beam_ids = [x for x in range(1,13+1)]
    
    print "Sending %i packets, interpacket sleep time %f seconds..."%(num_pkts, sleep_time)
    time1    = datetime.datetime.now()    

    for i in range(0,num_pkts):

        print "sending pkt " + str(i)
        
        # Change header values
        header.beam_id = beam_ids[i%13]
        header.pkt_cnt = i + 1
        
        if header.diode_state is 0: 
            header.diode_state = 1
        else: 
            header.diode_state = 0
        
        header.makeHeader()    
        # Create packet payload
        # payload = pkt.Payload()
        
        payload.xpol_data = np.random.random_integers(-2**7+1, 2**7-1, size=2048)
        payload.ypol_data = np.random.random_integers(-2**7+1, 2**7-1, size=2048)
        payload.makePayload()        
        
        MESSAGE= header.header + payload.payload
        
        sock.sendto( MESSAGE, (UDP_IP, UDP_PORT) )
        time.sleep(sleep_time)   
    time2 = datetime.datetime.now()

    t_taken = str(time2-time1)
    print "%i packets sent in %s seconds."%(num_pkts, t_taken)

if __name__ == '__main__':
    main()
