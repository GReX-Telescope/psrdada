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

def generateDummyData(size=2048):
    """ Some noisy sine waves """
    return sin_wave + np.random.random_integers(-32,32,size)

def main():
    # Simple option parsing so you can change the IP address via command line
    p = OptionParser()
    
    p.add_option("-i", "--ipaddr", dest="UDP_IP", help="UDP IP address", default="127.0.0.1", type="string")
    p.add_option("-p", "--port", dest="UDP_PORT", help="UDP IP port", default="12345", type="int")
    p.add_option("-n", "--numpkts", dest="num_pkts", help="No. of packets to send", default=500, type="int")
    p.add_option("-N", "--numbeams", dest="num_beams", help="No. of beams", default=1, type="int")
    p.add_option("-s", "--sleeptime", dest="sleep_time", help="Sleep time between packets", default=0, type="float")
    
    opts, args = p.parse_args(sys.argv[1:])
    UDP_IP, UDP_PORT = opts.UDP_IP, opts.UDP_PORT
    

    # Create packet header
    header = pkt.Header()
    payload = pkt.Payload()
    
    header.version     = 1     # Packet structure version numnber
    header.beam_id     = 1     # Antenna ID, unsigned integer
    header.pkt_cnt     = 1     # Packet counter / ID
    header.diode_state = 0     # Noise diode ON / OFF
    header.freq_state  = 0     # Frequency switching state
    header.makeHeader()

    # Set up the packet as a UDP datagram
    print "UDP target IP:", UDP_IP
    print "UDP target port:", UDP_PORT
    sock = socket.socket( socket.AF_INET, socket.SOCK_DGRAM ) # UDP

    # PACKET FLOOD LOOP
    # Loop through, sending data.
    # Nb: Too little sleep and the server will drop packets
    num_pkts  = opts.num_pkts # How many packets to send
    sleep_time= opts.sleep_time  # How long to sleep inbetween packets
    num_beams = opts.num_beams # Number of beams to generate dummy data for
    
    beam_ids = [x for x in range(1,num_beams+1)]
    
    print "Sending %i packets, interpacket sleep time %f seconds..."%(num_pkts, sleep_time)
    time1    = datetime.datetime.now()    
    for i in range(0,num_pkts):
        
        # Change header values
        header.pkt_cnt = i + 1
        header.makeHeader() 
        
        # Create packet payload        
        payload.xpol_data = generateDummyData(2048)
        payload.ypol_data = generateDummyData(2048)
        payload.makePayload()        
        
        b = bytes()
        MESSAGE= b.join((header.header, payload.payload))
        
        sock.sendto( MESSAGE, (UDP_IP, UDP_PORT) )
        time.sleep(sleep_time)   
    time2 = datetime.datetime.now()

    t_taken = str(time2-time1)
    print "%i packets sent in %s seconds."%(num_pkts, t_taken)

if __name__ == '__main__':
    
    # Setup some test vectors
    freq = 2.5e9
    amp  = 32.0
    t    = np.linspace(0,1,2048) / 25e6
    phase = np.random.random()
    sin_wave   = amp*  np.sin(2*np.pi*freq*t + phase)
    sin_wave2  = amp/2*np.sin(2*np.pi*freq*2*t + phase)
    sin_wave3  = amp/2*np.sin(2*np.pi*freq/2*t + phase)
    sin_wave   = sin_wave + sin_wave2 + sin_wave3
    
    main()
