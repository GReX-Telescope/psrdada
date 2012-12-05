#!/usr/bin/env python
# encoding: utf-8
"""
shinPacket.py

Class for HIPSR narrowband packet. Packet is split into Header()
and Payload(). 

Packet structure
Ethernet header:    16 bytes
IP header:          20 bytes
UDP header:         8 bytes
Packet Header:      3 bytes, 2 bits     <- created by this script!
Data:               up to 9000-48 bytes <- created by ROACH

"""
import sys, os, struct
import numpy as np

class Header(object):
    """Class for a HIPSR Narrowband UDP packet header."""
    def __init__(self):
        super(Header, self).__init__()
        
        # Set up the default header values
        self.version     = 1     # Packet structure version numnber
        self.beam_id     = 1     # Antenna ID, unsigned integer
        self.pkt_cnt     = 1     # Packet counter / ID
        self.diode_state = 0     # Noise diode ON / OFF
        self.freq_state  = 0     # Frequency switching state
        
        # Set up struct format characters
        # ! = network byte order (big endian)
        # B = unsigned char,        1 byte  (8 bits)
        # b = signed char,          1 byte  (8 bits)
        # H = unsigned short,       2 bytes (16 bits)
        # L = unsigned long,        4 bytes (32 bits)
        # Q = unsigned long long,   8 bytes (64 bits)
        # ? = boolean               1 bit
        self.header_fmt    = '!BBLBB'
                
        # Create default header
        self.makeHeader()
    
    def __repr__(self):
        to_print = (self.version, self.beam_id, self.pkt_cnt, self.diode_state, self.freq_state)
        a = "---------------------------------------------------------\n"
        b = "|version | beam_id | pkt_cnt | diode_state | freq_state |\n"
        c = "|    %3d |     %3d |     %3d |         %3d |        %3d |\n"%to_print
        d = "---------------------------------------------------------"
        return a + b + c + d
        
    def makeHeader(self):
        """Repacks a header after header values are changed."""
        self.header = struct.pack(
                    self.header_fmt, 
                    self.version,
                    self.beam_id,
                    self.pkt_cnt,
                    self.diode_state,
                    self.freq_state
                    )
    
    def unpackHeader(self, rx_header):
        """ Unpacks a header and updates header values """
        unpacked = False
        try:
            unpacked_data = struct.unpack(self.header_fmt, rx_header)
            unpacked = True
        except:
            print "warning: could not unpack header values"
        if unpacked:
            try: 
                self.header      = rx_header 
                self.version     = unpacked_data[0]     # Packet structure version numnber
                self.beam_id     = unpacked_data[1]     # Beam ID (0-13)
                self.pkt_cnt     = unpacked_data[2]     # Packet counter / ID
                self.diode_state = unpacked_data[3]     # Noise diode ON / OFF
                self.freq_state  = unpacked_data[4]     # Frequency switching state
            except:
                print "warning: could not convert unpacked data to headers"
        

class Payload(object):
    """Payload (ie. data) for the UDP packet."""
    def __init__(self):
        super(Payload, self).__init__()
        self.payload_fmt = '!4096b'
        self.xpol_data  = np.zeros(2048)
        self.ypol_data  = np.zeros(2048)
        self.payload    = struct.pack(self.payload_fmt,*np.zeros(4096))
    
    def __repr__(self):
        a = "x-pol: %s ...\n"%repr(self.xpol_data[0:10])
        b = "y-pol: %s ...\n"%repr(self.ypol_data[0:10])
        return a + b
    
    def makePayload(self):
        """Recreates a payload given data change"""
        data = np.append(self.xpol_data, self.ypol_data)
        self.payload = struct.pack(self.payload_fmt, *data)
    
    def unpackPayload(self, rx_data):
        """ Unpacks data payload and updates data array values """
        try:
            unpacked_data = struct.unpack(self.payload_fmt, rx_data)
            self.xpol_data = unpacked_data[:2048]
            self.ypol_data = unpacked_data[2048:]
            self.payload   = rx_data
        except:
            print "Warning: could not unpack payload"

    def getSize(self):
      return 4096
