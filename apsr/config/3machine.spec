# a specification contains parameters common to all bands ...

FILE_SIZE    800000000
OBS_OVERLAP  0
TELESCOPE    PKS
SOURCE       PSR_0505+2020
NPOL         2
NBIT         8
NDIM         1
NCHANNEL     2
NBAND        1
TSAMP        0.0298023223876953
# nb. this should result in a 64 MB/s data rate...
RA           05:05:00
DEC          20:20:01
HDR_SIZE     4096

# ... as well as parameters specific to individual bands

Band0_FREQ         1300
Band0_BW           20
Band0_TARGET_NODES node01

Band1_FREQ         1400
Band1_BW           40
Band1_TARGET_NODES node02

Band2_FREQ         1500
Band2_BW           40
Band2_TARGET_NODES node03
