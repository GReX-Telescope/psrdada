#

from numpy import *
import sys

# gen_info.py
# A script that grabs pertinent information from the ppta_cat.dat file
# and formats it so that it can easily be plotted in sn_hist2.py.

def gen_info(psr, ppta_cat):
  cat_dat = loadtxt(ppta_cat,dtype='string')
  index = where(cat_dat[:,0] == psr)[0][0]
  
  out_str = ["PSR "+str(cat_dat[index,0]),
             "P0:     "+str(cat_dat[index,1]),
             "DM:     "+str(cat_dat[index,2]),
             "GL:     "+str(cat_dat[index,3]),
             "GB:     "+str(cat_dat[index,4]),
             "PB:     "+str(cat_dat[index,5]),
             "S1400:  "+str(cat_dat[index,6])]

  return out_str

# percentile.py
# Reads in snr array and returns the percentile of its latest member.

def percentile(sn_arr,sn_val):
  last_sn = sn_val
  sn_arr.append(last_sn)
  sn_sort = sorted(sn_arr)
  for i in range(len(sn_sort)):
    if sn_sort[i] == last_sn:
      index = i

  percentile = (float(index)/len(sn_arr))*100
  #print "Last epoch has SNR greater than "+str(percentile)[0:5]+"% of others."
