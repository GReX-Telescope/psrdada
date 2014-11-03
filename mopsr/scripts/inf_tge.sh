#!/bin/sh -v
sudo /sbin/ethtool -S eth2 > rxa.txt; sleep 1; sudo /sbin/ethtool -S eth2 > rxb.txt

perl -e 'undef $/; 
use Data::Dumper;
open RXA, "<rxa.txt"; %rxa = <RXA> =~ /(\S+): *(\d+)/smg;
open RXB, "<rxb.txt"; %rxb = <RXB> =~ /(\S+): *(\d+)/smg;
#print Dumper \%rxb;exit;
foreach $key (sort keys %rxa){printf "%-30s %12u %12u %12u\n", $key, $rxa{$key}, $rxb{$key}, $rxb{$key}-$rxa{$key}};
printf "rx_packets+dropped_link_overflow %u\n", $rxb{"rx_packets"}-$rxa{"rx_packets"}+$rxb{"dropped_link_overflow"}-$rxa{"dropped_link_overflow"};
printf "rx_gbps %.2f\n", ($rxb{rx_bytes}-$rxa{rx_bytes})*8/1e9;
'




