package Bpsr;

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Time::HiRes qw(usleep ualarm gettimeofday tv_interval);
use Math::BigInt;
use Math::BigFloat;
use strict;
use vars qw($VERSION @ISA @EXPORT @EXPORT_OK);
use Sys::Hostname;
use Time::Local;
use POSIX qw(setsid);
use Dada;

require Exporter;
require AutoLoader;

@ISA = qw(Exporter AutoLoader);

@EXPORT_OK = qw(
  &getBpsrConfig
  &getBpsrConfigFile
  &set_ibob_levels
  &start_ibob
);

$VERSION = '0.01';

my $DADA_ROOT = $ENV{'DADA_ROOT'};

use constant MAX_VALUE        => 1;
use constant AVG_VALUES       => 2;


sub getBpsrConfig() {
  my $config_file = getBpsrCFGFile();
  my %config = Dada->readCFGFileIntoHash($config_file, 0);
  return %config;
}

sub getBpsrCFGFile() {
  return $DADA_ROOT."/share/bpsr.cfg";
}

#
# Sets ACC_LEN, adjusts iBOB levels and selects best bits
#
sub set_ibob_levels($$$$) {
               
  my ($module, $sock, $acc_len, $eol) = @_;

  my $result = "";
  my $response = "";  

  # Set the Accumulation length
  ($result, $response) = setAccLen("Bpsr", $sock, $acc_len, $eol);

  # Set the initial scale factor to Unity
  my $s0 = 4096;    # scale for pol 0
  my $s1 = 4096;    # scale for pol 1
  print $sock "regwrite reg_coeff_pol1 ".$s0.$eol;
  print $sock "regwrite reg_coeff_pol2 ".$s1.$eol;

  my $bit;                # Selected bits [0,1,2,3]
  my $f_bit = -1;         # No forced bit selection
  my $lvls = AVG_VALUES;  # Method for determining average power

  # Find the best scaling factors
  my $j=0;
  for ($j=0; $j < 3; $j++ ) {
    ($s0, $s1, $bit) = getNewScaleCoeff("Bpsr", $sock, $s0, $s1, $lvls, $f_bit, $eol);
  }

  # Now select the bits we want
  print $sock "regwrite reg_output_bitselect ".$bit.$eol;

  return ($s0, $s1, $bit);
  
}

#
# Starts the ibob onn the next 1 second boundary and 
# and returned the UTC of the start
#
sub start_ibob($$$) {

  my ($module, $sock, $eol) = @_;

  my $curr_time = time();
  my $prev_time = $curr_time;

  # A busy loop to continually get the time until
  # we are on a 1 second boundary;
  while ($curr_time < ($prev_time + 1)) {
    $curr_time = time();
  }

  # Sleep for 400 ms 400 ms 400 ms 400 ms 
  my @t1 = gettimeofday();
  Time::HiRes::usleep(400000);
  my @t2 = gettimeofday();

  # Issue to re-arm command
  print $sock "regwrite reg_arm 0".$eol;
  print $sock "regwrite reg_arm 1".$eol;

  # Clear the socket
  my @output = getResponse("Bpsr", $sock, 0.1);

  return $t2[0];

}


#
# Sets reg_acclen to (acc_len-1)
#
sub setAccLen($$$$) {

  my ($module, $socket, $acc_len, $eol) = @_;

  my $result = "ok";
  my $response = "";

  my $n = 100;
  my $k = $acc_len * 512;
  my $x = 1024;
  my $sync_period = $n*lcm("Bpsr",$k,$x);

  if ($socket) {
    print $socket "regwrite reg_acclen ".($acc_len-1).$eol;
    print $socket "regwrite reg_sync_period ".$sync_period.$eol;
    $response = $sync_period;
  } else {
    $result = "fail";
    $response = "socket was invalid";
  }

  return ($result, $response);

}
                                                                                                                                          
#
# Calculate the Greatest Common Denominator (GCD)
#
sub gcd($$$) {

  my ($module,$a,$b)=@_;

  while ($a!=$b) {
    if ($a>$b) {
      $a-=$b;
    } else {
      $b-=$a;
    }
  }
  return $a;

}
                                                                                                                                          
#
# Calculate the Least Common Multiple (LCM)
#
sub lcm {
  my ($module,$a,$b) = @_;
  return $a*$b/&gcd("Bpsr",$a,$b);
}

#
# Reads a reply from the ibob interface
#
sub getResponse($$$) {

  my ($module, $handle, $timeout) = @_;

  my $done = 0;
  my $read_set = new IO::Select($handle);
  my @output = ();
  my $rh = 0;
  my $line = 0;
  my $i=0;

  my $old_buffer = "";

  while (!$done) {

    my ($readable_handles) = IO::Select->select($read_set, undef, undef, $timeout);
    $done = 1;

    my $buffer = "";
    my $to_process = "";

    foreach $rh (@$readable_handles) {

      $buffer = "";
      $to_process = "";

      # read 1024 bytes at a time
      recv($handle, $buffer, 1023, 0);

      $buffer =~ s/\r//g;

      # If we had unprocessed text from before
      if (length($old_buffer) > 0) {
        $to_process = $old_buffer.$buffer;
        $old_buffer = "";
      } else {
        $to_process = $buffer;
      }
     
      # Check if we have a complete line in the data to process
      if ($to_process =~ m/\n/) {

         my @arr = split(/\n/, $to_process);

         # If there is only a newline...
         if ($#arr < 0) {
           push(@output, $arr[0]);

         } else {
         
           for ($i=0; $i<$#arr; $i++) {
             push (@output, $arr[$i]);
           }

           # If the final character happened to be a newline,
           # we should output the final line, otherwise, add
           # it to the old_buffer
           if ($to_process =~ m/(.)*\n$/) {
             push (@output, $arr[$#arr]);
           } else {
             $old_buffer = $arr[$#arr];
           }
         }

      } else {
        $old_buffer .= $to_process;
      }

      $done = 0;

    }
  }
  if (length($old_buffer) >= 0) {
    push(@output,$old_buffer);
  }

  return @output;

}


sub getGains($\@) {
                                                                                                  
  my ($module, $outputRef) = @_;
                                                                                                  
  my @output = @$outputRef;
                                                                                                  
  my $i=0;
  my @arr = ();
  my @gains = ();
                                                                                                  
  for ($i=0;$i<=512; $i++) {
    if (length($output[$i]) > 0) {
      @arr = split(/ \/ | \-> /,$output[$i]);
      push(@gains,$arr[4]);
    }
  }
  return @gains;
}

sub getSingleValue($$\@) {
  
  my ($module, $type, $valsRef) = @_;

  my @vals = @$valsRef;

  my $value = Math::BigInt->new(0);
  my $nvals = 0;
  my $i=0;

  # Just return the maximum value
  if ($type = MAX_VALUE) {

    for ($i=0; $i<=$#vals; $i++) {
      if ($vals[$i] > $value) {
        $value = $vals[$i];
      }
    }

  # Get the average value
  } elsif ($type == AVG_VALUES) {

    for ($i=0; $i<=$#vals; $i++) {
      $value += $vals[$i];
      $nvals++;
    }
    $value->bdiv($nvals);

  } else {
    $value = 0;
  }

  return int($value);
}

sub getNewScaleCoeff($$$$$$$) {
  
  my ($module, $handle, $scale1, $scale2, $type, $forced_bits, $eol) = @_;

  my @bitsel_min = qw(0 256 65536 16777216);
  my @bitsel_mid = qw(64 8192 2097152 536870912);
  my @bitsel_max = qw(255 65535 16777215 4294967295);
  my @output = ();

  # Set the scaling coeffs for each pol.
  print $handle "regwrite reg_coeff_pol1 ".$scale1.$eol;
  print $handle "regwrite reg_coeff_pol2 ".$scale2.$eol;

  # Get values for pol0
  print $handle "bramdump scope_output1/bram".$eol;
  @output = getResponse("Bpsr", $handle, 0.5);
  my @pol1 = getGains("Bpsr", @output);
  my $pol1_val = getSingleValue("Bpsr", $type, @pol1);

  @output = ();

  # Get values for pol1
  print $handle "bramdump scope_output3/bram".$eol;
  @output = getResponse("Bpsr", $handle, 0.5);
  my @pol2 = getGains("Bpsr", @output);
  my $pol2_val = getSingleValue("Bpsr", $type, @pol2);

  # Work out how to change the scaling factor to make the polarisations
  # roughly the same
  my $val;

  if ($type == MAX_VALUE) { # choose the lowest pol as the representative value (more bit selection downwards)
    $val = Math::BigFloat->new( ($pol1_val > $pol2_val) ? $pol2_val : $pol1_val);
  } else {
    $val = Math::BigFloat->new( ($pol1_val + $pol2_val) / 2 );
  } 

  # Find which bit selection window we are currently sitting in
  my $i=0;
  my $current_window = 0;
  my $desired_window = 0;
  for ($i=0; $i<4; $i++) {
                                                                                                  
    # If average (max) value is in the lower half
    if (($val > $bitsel_min[$i]) && ($val <= $bitsel_mid[$i])) {
      $current_window = $i;
                                                                                                  
      if ($i == 0) {
        $desired_window = 0;
                                                                                                  
      } else {
        $desired_window = ($i-1);
      }
    }
                                                                                                  
    # If average (max)n value is in the upper half, simply raise to
    # the top of this window
    if (($val > $bitsel_mid[$i]) && ($val <= $bitsel_max[$i])) {
      $current_window = $i;
      $desired_window = $i;
    }
  }

  if (($forced_bits >= 0) && ($forced_bits <= 3)) {
    $desired_window = $forced_bits;
  } else {
    if ($desired_window == 3) {
      $desired_window = 2;
    }
  }
   
  if (($pol1_val eq 0) || ($pol2_val eq 0)) {
    $pol1_val = 1;
    $pol2_val = 1;
  }

  my $desired_val;

  $desired_val =  Math::BigFloat->new(($bitsel_max[$desired_window]+1) / 2);


  my $gain_factor1 = Math::BigFloat->new($desired_val / $pol1_val);
  my $gain_factor2 = Math::BigFloat->new($desired_val / $pol2_val);

  $gain_factor1->bsqrt();
  $gain_factor2->bsqrt();

  $gain_factor1 *= $scale1;
  $gain_factor2 *= $scale2;
  
  $gain_factor1->bfloor();
  $gain_factor2->bfloor();

  return ($gain_factor1, $gain_factor2, $desired_window);
  
}


__END__
