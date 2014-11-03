<?PHP

define ("RX_MONITOR_PATH", "/data/mopsr/monitor/rx"); 
define ("RX_MOINTOR_PLOT", "/home/dada/linux_64/bin/mopsr_rxplot");

$rx_id = $_GET["rx_id"];
$type = $_GET["type"];
$res = $_GET["res"];
$mod = $_GET["mod"];

# find the most recent bin file for this RX_ID
$cmd = "find ".RX_MONITOR_PATH." -name '2???-??-??-??:??:??.".$rx_id.".bin' -printf '%f\n' | sort -n | tail -n 1";
$array = Array();
$bin_file = exec ($cmd, $array, $rval);

if (!file_exists(RX_MONITOR_PATH."/".$bin_file))
{
  $array[0] = "Could not find file matching '2???-??-??-??:??:??.".$rx_id.".bin'";
  plotError($array);
}
else
{
  $cmd = RX_MOINTOR_PLOT." -m ".$mod." -p ".$type." -g ".$res." ".RX_MONITOR_PATH."/".$bin_file;
  $output = array();
  $return_val = 0;

  # Test the execution to see if it will work
  $c = "source /home/dada/.dadarc; ".$cmd." -D /null 2>&1";
  $lastline = exec($c, $output, $return_val);
  # If an error ocurred, generate a PHP image containing the error text
  if (($lastline != "") || (count($output) > 0))
  {
    plotError($output);
  }
  else
  {
    $c =  "source /home/dada/.dadarc; ".$cmd." -D -/PNG";
    header('Content-Type: image/png');
    header('Content-Disposition: inline; filename="image.png"');
    passthru("source /home/dada/.dadarc; ".$c);
  }
}

function plotError($output)
{
  putenv('GDFONTPATH=' . realpath('.'));
  $font = "Arial";

  $text = "";
  for ($i=0; $i<count($output); $i++) {
    for ($j=0; $j<strlen($output[$i]);$j++) {
       $text .= $output[$i][$j];
      if (($j > 0) && ($j % 40 == 0)) {
        $text .= "\n";
      }
    }
    $text .= "\n";
  }
  header('Content-type: image/png');
  $im = imagecreatetruecolor(240, 180);
  $white = imagecolorallocate($im, 255, 255, 255);
  $grey = imagecolorallocate($im, 128, 128, 128);
  $black = imagecolorallocate($im, 0, 0, 0);
  imagefilledrectangle($im, 0, 0, 240, 180, $white);
  // Replace path by your own font path
  $font = '/usr/share/fonts/dejavu-lgc/DejaVuLGCSerif.ttf';

  // Add some shadow to the text
  //imagettftext($im, 20, 0, 11, 21, $grey, $font, $text);

  // Add the text
  imagettftext($im, 8, 0, 10, 20, $black, $font, $text);

  // Using imagepng() results in clearer text compared with imagejpeg()
  imagepng($im);
  imagedestroy($im);
} 
