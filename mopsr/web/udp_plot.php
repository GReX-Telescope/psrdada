<?PHP

define ("UDP_MONITOR_PLOT", "/home/dada/linux_64/bin/mopsr_dumpplot");

$pfb_id = $_GET["pfb_id"];
$pfb_input_id = $_GET["pfb_input_id"];
$host = $_GET["host"];
$type = $_GET["type"];
$res = $_GET["res"];
$schan = isset($_GET["schan"]) ? $_GET["schan"] : 0;
$echan = isset($_GET["echan"]) ? $_GET["echan"] : "39";
$echan = isset($_GET["echan"]) ? $_GET["echan"] : "319";

//$schan = isset($_GET["schan"]) ? $_GET["schan"] : "19";
//$echan = isset($_GET["echan"]) ? $_GET["echan"] : "19";

define ("UDP_MONITOR_PATH", "/data/mopsr/monitor/udp/".$pfb_id); 

# find the most recent bin file for this PFB_ID
$cmd = "ssh mpsr@".$host." \"find ".UDP_MONITOR_PATH." -name '2???-??-??-??:??:??.".$pfb_id.".dumped' | sort -n | tail -n 1\"";
$array = Array();
$dumped_file = exec ($cmd, $array, $rval);

if ($rval != 0)
{
  $array[0] = "Could not find file matching '2???-??-??-??:??:??.".$pfb_id.".dumped'";
  plotError($array);
}
else
{
  $cmd = "ssh mpsr@".$host." \"".UDP_MONITOR_PLOT." -t ".$type." -g ".$res." -a ".$pfb_input_id." -c ".$schan." -d ".$echan." ".$dumped_file." -D -/PNG\"";
  $output = array();
  $return_val = 0;

  # Test the execution to see if it will work
  # $c = "source /home/dada/.dadarc; ".$cmd." -D /null 2>&1";
  # $lastline = exec($c, $output, $return_val);
  # If an error ocurred, generate a PHP image containing the error text
  # if (($lastline != "") || (count($output) > 0))
  # {
  #   print_r($output);
  #   #plotError($output);
  # }
  # else
  {
    $c =  "source /home/dada/.dadarc; ".$cmd;
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
