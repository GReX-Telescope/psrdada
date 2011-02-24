<?PHP
$cmd = $_GET["cmd"];
$output = array();
$return_val = 0;

# Test the execution to see if it will work
$c = "source /home/dada/.bashrc; ".$cmd." -D /null 2>&1";
#echo $c."<BR>\n";
$lastline = exec($c, $output, $return_val);

# If an error ocurred, generate a PHP image containing the error text
if (($lastline != "") || (count($output) > 0)) {

  #$text = "1234567890123456789012345678901234567890\n";
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

# It seems to have worked ok, generate the real image
} else {
  $c =  "source /home/dada/.bashrc; ".$cmd." -D -/PNG";
  header('Content-Type: image/png');
  header('Content-Disposition: inline; filename="image.png"');
  passthru("source /home/dada/.bashrc; ".$c);
} 
