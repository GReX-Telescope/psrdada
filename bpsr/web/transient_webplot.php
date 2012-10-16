<?PHP

define ("RESULTS_DIR", "/data/bpsr/results");
define ("SCRIPTS_DIR", "/home/dada/linux_64/bin");

putenv("PATH=/bin:/usr/bin:".SCRIPTS_DIR);

$snr_cut = isset($_GET["snr_cut"]) ? $_GET["snr_cut"] : 6;
$filter_cut = isset($_GET["filter_cut"]) ? $_GET["filter_cut"] : 99;
$dm_cut = isset($_GET["dm_cut"]) ? $_GET["dm_cut"] : 1.5;
$beam_mask = isset($_GET["beam_mask"]) ? $_GET["beam_mask"] : 8191;
$utc_start = isset($_GET["utc_start"]) ? $_GET["utc_start"] : "unknown";
$trans_cmd = "/home/dada/linux_64/bin/trans_all.sh $snr_cut $beam_mask";

$output = array();
$return_val = 0;

if ($utc_start == "unknown")
{
  $cmd = "/usr/bin/find /data/bpsr/results -mindepth 2 -maxdepth 2 -type f -name 'all_candidates.dat' | /usr/bin/sort | /usr/bin/tail -n 1 | /usr/bin/awk -F/ '{print $(NF-1)}'";
  $utc_start = exec($cmd, $output, $return_val);
}

/*
# Test the execution to see if it will work
$full_cmd = "cd /data/bpsr/results/".$utc_start."; ".$trans_cmd." > /dev/null";
$output = array();
$lastline = exec($full_cmd, $output, $return_val);

# If an error ocurred, generate a PHP image containing the error text
if (($lastline != "") || (count($output) > 0))
{
  print_r($output);
  echo $lastlist;
  exit();
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
} 
else 
{
*/
  $cmd = SCRIPTS_DIR."/trans_gen_overview.py -cands_file ".RESULTS_DIR."/".$utc_start."/all_candidates.dat -snr_cut ".$snr_cut." -beam_mask ".$beam_mask." -filter_cut ".$filter_cut." -dm_cut ".$dm_cut." -std_out";

  # now generate out plot
  header("Cache-Control: no-cache, must-revalidate"); // HTTP/1.1
  header("Expires: Mon, 26 Jul 1997 05:00:00 GMT");   // Date in the past
  header('Content-Type: image/png');
  header('Content-Disposition: inline; filename="image.png"');
  passthru($cmd);
