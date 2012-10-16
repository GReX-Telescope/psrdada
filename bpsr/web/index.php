<?PHP

ini_set('display_errors',1); 
error_reporting(E_ALL);

include("state_banner.lib.php");
include("current_obs_window.lib.php");
include("plot_window.lib.php");
include("archival_summary_window.lib.php");
include("transient_window.lib.php");
include("machine_summary.lib.php");
include("log_list.lib.php");

$state_banner = new state_banner();
$current_obs = new current_obs();
$plot_window = new plot_window();
$archival_summary = new archival_summary();
$transient_window = new transient_window();
$machine_summary = new machine_summary();
$log_list = new log_list();

// get the required style sheets
$css = array();
$css = array_merge($css, $state_banner->css);
$css = array_merge($css, $current_obs->css);
$css = array_merge($css, $plot_window->css);
$css = array_merge($css, $archival_summary->css);
$css = array_merge($css, $transient_window->css);
$css = array_merge($css, $machine_summary->css);
$css = array_merge($css, $log_list->css);

// get the required external javascript 
$ejs = array();
$ejs = array_merge($ejs, $state_banner->ejs);
$ejs = array_merge($ejs, $current_obs->ejs);
$ejs = array_merge($ejs, $plot_window->ejs);
$ejs = array_merge($ejs, $archival_summary->ejs);
$ejs = array_merge($ejs, $transient_window->ejs);
$ejs = array_merge($ejs, $machine_summary->ejs);
$ejs = array_merge($ejs, $log_list->ejs);

$css = array_unique($css);
$ejs = array_unique($ejs);

?>
<html>
<head>
  <title>BPSR</title>
  <link rel='shortcut icon' href='/bpsr/images/bpsr_favico.ico'/>
<?
  for ($i=0; $i<count($css); $i++)
    echo "  <link rel='stylesheet' type='text/css' href='".$css[$i]."'>\n";
  for ($i=0; $i<count($ejs); $i++)
    echo "  <script type='text/javascript' src='".$ejs[$i]."'></script>\n";
?>

  <script type="text/javascript">

    function poll_server() 
    {
      <? echo $state_banner->javaScriptCallback()."\n";?>
      <? echo $current_obs->javaScriptCallback()."\n";?>
      <? echo $plot_window->javaScriptCallback()."\n"; ?>
      <? echo $archival_summary->javaScriptCallback()."\n"; ?>
      <? echo $transient_window->javaScriptCallback()."\n"; ?>
      <? echo $machine_summary->javaScriptCallback()."\n"; ?>
      <? echo $log_list->javaScriptCallback()."\n"; ?>
      setTimeout('poll_server()', 4000);
    }

  </script>
<?
  $state_banner->printJavaScriptHead();
  $current_obs->printJavaScriptHead();
  $plot_window->printJavaScriptHead();
  $archival_summary->printJavaScriptHead();
  $transient_window->printJavaScriptHead();
  $machine_summary->printJavaScriptHead();
  $log_list->printJavaScriptHead();
?> 
</head>

<body onload="poll_server()">
<?
  $state_banner->printJavaScriptBody();
  $current_obs->printJavaScriptBody();
  $plot_window->printJavaScriptBody();
  $archival_summary->printJavaScriptBody();
  $transient_window->printJavaScriptBody();
  $machine_summary->printJavaScriptBody();
  $log_list->printJavaScriptBody();

  echo "<table border=0 cellspacing='0px' cellpadding='5px' width='100%'>\n";
  echo "  <tr>\n";
  echo "    <td colspan=2>\n";
  $state_banner->printHTML();
  echo "    </td>\n";
  echo "  </tr>\n";

  echo "  <tr>\n";
  echo "    <td style='vertical-align: top;' width='630px'>\n";

  $current_obs->openBlockHeader("Observation Summary");
    $current_obs->printHTML();
  $current_obs->closeBlockHeader();

  echo "<img src='/images/spacer.gif' width='1px' height='10px'>\n";

  $plot_window->openBlockHeader("Diagnostic Plots");
    $plot_window->printHTML();      
  $plot_window->closeBlockHeader();

  echo "<img src='/images/spacer.gif' width='1px' height='10px'>\n";

  $current_obs->openBlockHeader("Archival Summary");
    $archival_summary->printHTML();
  $current_obs->closeBlockHeader();

  $current_obs->openBlockHeader("Transient Summary");
    $transient_window->printHTML();
  $current_obs->closeBlockHeader();

  echo "  </td>\n";
  echo "  <td style='vertical-align: top;'>\n";
    $machine_summary->printHTML();
    echo "<img src='/images/spacer.gif' width='1px' height='10px'>\n";
    $log_list->printHTML();
  echo "    </td>\n";
  echo "  </tr>\n";
  echo "</table>\n";

  ?>
</body>
</html>

