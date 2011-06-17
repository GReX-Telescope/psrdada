
<?PHP

include("state_banner.lib.php");
include("observation_summary.lib.php");
include("plot_window.lib.php");
include("gains.lib.php");
include("machine_summary.lib.php");
include("log_list.lib.php");

$state_banner = new state_banner();
$observation_summary = new observation_summary();
$plot_window = new plot_window();
$gains = new gains();
$machine_summary = new machine_summary();
$log_list = new log_list();

// get the required style sheets
$css = array();
$css = array_merge($css, $state_banner->css);
$css = array_merge($css, $observation_summary->css);
$css = array_merge($css, $plot_window->css);
$css = array_merge($css, $gains->css);
$css = array_merge($css, $machine_summary->css);
$css = array_merge($css, $log_list->css);

// get the required external javascript 
$ejs = array();
$ejs = array_merge($ejs, $state_banner->ejs);
$ejs = array_merge($ejs, $observation_summary->ejs);
$ejs = array_merge($ejs, $plot_window->ejs);
$ejs = array_merge($ejs, $gains->ejs);
$ejs = array_merge($ejs, $machine_summary->ejs);
$ejs = array_merge($ejs, $log_list->ejs);

$css = array_unique($css);
$ejs = array_unique($ejs);

$css = array_values($css);
$ejs = array_values($ejs);

?>
<html>
<head>
  <title>APSR</title>
  <link rel='shortcut icon' href='/apsr/images/apsr_favico.ico'/>
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
      <? echo $observation_summary->javaScriptCallback()."\n";?>
      <? echo $plot_window->javaScriptCallback()."\n"; ?>
      <? echo $gains->javaScriptCallback()."\n"; ?>
      <? echo $machine_summary->javaScriptCallback()."\n"; ?>
      <? echo $log_list->javaScriptCallback()."\n"; ?>
      setTimeout('poll_server()', 4000);
    }

  </script>
<?
  $state_banner->printJavaScriptHead();
  $observation_summary->printJavaScriptHead();
  $plot_window->printJavaScriptHead();
  $gains->printJavaScriptHead();
  $machine_summary->printJavaScriptHead();
  $log_list->printJavaScriptHead();
?> 
</head>

<body onload="poll_server()">
  <div class="background"></div>
  <div class="Main">
<?
  $state_banner->printJavaScriptBody();
  $observation_summary->printJavaScriptBody();
  $plot_window->printJavaScriptBody();
  $gains->printJavaScriptBody();
  $machine_summary->printJavaScriptBody();
  $log_list->printJavaScriptBody();

  echo "<table border=0 cellspacing='0px' cellpadding='5px' width='100%'>\n";
  echo "  <tr>\n";
  echo "    <td colspan=2>\n";
  $state_banner->printHTML();
  echo "    </td>\n";
  echo "  </tr>\n";

  echo "  <tr>\n";
  echo "    <td style='vertical-align: top;' width='530px'>\n";

  $observation_summary->openBlockHeader("Observation Summary");
    $observation_summary->printHTML();
  $observation_summary->closeBlockHeader();

  echo "<img src='/images/spacer.gif' width='1px' height='10px'>\n";

  $plot_window->openBlockHeader("");
    $plot_window->printHTML();      
  $plot_window->closeBlockHeader();

  echo "<img src='/images/spacer.gif' width='1px' height='10px'>\n";


  $gains->openBlockHeader("");
    $gains->printHTML();
  $gains->closeBlockHeader();

  echo "  </td>\n";
  echo "  <td style='vertical-align: top;'>\n";
    $machine_summary->printHTML();
    echo "<img src='/images/spacer.gif' width='1px' height='10px'>\n";
    $log_list->printHTML();
  echo "    </td>\n";
  echo "  </tr>\n";
  echo "</table>\n";

  ?>
</div>
</body>
</html>

