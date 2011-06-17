<?PHP

include("current_observation.lib.php");
include("machine_summary.lib.php");
include("state_update.lib.php");
include("log_list.lib.php");
include("plot_window.lib.php");
include("archival_summary.lib.php");

$state_update = new state_update();
$current_observation = new current_observation();
$machine_summary = new machine_summary();
$log_list = new log_list();
$plot_window = new plot_window();
$archival_summary = new archival_summary();

// get the required style sheets
$css = array();
$css = array_merge($css, $state_update->css);
$css = array_merge($css, $current_observation->css);
$css = array_merge($css, $machine_summary->css);
$css = array_merge($css, $log_list->css);
$css = array_merge($css, $plot_window->css);
$css = array_merge($css, $archival_summary->css);

// get the required external javascript 
$ejs = array();
$ejs = array_merge($ejs, $state_update->ejs);
$ejs = array_merge($ejs, $current_observation->ejs);
$ejs = array_merge($ejs, $machine_summary->ejs);
$ejs = array_merge($ejs, $log_list->ejs);
$ejs = array_merge($ejs, $plot_window->ejs);
$ejs = array_merge($ejs, $archival_summary->ejs);

$css = array_unique($css);
$ejs = array_unique($ejs);

?>
<html>
<head>
  <title>CASPSR</title>
  <link rel='shortcut icon' href='/caspsr/images/caspsr_favicon.ico'/>
<?
  for ($i=0; $i<count($css); $i++)
    echo "   <link rel='stylesheet' type='text/css' href='".$css[$i]."'>\n";
  for ($i=0; $i<count($ejs); $i++)
    echo "   <script type='text/javascript' src='".$ejs[$i]."'></script>\n";
?>

  <script type="text/javascript">

    function poll_server() 
    {
      <? echo $plot_window->javaScriptCallback()."\n"; ?>
      <? echo $state_update->javaScriptCallback()."\n"; ?>
      <? echo $current_observation->javaScriptCallback()."\n";?>
      <? echo $archival_summary->javaScriptCallback()."\n"; ?>
      <? echo $machine_summary->javaScriptCallback()."\n"; ?>
      <? echo $log_list->javaScriptCallback()."\n"; ?>
      setTimeout('poll_server()', 4000);
    }

  </script>
<?
  $plot_window->printJavaScriptHead();
  $state_update->printJavaScriptHead();
  $current_observation->printJavaScriptHead();
  $archival_summary->printJavaScriptHead();
  $machine_summary->printJavaScriptHead();
  $log_list->printJavaScriptHead();
?> 
</head>

<body onload="poll_server()">
<?
  $plot_window->printJavaScriptBody();
  $state_update->printJavaScriptBody();
  $current_observation->printJavaScriptBody();
  $archival_summary->printJavaScriptBody();
  $machine_summary->printJavaScriptBody();
  $log_list->printJavaScriptBody();
?>
  <div class="PageBackgroundSimpleGradient">
  </div>
  <div class="Main">
    <div class="contentLayout">
      <div class="sidebar1">
        <div style='text-align: center; vertical-align: middle;'>
          <img src="/caspsr/images/caspsr_logo_200x60.png" width=200 height=60>
        </div>
<?
  $plot_window->printHTML();      
?>
      </div>
    </div>
    <div class="content">
<?
  $state_update->printHTML();
?> 
      <div style='position: relative; overflow-x: hidden; overflow-y: hidden; margin: 0px; margin-right: 10px; min-height: 15px; min-width:15px; height: 100%;'>
        <div style='position: relative; padding: 0; margin: 0; overflow-x: hidden; overflow-y: hidden; float: right; width: 200px; height: 100%;'>
<?
  $archival_summary->printHTML(" style='margin: 0px; margin-right: 10px;'");
?>
        </div>
        <div style='position: relative; padding:0 margin: 0; overflow-x: hidden; overflow-y: hidden'>
<?
  $current_observation->printHTML(" style='margin: 0; margin-right: 10px;'");
?>
        </div>
      </div>
<?
  $machine_summary->printHTML();
  $log_list->printHTML();
?>
    </div>
  </div>
</body>
</html>
