
<?PHP

include("state_banner.lib.php");
include("beam_config.lib.php");

$state_banner = new state_banner();
$beam_config = new beam_config();

// get the required style sheets
$css = array();
$css = array_merge($css, $state_banner->css);
$css = array_merge($css, $beam_config->css);

// get the required external javascript 
$ejs = array();
$ejs = array_merge($ejs, $state_banner->ejs);
$ejs = array_merge($ejs, $beam_config->ejs);

$css = array_unique($css);
$ejs = array_unique($ejs);

echo $state_banner->doc_type."\n";
?>
<html>
<head>
  <title>BPSR | Beam Configuration</title>
  <link rel='shortcut icon' href='/bpsr/images/bpsr_favico.ico'/>

<?
  for ($i=0; $i<count($css); $i++)
    echo "   <link rel='stylesheet' type='text/css' href='".$css[$i]."'>\n";
  for ($i=0; $i<count($ejs); $i++)
    echo "   <script type='text/javascript' src='".$ejs[$i]."'></script>\n";
?>

  <script type="text/javascript">

    function poll_server() 
    {
      <? echo $state_banner->javaScriptCallback()."\n";?>
      <? echo $beam_config->javaScriptCallback()."\n";?>
      setTimeout('poll_server()', 4000);
    }

  </script>
<?
  $state_banner->printJavaScriptHead();
  $beam_config->printJavaScriptHead();
?> 
</head>

<body onload="poll_server()">
<?
  $state_banner->printJavaScriptBody();
  $beam_config->printJavaScriptBody();

  echo "<table border=0 cellspacing='0px' cellpadding='5px' width='100%'>\n";
  echo "  <tr>\n";
  echo "    <td colspan=2>\n";
  $state_banner->printHTML();
  echo "    </td>\n";
  echo "  </tr>\n";

  echo "  <tr>\n";
  echo "    <td style='vertical-align: top;' width='630px'>\n";

  $beam_config->printHTML();

  echo "    </td>\n";
  echo "  </tr>\n";
  echo "</table>\n";

  ?>
</body>
</html>

