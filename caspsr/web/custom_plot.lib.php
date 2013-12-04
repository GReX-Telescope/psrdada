<?PHP

include_once("caspsr_webpage.lib.php");
include_once("definitions_i.php");
include_once("functions_i.php");
include_once($instrument.".lib.php");

class custom_plot extends caspsr_webpage 
{

  var $cfg = array();
  var $small = "240x180";
  var $large = "1024x768";
  var $common_options = "-jpC";
  var $pvfl_options = "-p flux -jTF";
  var $pvt_options = "-p time -jFD";
  var $pvfr_options = "-p freq -jTD";
  var $web_style = "";
  var $prev_obs = "";
  var $next_obs = "";
  var $results = array();
  var $dir = "";

  function custom_plot()
  {
    caspsr_webpage::caspsr_webpage();
    $inst = new caspsr();
    $this->web_style = $inst->config["SCRIPTS_DIR"]."/web_style.txt";
  }

  function printJavaScriptHead($get)
  {

    $this->dir = $get["basedir"]."/".$get["utc_start"];

    # check that the summed archives in the results dir exist
    if (($get["basedir"] == "") || (!file_exists($get["basedir"]))) {
      $esults = array(); 
    } elseif (($get["utc_start"] == "") || (!file_exists($this->dir))) {
      $this->results = array();
    } else {
      $this->results = caspsr::getObsSources($this->dir);

      # get the adjacaent observations
      $cmd = "ls -1 ".$get["basedir"]." | sort | grep ".$get["utc_start"]." -B 1 -C 1";
      $array = array();
      $rval = 0;
      $lastline = exec($cmd, $array, $rval);

      for ($i=0; $i<count($array); $i++) {
        if ($array[$i] == $get["utc_start"]) {
          # do nothing
        } elseif ($i == 0) {
          $this->prev_obs = $array[$i];
        } else {
          $this->next_obs = $array[$i];
        }
      }
    }
?>
    <script type="text/javascript">

    function regen_all() {
      regen("pvfl");
      regen("pvt");
      regen("pvfr");
    }

    function regen(type) {
  
      var common_opts = document.getElementById("common_options").value;
      var specific_opts = document.getElementById(type).value;
      var ar_base = "<?echo $this->dir?>/";
      var ar_ext = (type == "pvfr") ? "_f.tot" : "_t.tot";
      var plot_url = "pgwebplot.php?cmd=psrplot "+specific_opts+" "+common_opts+" "+ar_base;
  
<?  
      $srcs = array_keys($this->results);
      for ($i=0; $i<count($srcs); $i++) {
        $src = str_replace("+", "%2B",$srcs[$i]);
        echo "      var img_".$i."  = document.getElementById('img_".$i."_'+type);\n";
        echo "      var a_".$i."    = document.getElementById('a_".$i."_'+type);\n";
        echo "      var span_".$i." = document.getElementById('span_".$i."_'+type);\n";
        echo "      img_".$i.".src  = plot_url+'".$src."'+ar_ext+' -g ".$this->small." -s ".$this->web_style."';\n";
        echo "      a_".$i.".href   = plot_url+'".$src."'+ar_ext+' -g ".$this->large."';\n";
      }
?>
    }

  </script>
<?
  }

  function printSideBarHTML($get) 
  {
     $this->openBlockHeader("Search Parameters");
?>
<?
     $this->closeBlockHeader();
  }

  /*************************************************************************************************** 
   *
   * HTML for this page 
   *
   ***************************************************************************************************/
  function printHTML($get) 
  {

?>
<html>
<head>
  <title>CASPSR | Custom Plot | <?echo $get["utc_start"]?></title>
<?
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";

    $this->printJavaScriptHead($get);
?>
</head>

<body>
<?
    $this->printJavaScriptBody();
?>
  <div class='PageBackgroundSimpleGradient'>
  </div>
  <div class='Main'>
      <div style='text-align: left; vertical-align: middle; padding-left: 10px'>
        <img src="/caspsr/images/caspsr_logo_200x60.png" width=200 height=60>
      </div>
      <div class="content">
<?
        $this->printMainHTML($get);
?>
      </div><!-- conent -->
  </div><!-- Main -->
</body>
</html>
<?
  }

  function printMainHTML($get)
  {

    $this->openBlockHeader("Custom Plots");
?>

  <table cellspacing=0 cellpadding=5>
  <tr>
    <th></th>
<?
  echo "<td colspan=3 align=center>";
  if ($this->prev_obs != "")
    echo "<a href='custom_plot.lib.php?basedir=".$get["basedir"]."&utc_start=".$this->prev_obs."'>Prev Obs</a>&nbsp;&nbsp;<<&nbsp;&nbsp;";
  else
    echo "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;";

  echo "<b>".$get["utc_start"]."</b>\n";

  if ($this->next_obs != "")
    echo "&nbsp;&nbsp;>>&nbsp;&nbsp;<a href='custom_plot.lib.php?basedir=".$get["basedir"]."&utc_start=".$this->next_obs."'>Next Obs</a>";
  else
    echo "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;";

  echo "</td>\n";
?>
  </tr>

<? 
  if (count($this->results) == 0) { 
    echo "<tr><td colspan=4 align=center><font color=red size=+1>No Data Found</font></td></tr>\n";
  } else {
 ?>
  <tr>
    <td></td>
    <td colspan=3 style='text-align: center'>
      <input type=text id="common_options" size="32" value="<?echo $this->common_options?>">&nbsp;&nbsp;
      <input type="button" value="Regen All" onClick="regen_all()">
    </td>
  </tr>
  <tr>
    <td></td>
    <td style='text-align: center'>
      <input type=text size="16" id="pvfl" value="<?echo $this->pvfl_options?>">&nbsp;&nbsp;
      <input type="button" onClick="regen('pvfl')" value="Regen">
    </td>
    <td style='text-align: center'>
      <input type=text size="16" id="pvt" value="<?echo $this->pvt_options?>">&nbsp;&nbsp;
      <input type="button" onClick="regen('pvt')" value="Regen">
    </td>
    <td style='text-align: center'>
      <input type=text size="16" id="pvfr" value="<?echo $this->pvfr_options?>">&nbsp;&nbsp;
      <input type="button" onClick="regen('pvfr')" value="Regen">
    </td>
  </tr>
<?
  $psrs = array_keys($this->results);

  for ($i=0; $i<count($psrs); $i++) {
    $p = $psrs[$i];
    $p = str_replace("+", "%2B",$psrs[$i]);
    echo "<tr>\n";

    echo "  <td><input type='hidden' id='source' value='".$p."'>\n";
    echo "    <table>\n";
    echo "      <tr><td colspan=2><b>".$this->results[$psrs[$i]]["src"]."</b></td></tr>\n";
    echo "      <tr><td>INT</td><td>".$this->results[$psrs[$i]]["int"]."</td></tr>\n";
    echo "      <tr><td>DM</td><td>".sprintf("%5.4f",$this->results[$psrs[$i]]["dm"])."</td></tr>\n";
    echo "      <tr><td>P0</td><td>".sprintf("%5.4f",$this->results[$psrs[$i]]["p0"])."</td></tr>\n";
    echo "      <tr><td>Nsub</td><td>".$this->results[$psrs[$i]]["nsubint"]."</td></tr>\n";
    echo "      <tr><td>SNR</td><td>".sprintf("%5.4f",$this->results[$psrs[$i]]["snr"])."</td></tr>\n";
    echo "    </table>\n";
    echo "  </td>\n";
    echo "  <td>\n";
    echo "    <a id='a_".$i."_pvfl' href='pgwebplot.php?cmd=psrplot ".$this->pvfl_options." ".$this->common_options." -g $this->large ".$this->dir."/".$p."_t.tot -D -/PNG''>\n";
    echo "    <img id='img_".$i."_pvfl' src='pgwebplot.php?cmd=psrplot ".$this->pvfl_options." ".$this->common_options." -g $this->small ".$this->dir."/".$p."_t.tot -s ".$this->web_style."'></a><br>\n";
    echo "  </td>\n";

    echo "  <td>\n";
    echo "    <a id='a_".$i."_pvt' href='pgwebplot.php?cmd=psrplot ".$this->pvt_options." ".$this->common_options." -g $this->large ".$this->dir."/".$p."_t.tot -D -/PNG'>\n";
    echo "    <img id='img_".$i."_pvt' src='pgwebplot.php?cmd=psrplot ".$this->pvt_options." ".$this->common_options." -g $this->small  ".$this->dir."/".$p."_t.tot -s ".$this->web_style."'></a><br>\n";
    echo "  </td>\n";

    echo "  <td>\n";
    echo "    <a id='a_".$i."_pvfr' href='pgwebplot.php?cmd=psrplot ".$this->pvfr_options." ".$this->common_options." -g $this->large ".$this->dir."/".$p."_f.tot -D -/PNG'>\n";   
    echo "    <img id='img_".$i."_pvfr' src='pgwebplot.php?cmd=psrplot ".$this->pvfr_options." ".$this->common_options." -g $this->small ".$this->dir."/".$p."_f.tot -s ".$this->web_style."'></a><br>\n";
    echo "  </td>\n";
    echo "</tr>\n";
  }
  echo "</table>\n";
  }
    $this->closeBlockHeader();
  }

  function handleRequest()
  {
    $this->printHTML($_GET);
  }

}
$obj = new custom_plot();
$obj->handleRequest($_GET);
