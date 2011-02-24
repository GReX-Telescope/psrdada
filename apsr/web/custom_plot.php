<?PHP
include("definitions_i.php");
include("functions_i.php");
include("apsr_functions_i.php");

$config = getConfigFile(SYS_CONFIG);
$utc_start = $_GET["utc_start"];
$basedir = $_GET["basedir"];

$dir = $basedir."/".$utc_start;

# check that the summed archives in the results dir exist
if (($basedir == "") || (!file_exists($basedir))) {
  $results = array();
} elseif (($utc_start == "") || (!file_exists($dir))) {
  $results = array();
} else {
  $results = getObsSources($dir);

  $small = "240x180";
  $large = "800x600";

  $common_options = "-jpC";
  $pvfl_options = "-p flux -jTF";
  $pvt_options = "-p time -jFD";
  $pvfr_options = "-p freq -jTD";

  # get the adjacaent observations
  $cmd = "ls -1 ".$basedir." | sort | grep ".$utc_start." -B 1 -C 1";
  $array = array();
  $rval = 0;
  $lastline = exec($cmd, $array, $rval);
  $prev_obs = "";
  $next_obs = "";

  for ($i=0; $i<count($array); $i++) {
    if ($array[$i] == $utc_start) {

    } elseif ($i == 0) {
      $prev_obs = $array[$i];
    } else {
      $next_obs = $array[$i];
    }
  }

}


?>
<html>
<head>
  <? echo STYLESHEET_HTML; ?>
  <? echo FAVICO_HTML?>
  <script type="text/javascript">

    function regen_all() {
      regen("pvfl");
      regen("pvt");
      regen("pvfr");
    }

    function regen(type) {

      var common_opts = document.getElementById("common_options").value;
      var specific_opts = document.getElementById(type).value;
      var ar_base = "<?echo $dir?>/";
      var ar_ext = (type == "pvfr") ? "_f.ar" : "_t.ar";
      var plot_url = "pgwebplot.php?cmd=psrplot "+specific_opts+" "+common_opts+" "+ar_base;

<?
  $srcs = array_keys($results);
for ($i=0; $i<count($srcs); $i++) {
  $src = str_replace("+", "%2B",$srcs[$i]);
  echo "      var img_".$i."  = document.getElementById('img_".$i."_'+type);\n";
  echo "      var a_".$i."    = document.getElementById('a_".$i."_'+type);\n";
  echo "      var span_".$i." = document.getElementById('span_".$i."_'+type);\n";
  echo "      img_".$i.".src  = plot_url+'".$src."'+ar_ext+' -g $small -s ".$config["SCRIPTS_DIR"]."/web_style.txt';\n";
  echo "      a_".$i.".href   = plot_url+'".$src."'+ar_ext+' -g $large';\n";
      }
?>
    }

  </script>
</head>

<body>

  <table border=0 width="100%" cellspacing=0 cellpadding=5>
  <tr>
    <th></th>
<?
  echo "<td colspan=3 align=center>";
  if ($prev_obs != "") 
    echo "<a href='custom_plot.php?basedir=".$basedir."&utc_start=".$prev_obs."'>Older Obs</a>&nbsp;&nbsp;<<&nbsp;&nbsp;";
  else
    echo "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;";
  
  echo "<b>".$utc_start."</b>\n";

  if ($next_obs != "") 
    echo "&nbsp;&nbsp;>>&nbsp;&nbsp;<a href='custom_plot.php?basedir=".$basedir."&utc_start=".$next_obs."'>Next Obs</a>";
  else
    echo "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;";
  
  echo "</td>\n";
?>
  </tr>

<? if (count($results) == 0) { ?>
  <tr><td colspan=4 align=center><font color=red size=+1>No Data Found</font></td></tr>
<? } else { ?>

  <tr>
    <td></td>
    <td colspan=3>
      <input type=text id="common_options" size="48" value="<?echo $common_options?>">&nbsp;&nbsp;
      <input type="button" value="Regen All" onClick="regen_all()">
    </td>
  </tr>
  <tr>
    <td></td>
    <td>
      <input type=text size="16" id="pvfl" value="<?echo $pvfl_options?>">&nbsp;&nbsp;
      <input type="button" onClick="regen('pvfl')" value="Regen">
    </td>
    <td>
      <input type=text size="16" id="pvt" value="<?echo $pvt_options?>">&nbsp;&nbsp;
      <input type="button" onClick="regen('pvt')" value="Regen">
    </td>
    <td>
      <input type=text size="16" id="pvfr" value="<?echo $pvfr_options?>">&nbsp;&nbsp;
      <input type="button" onClick="regen('pvfr')" value="Regen">
    </td>
  </tr>
<?
  $psrs = array_keys($results);

  for ($i=0; $i<count($psrs); $i++) {
    $p = $psrs[$i];
    $p = str_replace("+", "%2B",$psrs[$i]);
    echo "<tr>\n";

    echo "  <td width=320><input type='hidden' id='source' value='".$p."'>\n";
    echo "    <tt>\n";
    echo "    ".$results[$psrs[$i]]["src"]."<BR>\n";
    echo "    INT: ".$results[$psrs[$i]]["int"]."<BR>\n";
    echo "    DM: ".$results[$psrs[$i]]["dm"]."<BR>\n";
    echo "    PO: ".$results[$psrs[$i]]["p0"]."<BR>\n";
    echo "    N SUBINT: ".$results[$psrs[$i]]["nsubint"]."<BR>\n";
    echo "    SNR: ".$results[$psrs[$i]]["snr"]."<BR>\n";
    echo "    </tt>\n";
    echo "  </td>\n";
    echo "  <td>\n";
    echo "    <a id='a_".$i."_pvfl' href='pgwebplot.php?cmd=psrplot ".$pvfl_options." ".$common_options." -g $large ".$dir."/".$p."_t.ar -D -/PNG''>\n";
    echo "    <img id='img_".$i."_pvfl' src='pgwebplot.php?cmd=psrplot ".$pvfl_options." ".$common_options." -g $small ".$dir."/".$p."_t.ar -s ".$config["SCRIPTS_DIR"]."/web_style.txt'></a><br>\n";
    echo "  </td>\n";

    echo "  <td>\n";
    echo "    <a id='a_".$i."_pvt' href='pgwebplot.php?cmd=psrplot ".$pvt_options." ".$common_options." -g $large ".$dir."/".$p."_t.ar -D -/PNG'>";
    echo "    <img id='img_".$i."_pvt' src='pgwebplot.php?cmd=psrplot ".$pvt_options." ".$common_options." -g $small  ".$dir."/".$p."_t.ar -s ".$config["SCRIPTS_DIR"]."/web_style.txt'></a><br>\n";
    echo "  </td>\n";

    echo "  <td>\n";
    echo "    <a id='a_".$i."_pvfr' href='pgwebplot.php?cmd=psrplot ".$pvfr_options." ".$common_options." -g $large ".$dir."/".$p."_f.ar -D -/PNG'>";
    echo "    <img id='img_".$i."_pvfr' src='pgwebplot.php?cmd=psrplot ".$pvfr_options." ".$common_options." -g $small ".$dir."/".$p."_f.ar -s ".$config["SCRIPTS_DIR"]."/web_style.txt'></a><br>\n";

    echo "  </td>\n";
    echo "</tr>\n";
  }
}
?>
  </table>
</body>
</html>

