<?PHP

include("functions_i.php");
include("definitions_i.php");

$config = getConfigFile(SYS_CONFIG);
$machine = $_GET["machine"];
$machines = array();


$gang_network = "http://".$_SERVER["HTTP_HOST"]."/ganglia/graph.php?g=network_report&z=medium&c=APSR%20Clients&m=&r=hour&s=descending&hc=4";
$gang_load = "http://".$_SERVER["HTTP_HOST"]."/ganglia/graph.php?g=load_report&z=medium&c=APSR%20Clients&m=&r=hour&s=descending&hc=4";

/* If nexus, then we are running this for all nodes */
if ($machine != "nexus") {

  $single_machine = "&single_machine=".$machine;
  $machines = array($machine);

} else {

  $single_machine = "";
  for ($i=0; $i<$config["NUM_PWC"]; $i++) {
    array_push($machines,$config["PWC_".$i]);
  }  

}

// Don't allow this page to be cached, since it should always be fresh.
header("Cache-Control: no-cache, must-revalidate"); // HTTP/1.1
header("Expires: Mon, 26 Jul 1997 05:00:00 GMT"); // Date in the past
?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <link rel="STYLESHEET" type="text/css" href="./style_log.css">
  <!-- jsProgressBarHandler prerequisites : prototype.js -->
  <script type="text/javascript" src="js/prototype.js"></script>
  <!-- jsProgressBarHandler core -->
  <script type="text/javascript" src="js/jsProgressBarHandler.js"></script>

  <script type="text/javascript">

  var url = "http://<?echo $_SERVER["HTTP_HOST"]?>/control/client_command.php?cmd=get_status&raw=1<?echo $single_machine?>"

  function looper() {
    request()
    setTimeout('looper()',2000)
  }

  function request() {

    if (window.XMLHttpRequest)
      http_request = new XMLHttpRequest();
    else
      http_request = new ActiveXObject("Microsoft.XMLHTTP");

    http_request.onreadystatechange = function() {
      handle_data(http_request)
    };
    http_request.open("GET", url, true);
    http_request.send(null);

  }
                                                                                                                                    
  function handle_data(http_request) {
    if (http_request.readyState == 4) {
      var response = String(http_request.responseText)
      var lines = response.split("\n");
      var i = 0;
      var now = new Date();

      for (i=0; i<lines.length; i++) {
      
        if (lines[i].length > 0) {
     
          var values = lines[i].split(":");
    
          var host = values[0];
          var result = values[1];

          if (result == "ok") {
            var statuses= values[2].split(";;;");
            
            var disks = statuses[0].split(" ");
            var dbs   = statuses[1].split(" ");
            var loads = statuses[2].split(",");
            var unproc_gb = parseFloat(statuses[3])/1024;

            var disk_percent = Math.floor(parseInt(disks[1]) / parseInt(disks[0])*100); 
            var disk_gb = (parseInt(disks[0]) - parseInt(disks[1])) / 1024.0; 

            var db_percent = Math.floor(parseInt(dbs[1]) / parseInt(dbs[0])*100);
            var load_percent = Math.floor((parseFloat(loads[0])/8)*100);

            var mb_per_block = <?echo $config[$config["PROCESSING_DATA_BLOCK"]."_BLOCK_BUFSZ"]?> / (1024*1024*1024);
            var gb_unprocessed = (mb_per_block*parseFloat(dbs[1]) + unproc_gb);
<?
for ($i=0; $i<count($machines); $i++) {
  echo "          if (host == \"".$machines[$i]."\") {\n";
  echo "            ".$machines[$i]."_db.setPercentage(db_percent);\n";
  echo "            document.getElementById(\"".$machines[$i]."_db_value\").innerHTML = \"&nbsp;\"+dbs[1]+\"&nbsp;of&nbsp;\"+dbs[0]\n";
  echo "            ".$machines[$i]."_disk.setPercentage(disk_percent);\n";
  echo "            document.getElementById(\"".$machines[$i]."_disk_unproc\").innerHTML = \"&nbsp;\"+unproc_gb.toFixed(1)\n";
  echo "            document.getElementById(\"".$machines[$i]."_disk_left\").innerHTML = \"&nbsp;\"+disk_gb.toFixed(1)\n";
  echo "            ".$machines[$i]."_load.setPercentage(load_percent);\n";
  echo "            document.getElementById(\"".$machines[$i]."_load_value\").innerHTML = \"&nbsp;\"+loads[0]\n";
  echo "            document.getElementById(\"".$machines[$i]."_gb_unproc\").innerHTML = \"&nbsp;\"+gb_unprocessed.toFixed(1)\n";
  echo "          }\n";
}
?>
          }
        }
      }
      var theTime = now.getTime();
      document.getElementById("load").src = "<?echo $gang_load?>?"+theTime;
      document.getElementById("network").src = "<?echo $gang_network?>?"+theTime;
    }
  }
  </script>

</head>
<body onload="looper()">

  <!-- Progress Bars for All 3 data blocks -->
  <script type="text/javascript">
    Event.observe(window, 'load', function() {
<?

for ($i=0; $i<count($machines); $i++) {

  echo $machines[$i]."_db = new JS_BRAMUS.jsProgressBar($('".$machines[$i]."_db_progress_bar'), 0, ";
  echo " { width : 80, showText : false, barImage : Array( 'images/jsprogress/percentImage_back1_80.png', 'images/jsprogress/percentImage_back2_80.png', 'images/jsprogress/percentImage_back3_80.png', 'images/jsprogress/percentImage_back4_80.png') } );\n";

  echo $machines[$i]."_disk = new JS_BRAMUS.jsProgressBar($('".$machines[$i]."_disk_progress_bar'), 0, ";
  echo " { width : 80, showText : false, barImage : Array( 'images/jsprogress/percentImage_back1_80.png', 'images/jsprogress/percentImage_back2_80.png', 'images/jsprogress/percentImage_back3_80.png', 'images/jsprogress/percentImage_back4_80.png') } );\n";

  echo $machines[$i]."_load = new JS_BRAMUS.jsProgressBar($('".$machines[$i]."_load_progress_bar'), 0, ";
  echo " { width : 80, showText : false, barImage  : Array( 'images/jsprogress/percentImage_back1_80.png', 'images/jsprogress/percentImage_back2_80.png', 'images/jsprogress/percentImage_back3_80.png', 'images/jsprogress/percentImage_back4_80.png') } );\n";

}

?>
  }, false);
  </script>


<table cellpadding=0 border=0 width=100%>
<tr>
  <td></td>
  <td colspan=2> <h3>Machine Load</h3> </td>
  <td colspan=2> <h3>Data Block</h3> </td>
  <td> <h3>Disk [GB]</h3> </td><td width=40px>Used</td><td width=60px>Free</td><td>Total.</td>
</tr>

<?
for ($i=0; $i<count($machines); $i++) {
?>
<tr>
  <td align="right"><?echo $machines[$i].": "?></td>

  <td width="80px">
  <? echo "<span id=\"".$machines[$i]."_load_progress_bar\">[  Loading Progress Bar ]</span></td>\n"; ?>
  </td>

  <td>
  <? echo "<span id=\"".$machines[$i]."_load_value\"></span>\n"; ?>

  <td width="80px">
  <? echo "<span id=\"".$machines[$i]."_db_progress_bar\">[  Loading Progress Bar ]</span></td>\n"; ?>
  </td>

  <td>
   <? echo "<span id=\"".$machines[$i]."_db_value\"></span>\n"; ?>
  </td>

  <td width="80px">
  <?  echo "<span id=\"".$machines[$i]."_disk_progress_bar\">[  Loading Progress Bar ]</span>\n";?>
  </td>

  <td>
   <? echo "<span id=\"".$machines[$i]."_disk_unproc\"></span>\n"?>
  </td>

  <td>
  <?echo "<span id=\"".$machines[$i]."_disk_left\"></span>\n";?>
  </td>

  <td>
  <?echo "<span id=\"".$machines[$i]."_gb_unproc\"></span>\n";?>
  </td>

</tr>
<tr height="100%"><td></td></tr>
<? } ?>
</table>

<center>
<img id="load" src="<?echo $gang_load?>">
<br>
<img id="network" src="<?echo $gang_network?>">
</center>

</body>
</table>
</html>
  
