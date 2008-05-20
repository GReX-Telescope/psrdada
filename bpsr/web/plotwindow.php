<?PHP


include("definitions_i.php");
include("functions_i.php");

$config = getConfigFile(SYS_CONFIG);
$conf = getConfigFile(DADA_CONFIG,TRUE);
$spec = getConfigFile(DADA_SPECIFICATION, TRUE);

?>
<html>
<head>
  <? echo STYLESHEET_HTML; ?>
  <? echo FAVICO_HTML?>

  <script type="text/javascript">


    /* Creates a pop up window */
    function popWindow(URL,width,height) {
                                                                                                                                      
      var width = width || "1024";
      var height = height || "768";

      URL = URL + "&obsid=" + document.getElementById("utc_start").value;

      day = new Date();
      id = day.getTime();
      eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,scrollbars=1,location=1,statusbar=1,menubar=1,resizable=1,width="+width+",height="+height+"');");
    }

    /* Looping function to try and refresh the images */
    function looper() {
      request()
      setTimeout('looper()',5000)
    }

    /* Gets the data from the URL */
    function request() {
      if (window.XMLHttpRequest)
        http_request = new XMLHttpRequest()
      else
        http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
      http_request.onreadystatechange = function() {
        handle_data(http_request)
      }

      var type = "bandpass";

      if (document.imageform.imagetype[0].checked == true) {
        type = "bandpass";
      }

      if (document.imageform.imagetype[1].checked == true) {
        type = "dm0timeseries";
      }

      /* This URL will return the names of the 5 current */
      var url = "bpsr/plotupdate.php?results_dir=<?echo $config["SERVER_RESULTS_DIR"]?>&type="+type;

      http_request.open("GET", url, true)
      http_request.send(null)
    }

    /* Parses the HTTP response and makes changes to images
     * as requried */
    function handle_data(http_request) {
      if (http_request.readyState == 4) {
        var response = String(http_request.responseText)
        var lines = response.split(";;;")

<?
        for ($i=0; $i<$config["NUM_PWC"]; $i++) {
          echo "        var img".$i."_line = lines[".$i."].split(\":::\")\n";
        }

        for ($i=0; $i<$config["NUM_PWC"]; $i++) {
          echo "        var img".$i." = img".$i."_line[1]\n";
        }

        for ($i=0; $i<$config["NUM_PWC"]; $i++) {

          echo "        if (document.getElementById(\"beam".($i+1)."\").src != img".$i.") {\n";
          echo "          document.getElementById(\"beam".($i+1)."\").src = img".$i."\n";
          echo "        }\n";
        }
        
        echo "       var utc_start = lines[".$config["NUM_PWC"]."]\n";
?>
        document.getElementById("utc_start").value = utc_start;
      }
    }

  </script>

</head>
<body onload="looper()">
<center>
<table width=100%>
<input id="utc_start" type="hidden" value=""></input>

<tr>

<td>Show:</td>
    <td>
      <form name="imageform">
      <input type="radio" name="imagetype" id="imagetype" value="bandpass" checked onClick="request()">Bandpass<br>
      <input type="radio" name="imagetype" id="imagetype" value="dm0timeseries" onClick="request()">Time Series<br>
      </form>
    </td>
  <td width=20>&nbsp;&nbsp;</td>
<td valign=bottom>Beam: </td><td>
<div class="btns">
<?
for ($i=0; $i<$config["NUM_PWC"]; $i++) {
?>
  <a href="javascript:popWindow('bpsr/beamwindow.php?beamid=<?echo ($i+1)?>', 1024, 800)" class="btn" > <span><?echo ($i+1)?></span> </a>
<?
}
?>
</div>
</td></tr></table>
</center>

<center>
<table border=0 cellspacing=5 cellpadding=5>

  <tr height=42>
    <?echoBlank()?>
    <?echoBeam(13)?>
    <?echoBlank()?>
    <?echoBeam(12)?>
    <?echoBlank()?> 
  </tr>
  <tr height=42>
    <?echoBlank()?>
    <?echoBeam(6)?>
    <?echoBlank()?>
  </tr>
  <tr height=42>
    <?echoBlank()?>
    <?echoBeam(7)?>
    <?echoBeam(5)?>
    <?echoBlank()?> 
  </tr>

  <tr height=42>
    <?echoBeam(8)?>
    <?echoBeam(1)?>
    <?echoBeam(11)?>
  </tr>

  <tr height=42>
    <?echoBeam(2)?>
    <?echoBeam(4)?>
  </tr>

  <tr height=42>
    <?echoBlank()?>
    <?echoBeam(3)?>
    <?echoBlank()?>
  </tr>

  <tr height=42>
    <?echoBlank()?>
    <?echoBeam(9)?>
    <?echoBeam(10)?>
    <?echoBlank()?>
  </tr>
  
  <tr height=42>
    <?echoBlank()?>
    <?echoBlank()?>
    <?echoBlank()?>
  </tr>
</table>
</center>



</body>
</html>

<?

function echoBlank() {

  echo "<td ></td>\n";
}

function echoBeam($beam_no) {

  echo "<td rowspan=2>";
  echo "<a href=\"javascript:popWindow('bpsr/beamwindow.php?beamid=".$beam_no."')\">";
  echo "<img src=\"/images/blankimage.gif\" width=112 height=84 id=\"beam".$beam_no."\" TITLE=\"Beam ".$beam_no."\" alt=\"Beam ".$beam_no."\">\n";
  echo "</a></td>\n";

}

