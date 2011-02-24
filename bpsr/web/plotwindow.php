<?PHP

include ("bpsr.lib.php");

$inst = new bpsr();

$nbeam = 13;

?>
<html>
<?
  $inst->open_head();
  $inst->print_head_int("BPSR Plot Window", 0);
?>

  <script type="text/javascript">


    /* Creates a pop up window */
    function popWindow(URL,width,height) {

      var width = "1300";
      var height = "820";

      var utc_start = parent.infowindow.document.getElementById("UTC_START").innerHTML;

      URL = URL + "&obsid=" + utc_start;

      day = new Date();
      id = day.getTime();
      eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,scrollbars=1,location=1,statusbar=1,menubar=1,resizable=1,width="+width+",height="+height+"');");
    }

    /* Looping function to try and refresh the images */
    function looper() {
      request()
      setTimeout('looper()',5000)
    }

    /* Parses the HTTP response and makes changes to images
     * as requried */
    function handle_data(http_request) {

      if (http_request.readyState == 4) {
        var response = String(http_request.responseText)
        var lines = response.split("\n");
        var currImg
        var beam
        var size
        var type
        var img

        for (i=0; i < lines.length-1; i++) {

          values = lines[i].split(":::");
          beam = values[0];
          size = values[1];
          type = values[2];
          img  = values[3];
        
          currImg = document.getElementById("beam"+beam);
          if (currImg) {
            if (currImg.src != img) {
              currImg.src = img
            }
          }
        }
      }
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

      var type = "bp";

      if (document.imageform.imagetype[0].checked == true) {
        type = "bp";
      }

      if (document.imageform.imagetype[1].checked == true) {
        type = "ts";
      }

      if (document.imageform.imagetype[2].checked == true) {
        type = "fft";
      }

      //if (document.imageform.imagetype[3].checked == true) {
      //  type = "dts";
      //}

      if (document.imageform.imagetype[3].checked == true) {
        type = "pdbp";
      }

      if (document.imageform.imagetype[4].checked == true) {
        type = "pvf";
      }

      /* This URL will return the names of the 5 current */
      var url = "plotupdate.php?type="+type+"&size=112x84&beam=all";

      http_request.open("GET", url, true)
      http_request.send(null)
    }


  </script>
<?
  $inst->close_head();
?>
<body onload="looper()">
<script type="text/javascript" src="/js/wz_tooltip.js"></script>
<center>
<table border=0 cellspacing=0 cellpadding=5>

  <tr>
    <td rowspan=3 valign="top">
      <form name="imageform" class="smalltext">
      <input type="radio" name="imagetype" id="imagetype" value="bp" checked onClick="request()">Bandpass<br>
      <input type="radio" name="imagetype" id="imagetype" value="ts" onClick="request()">Time Series<br>
      <input type="radio" name="imagetype" id="imagetype" value="fft" onClick="request()">Fluct. PS<br>
      <input type="radio" name="imagetype" id="imagetype" value="pdbp" onClick="request()">PD Bandpass<br>
      <input type="radio" name="imagetype" id="imagetype" value="pvf" onClick="request()">Phase v Freq<br>
      </form>
    </td>

    <?echoBeam(13)?>
    <?echoBlank()?>
    <?echoBeam(12)?>
    <?echoBlank()?> 
  </tr>
  <tr height=42>
    <?echoBeam(6)?>
    <?echoBlank()?>
  </tr>
  <tr height=42>
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

  $beam_str = sprintf("%02d", $beam_no);

  echo "<td rowspan=2 align=right>";
  echo "<a border=0px href=\"javascript:popWindow('beamwindow.php?beamid=".$beam_no."')\">";
  echo "<img src=\"/images/blankimage.gif\" border=0 width=113 height=85 id=\"beam".$beam_str."\" TITLE=\"Beam ".$beam_str."\" alt=\"Beam ".$beam_no."\" ".$mousein." ".$mouseout.">\n";
  echo "</a></td>\n";

}

