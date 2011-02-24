<?PHP

include("bpsr.lib.php");
include("bpsr_webpage.lib.php");

class plot_window extends bpsr_webpage 
{

  function plot_window()
  {
    bpsr_webpage::bpsr_webpage();
    $this->title = "BPSR Plot Window";
  }

  function javaScriptCallback()
  {
    return "plot_window_request();";
  }

  function printJavaScriptHead()
  {

    $inst = new bpsr();

?>
    <script type='text/javascript'>  

      var npsrs = 0;
      var utc_start = "";
      var psrs = new Array();

      function popPlotWindow(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,scrollbars=1,location=1,statusbar=0,menubar=1,resizable=1,width=1400,height=870');");
      }

      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
      }

      function reset_others(excluded) 
      {

        var imgs = document.getElementsByTagName('img');
        var i=0;
        var tmpstr = "resetting ";
        for (i=0; i< imgs.length; i++) {
          if ((imgs[i].id.indexOf("beam") == 0) && (excluded.indexOf(imgs[i].id) == -1)) {
            imgs[i].src = "/bpsr/images/bpsr_beam_disabled_240x180.png";
            tmpstr += "id="+imgs[i].id+",";
          }
        }
      }


      function handle_plot_window_request(pw_http_request) 
      {
        if (pw_http_request.readyState == 4) {

          var response = String(pw_http_request.responseText)
          var lines = response.split("\n");
          var currImg
          var beam
          var size
          var type
          var img

          var set = new Array();

          for (i=0; i < lines.length-1; i++) {

            values = lines[i].split(":::");
            beam = values[0];
            size = values[1];
            type = values[2];
            img  = values[3];
          
            currImg = document.getElementById("beam"+beam);
            if (currImg) {
              set.push("beam"+beam);
              if (currImg.src != img) {
                currImg.src = img
              }
            }
          }
          reset_others(set);
        }
      }

      function plot_window_request() 
      {
        var host = "<?echo $inst->config["SERVER_HOST"];?>";
        var port = "<?echo $inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "plot_window.lib.php?update=true&host="+host+"&port="+port;

        var type = "bp";
        if (document.imageform.imagetype[0].checked == true) 
          type = "bp";

        if (document.imageform.imagetype[1].checked == true) 
          type = "ts";

        if (document.imageform.imagetype[2].checked == true) 
          type = "fft";

        if (document.imageform.imagetype[3].checked == true) 
          type = "pdbp";

        if (document.imageform.imagetype[4].checked == true) 
          type = "pvf";

        url += "&type="+type+"&size=112x84&beam=all";

        if (window.XMLHttpRequest)
          pw_http_request = new XMLHttpRequest();
        else
          pw_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        pw_http_request.onreadystatechange = function() {
          handle_plot_window_request(pw_http_request)
        };
        pw_http_request.open("GET", url, true);
        pw_http_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
  <center>
    <table border=0 cellspacing=0 cellpadding=5>

      <tr>
        <td rowspan=3 valign="top" align='left'>
          <form name="imageform" class="smalltext">
            <input type="radio" name="imagetype" id="imagetype" value="bp" checked onClick="plot_window_request()">Bandpass<br>
            <input type="radio" name="imagetype" id="imagetype" value="ts" onClick="plot_window_request()">Time Series<br>
            <input type="radio" name="imagetype" id="imagetype" value="fft" onClick="plot_window_request()">Fluct. PS<br>
            <input type="radio" name="imagetype" id="imagetype" value="pdbp" onClick="plot_window_request()">PD Bandpass<br>
            <input type="radio" name="imagetype" id="imagetype" value="pvf" onClick="plot_window_request()">Phase v Freq<br>
          </form>
        </td>
        
        <?$this->echoBeam(13)?>
        <?$this->echoBlank()?>
        <?$this->echoBeam(12)?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(6)?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(7)?>
        <?$this->echoBeam(5)?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(8)?>
        <?$this->echoBeam(1)?>
        <?$this->echoBeam(11)?>
      </tr>

      <tr height=42>
        <?$this->echoBeam(2)?>
        <?$this->echoBeam(4)?>
      </tr>

      <tr height=42>
        <?$this->echoBlank()?>
        <?$this->echoBeam(3)?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBlank()?>
        <?$this->echoBeam(9)?>
        <?$this->echoBeam(10)?>
        <?$this->echoBlank()?>
      </tr>

      <tr height=42>
        <?$this->echoBlank()?>
        <?$this->echoBlank()?>
        <?$this->echoBlank()?>
      </tr>
    </table>
  </center>

<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    $type = $_GET["type"];
    $size = "112x84";

    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    list ($socket, $result) = openSocket($host, $port);

    if ($result == "ok") {

      $bytes_written = socketWrite($socket, $type."_img_info\r\n");
      $string = socketRead($socket);
      socket_close($socket);

      # Add the require URL links to the image
      $lines = split(";;;", $string);
      $string = "";
      for ($i=0; $i<count($lines)-1; $i++) {

        $parts = split(":::", $lines[$i]);
        $node = -1;
        if ($type == "pdbp") {
          $inst = new bpsr();
          for ($j=0; $j<$inst->ibobs["NUM_IBOB"]; $j++) {
            if ($parts[0] == $inst->ibobs["CONTROL_IP_".$j]) {
              $node = $inst->ibobs["BEAM_".$j];
            }
          }
        } else {
          $node = $parts[0];
        }
        if ($node != -1) {
          $string .= $node.":::".$size.":::".$type.":::".$url."/bpsr/results/".$parts[1]."\n";;
        } else {
          $string .= $node.":::".$size.":::".$type.":::".$url."/images/blankimage.gif\n";
        }

      }

    } else {
      $string = "Could not connect to $host:$port<BR>\n";
    }

    echo $string;
  }

  function echoBlank() 
  {
    echo "<td ></td>\n";
  }

  function echoBeam($beam_no) 
  {
    $beam_str = sprintf("%02d", $beam_no);

    echo "<td rowspan=2 align=right>";
    echo "<a border=0px href=\"javascript:popPlotWindow('beam_viewer.lib.php?single=true&beam=".$beam_str."')\">";
    echo "<img src=\"/images/blankimage.gif\" border=0 width=113 height=85 id=\"beam".$beam_str."\" TITLE=\"Beam ".$beam_str."\" alt=\"Beam ".$beam_no."\">\n";
    echo "</a></td>\n";
  }

}

handledirect("plot_window");

