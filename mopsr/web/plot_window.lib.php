<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class plot_window extends mopsr_webpage 
{

  var $inst = 0;
  var $verbose = false;

  function plot_window()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MOPSR Plot Window";
    $this->verbose = isset($_GET["verbose"]);
  }

  function javaScriptCallback()
  {
    return "plot_window_request();";
  }

  function printJavaScriptHead()
  {
    $this->inst = new mopsr();

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
            imgs[i].src = "/mopsr/images/mopsr_beam_disabled_240x180.png";
            tmpstr += "id="+imgs[i].id+",";
          }
        }
      }

      function handle_plot_window_request(pw_xml_request) 
      {
<?
      if ($this->verbose)
        echo "        var verbose = true;\n";
      else
        echo "        var verbose = false;\n";
?>

        if (pw_xml_request.readyState == 4)
        {
          var xmlDoc = pw_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;

            // determine which types of images are currently active
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

  
            var set = new Array();
            var beams = xmlObj.getElementsByTagName("beam");

            var i = 0;
            for (i=0; i<beams.length; i++)
            {
              var beam = beams[i];
              var beam_name = beam.getAttribute("name");
              var img_element = document.getElementById("beam"+beam_name);

              var j = 0;
              for (j=0; j<beam.childNodes.length; j++)
              {
                img = beam.childNodes[j];
                if (img.nodeType == 1)
                { 
                  if (img.getAttribute("type") == type)
                  {
                    set.push("beam"+beam_name);
                    img_element.src = http_server + url_prefix + img.childNodes[0].nodeValue;
                  }
                }
              }
            }

            reset_others(set);
          }
        }
      }
                  
      function plot_window_request() 
      {
        var host = "<?echo $this->inst->config["SERVER_HOST"];?>";
        var port = "<?echo $this->inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "plot_window.lib.php?update=true&host="+host+"&port="+port;

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
          pw_xml_request = new XMLHttpRequest();
        else
          pw_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        pw_xml_request.onreadystatechange = function() {
          handle_plot_window_request(pw_xml_request)
        };
        pw_xml_request.open("GET", url, true);
        pw_xml_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
  <center>
    <table border=0 cellspacing=0 cellpadding=4>

      <tr>
        <td rowspan=3 valign="top" align='left'>
          <form name="imageform" class="smalltext">
            <input type="radio" name="imagetype" id="imagetype" value="bp" onClick="plot_window_request()">Bandpass<br>
            <input type="radio" name="imagetype" id="imagetype" value="ts" onClick="plot_window_request()">Time Series<br>
            <input type="radio" name="imagetype" id="imagetype" value="fft" onClick="plot_window_request()">Fluct. PS<br>
            <input type="radio" name="imagetype" id="imagetype" value="pdbp" checked onClick="plot_window_request()">PD Bandpass<br>
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
<!--
    <a href="/mopsr/transient_window.lib.php?single=true" target="_popup">Transients!!!</a>
-->
  </center>

<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    list ($socket, $result) = openSocket($host, $port);

    $xml = "<plot_update>";
    $xml .= "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .= "<url_prefix>/mopsr/results/</url_prefix>";

    $data = "";
    $response = "initial";

    if ($result == "ok") 
    {
      $xml .= "<images>";
      $bytes_written = socketWrite($socket, "img_info\r\n");
      $max = 100;
      while ($socket && ($result == "ok") && ($response != "") && ($max > 0))
      {
        list ($result, $response) = socketRead($socket);
        if ($result == "ok") 
        {
          $data .= $response;
        }
        $max--;
      }
      if (($result == "ok") && ($socket))
        socket_close($socket);
      $socket = 0;
      $xml .= $data;
      $xml .="</images>";
    } 

    $xml .= "</plot_update>";

    header('Content-type: text/xml');
    echo $xml;
  }

  function echoBlank() 
  {
    echo "<td><img src='/images/spacer.gif' width='113px' height='42px'</td>\n";
  }

  function echoBeam($beam_no) 
  {
    $beam_str = sprintf("%02d", $beam_no);
    $roach_name = "Inactive";

    for ($i=0; $i<$this->inst->roach["NUM_ROACH"]; $i++)
      if ($this->inst->roach["BEAM_".$i] == $beam_str)
        $roach_name = $this->inst->roach["ROACH_".$i];

    echo "<td rowspan=2 align=right>\n";
    echo "          <a border=0px href=\"javascript:popPlotWindow('beam_viewer.lib.php?single=true&beam=".$beam_str."')\">";
    echo "<img src=\"/images/blankimage.gif\" border=0 width=113 height=85 id=\"beam".$beam_str."\" TITLE=\"Beam ".$beam_str." - ".$roach_name."\" alt=\"Beam ".$beam_no."\">";
    echo "</a>\n";
    echo "        </td>\n";
  }

}

handleDirect("plot_window");

