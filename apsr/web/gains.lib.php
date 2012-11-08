<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class gains extends apsr_webpage 
{

  var $machines = array();
  var $config = array();

  function gains()
  {
    apsr_webpage::apsr_webpage();

    $this->callback_freq = 10000;

    array_push($this->ejs, "/js/prototype.js");
    array_push($this->ejs, "/js/jsProgressBarHandler.js");

    $inst = new apsr();
    $this->config = $inst->config;

    /* generate a list of machines */
    for ($i=0; $i<$this->config["NUM_PWC"]; $i++) {
      array_push($this->machines, $this->config["PWC_".$i]);
    }

    $this->machines = array_unique ($this->machines); 
    $this->machines = array_values ($this->machines); 
  }

  function javaScriptCallback()
  {
    return "gains_request();";
  }

  function printJavaScriptHead()
  {
?>
    <style type="text/css">
      table.gains {
        /*border-collapse: collapse;*/
      }
      table.gains th {
        font-size: 8pt;
        font-weight: normal;
        /*margin: 0px 10px 0px 10px;
        border-style: none none none solid;
        border-width: 0px 0px 0px 1px;*/
      }
      table.gains td { 
        font-size: 8pt;
        margin: 0px;
        border-style: none none none none;
        border-width: 0px;
      }
      table.gains span { 
        font-size: 8pt;
      }
      #gap {
        padding-right: 20px;
      }
    </style>


    <script type='text/javascript'>  

      function handle_gains_request(ga_http_request) 
      {
        if (ga_http_request.readyState == 4) {

          var response = String(ga_http_request.responseText)
          var values;

          /* set the values to 0 */
          if ((response.indexOf("Could not connect") == 0) || (response.indexOf("Connection reset by peer") == 0)) {
            values = new Array(<?for ($i=0; $i<count($this->machines); $i++) { echo "32768,32768,";}?>65535);
          } else {
            values = response.split(" ");
          }

          var val;
          var percent;
          var max = values[<?echo (count($this->machines)*2);?>];
          var logmax = Math.log(max);
          //document.getElementById('max_gain').innerHTML = "Max Gain "+max

<?
          for ($i=0; $i<count($this->machines); $i++) 
          {
            $m = $this->machines[$i];
            # PWC_i pol0
            echo "      val = values[".(2*$i)."];\n";
            echo "      percent = Math.floor(100 * Math.log(parseInt(val)) / logmax);\n";
            echo "      ".$m."_pol0.setPercentage(percent);\n";
            echo "      document.getElementById('".$m."_pol0_value').innerHTML = '&nbsp;'+val;\n";

            # PWC_i pol1
            echo "      val = values[".((2*$i)+1)."];\n";
            echo "      percent = Math.floor(100 * Math.log(parseInt(val)) / logmax);\n";
            echo "      ".$m."_pol1.setPercentage(percent);\n";
            echo "      document.getElementById('".$m."_pol1_value').innerHTML = '&nbsp;'+val;\n";
          }
?>
        }
      }

      function gains_request() 
      {
        var url = "gains.lib.php?update=true&host=<?echo $this->config["SERVER_HOST"]?>&port=<?echo $this->config["SERVER_WEB_MONITOR_PORT"]?>";
  
        if (window.XMLHttpRequest)
          ga_http_request = new XMLHttpRequest();
        else
          ga_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        ga_http_request.onreadystatechange = function() {
          handle_gains_request(ga_http_request)
        };
        ga_http_request.open("GET", url, true);
        ga_http_request.send(null);
      }

    </script>
<?
  }

  function printJavaScriptBody() 
  {
?>
    <script type="text/javascript">
      Event.observe(window, 'load',  function() 
      {
 <?
        for ($i=0; $i<count($this->machines); $i++) 
        {
          $m = $this->machines[$i];

          echo "        ".$m."_pol0 = new JS_BRAMUS.jsProgressBar($('".$m."_pol0_bar'), 0, ";
          echo " { width : 12, height : 40, showText : false, animate : false, horizontal : false, ".
               "boxImage : '/images/jsprogress/verticalPercentImage.png', ".
               "barImage : '/images/jsprogress/verticalPercentImage_back.png' } );\n"; 

          echo "        ".$m."_pol1 = new JS_BRAMUS.jsProgressBar($('".$m."_pol1_bar'), 0, ";
          echo " { width : 12, height : 40, showText : false, animate : false, horizontal : false, ".
               "boxImage : '/images/jsprogress/verticalPercentImage.png', ".
               "barImage : '/images/jsprogress/verticalPercentImage_back.png' } );\n"; 

        }
?>
      }, false);
    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
    <center>
    <table class='gains' border='0'>
      <tr>
<?
    for ($i=0; $i<count($this->machines); $i++) {
      $m = $this->machines[$i];
      if ($i == 0)
        echo "<td align='center' colspan='2'>".substr($m,4,2)."</td>\n";
      else
        echo "<th align='center' colspan='2'>".substr($m,4,2)."</th>\n";
    }
?>
      </tr>
      <tr>
<?
    for ($i=0; $i<count($this->machines); $i++) {
      $m = $this->machines[$i];
      echo "<td align='center'>\n";
      echo "<span id=\"".$m."_pol0_bar\">[ Loading Progress Bar ]</span>\n";
      echo "<td align='center'>\n";
      echo "<span id=\"".$m."_pol1_bar\">[ Loading Progress Bar ]</span>\n";
      echo "</td>\n";
    }
?>
      </tr>
      <tr>
<?
    for ($i=0; $i<count($this->machines); $i++) {
      $m = $this->machines[$i];
      echo "<td colspan='2'>\n";
      echo "<div id='".$m."_pol0_value'></div>\n";
      echo "<div id='".$m."_pol1_value'></div>\n";
      echo "</td>\n";
    }
?>
      </tr>
      <!--<tr><td colspan='32' align='center'><span id="max_gain"></span></td></tr>-->
    </table>
    </center>
<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    if ($this->config["USE_DFB_SIMULATOR"] == 1) 
      $max_gain = 100;
    else
      $max_gain = 65535;

    $output = "";

    list ($socket, $result) = openSocket($host, $port);

    if ($result == "ok") 
    {
      $bytes_written = socketWrite($socket, "gain_info\r\n");
      list ($result, $output) = socketRead($socket);
      socket_close($socket);
    }
    else 
    {
      $output = "Could not connect to $host:$port<BR>\n";
    }
    echo $output." ".$max_gain;
  }
}

handleDirect("gains");

