<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class plot_window extends mopsr_webpage 
{
  var $inst = 0;

  var $verbose = false;

  var $pfbs_per_arm = 12;

  var $arm_prefixes = array ("EG", "WG");

  var $ants_per_pfb = 16;

  var $types = array ("fl", "fr", "ti", "bp", "pm", "tc");

  var $corr_types = array ("ad", "po", "sn", "bd");

  var $bf_cfg;

  function plot_window()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MOPSR Plot Window";
    $this->callback_freq = 2000;
    $this->verbose = isset($_GET["verbose"]);
    $this->inst = new mopsr();
    $this->bf_cfg  = $this->inst->configFileToHash(BF_FILE);
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

      function popPlotWindow(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,scrollbars=1,location=1,statusbar=0,menubar=1,resizable=1,width=1400,height=870');");
      }

      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1200,height=900');");
      }

      function reset_other_imgs (excluded) 
      {
        var imgs = document.getElementsByTagName('img');
        var i=0;
        for (i=0; i< imgs.length; i++) 
        {
          if ((imgs[i].id != "") && (imgs[i].id.indexOf("pw_") == 0) && (excluded.indexOf(imgs[i].id) == -1))
          {
            imgs[i].height = 0;
          }
        }
      }

      function reset_other_trs(excluded) 
      {
        var trs = document.getElementsByTagName('tr');
        var i=0;

        for (i=0; i< trs.length; i++) 
        {
          if ((trs[i].id != "") && (trs[i].id.indexOf("pw_") == 0) && (excluded.indexOf(trs[i].id) == -1))
          {
            trs[i].style.display = "none";
          }
        }
      }

      function handle_plot_window_request(pw_xml_request) 
      {
        if (pw_xml_request.readyState == 4)
        {
          var xmlDoc = pw_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;

            var ants = xmlObj.getElementsByTagName("ant");

            var i = 0;
            var img_id, tr_id;
            var excluded_imgs = new Array();
            var excluded_trs = new Array();
            for (i=0; i<ants.length; i++)
            {
              var ant = ants[i];
              var ant_name = ant.getAttribute("name");
              var ant_type = ant.getAttribute("type");
              var ant_reason = ant.getAttribute("reason");
              var j = 0;

              var td_name_id = "pw_name_" + ant_type;
              if ( ant_reason ) {
                document.getElementById(td_name_id).innerHTML = ant_name + "<br>" + ant_reason;
              } else {
                document.getElementById(td_name_id).innerHTML = ant_name;
              }

              for (j=0; j<ant.childNodes.length; j++)
              {
                img = ant.childNodes[j];
                if (img.nodeType == 1)
                {
                  var type = img.getAttribute("type");
                  var imgurl = http_server + url_prefix + img.childNodes[0].nodeValue;
                  img_id = "pw_" + type + "_" + ant_type;
                  if (ant_name == "FB")
                  {
                    tr_id = "pw_tr_" + type + "_"+ ant_type;
                  }
                  else
                  {
                    tr_id = "pw_tr_" + ant_type;
                  }

                  excluded_imgs.push(img_id);
                  excluded_trs.push(tr_id);

                  try {
                    document.getElementById (tr_id).style.display = "";
                  } catch (e) {
                    alert("could not set display of tr_id="+tr_id+ " to not none");
                  }

                  try {

                    if ((parseInt(img.getAttribute("width")) > 300) || (ant_type == "FB"))
                    {
                      document.getElementById (img_id + "_link").href = "javascript:popImage('"+imgurl+"')";
                    }

                    if ((parseInt(img.getAttribute("width")) <= 300) || (ant_type == "FB"))
                    {
                      document.getElementById (img_id).src = imgurl;
                      if (ant_type != "FB")
                      {
                        document.getElementById (img_id).height = img.getAttribute("height");
                      }
                      else
                      {
                        document.getElementById (img_id).height = "480";
                      } 
                    }
                  } catch (e) {
                  } 
                }
              }
            }
            reset_other_imgs(excluded_imgs);
            reset_other_trs(excluded_trs);
          }
        }
      }
                  
      function plot_window_request() 
      {
        var host = "<?echo $this->inst->config["SERVER_HOST"];?>";
        var port = "<?echo $this->inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "plot_window.lib.php?update=true&host="+host+"&port="+port;

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
    <table border=0 cellspacing=0 cellpadding=2>

      <tr id='pulsar_header' style='display: none;'>
        <th>Input</th>
        <th>Flux vs Phase</th>
        <th>Freq vs Phase</th>
        <th>Time vs Phase</th>
        <th>Bandpass</th>
        <th>Power Monitor</th>
      </tr>
<?
      for ($i=0; $i<$this->bf_cfg{"NUM_TIED_BEAMS"}; $i++)
      {
        $id = "TB".$i;
        echo "<tr id='pw_tr_".$id."' style='display: none;'>\n";
          echo "<td id='pw_name_".$id."'></td>\n";
          foreach ($this->types as $type)
          {
            echo "<td>";
            echo "<a href='' id='pw_".$type."_".$id."_link'>";
            echo "<img id='pw_".$type."_".$id."' src='' width='121px' height='91px'>";
            echo "</a>";
            echo "</td>\n";
          }
      }
          
?>
      </tr>
<?
    for ($iarm=0; $iarm<count($this->arm_prefixes); $iarm++)
    {
      for ($i=0; $i<$this->pfbs_per_arm; $i++)
      {
        for ($j=0; $j<$this->ants_per_pfb; $j++)
        {
          $id = sprintf("%s%02d_%d", $this->arm_prefixes[$iarm], ($i+1), $j);
          echo "<tr id='pw_tr_".$id."' style='display: none;'>\n";
          echo "<td id='pw_name_".$id."'></td>\n";
          foreach ($this->types as $type)
          {
            echo "<td>";
            echo "<a href='' id='pw_".$type."_".$id."_link'>";
            echo "<img id='pw_".$type."_".$id."' src='' width='121px' height='91px'>";
            echo "</a>";
            echo "</td>\n";
          }
          echo "</tr>\n";
        }
      }
    }
?>
    </table>
    <table border=0 cellspacing=0 cellpadding=2>
      <tr id='correlation_header' style='display: none;'>
        <th>Ant Delay</th><th>Ant Phase</th><th>Delay</th><th>SNR</th>
      </tr>
<?
      for ($ichan=0; $ichan<40; $ichan++)
      {
        $id = "CORR";
        echo "<tr id='pw_tr_".$id."' style='display: none;'>\n";
        echo "<td id='pw_name_".$id."'>".$id."</td>\n";
        foreach ($this->corr_types as $type)
        {
          echo "<td>";
          echo "<a href='' id='pw_".$type."_".$id."_link'>";
          echo "<img id='pw_".$type."_".$id."' src='' width='160px' height='120px'>";
          echo "</a>";
          echo "</td>\n";
        }

        echo "</tr>\n";
      }
?>

    </table>

    <table border=0 cellspacing=0 cellpadding=2>
      <tr id='fanbeam_header' style='display: none;'>
        <td id='pw_name_FB'></td>
      </tr>
<?
      for ($i=127; $i>=0; $i--)
      {
        $id = "FB";
        $type = sprintf("%02d", $i);
        echo "<tr id='pw_tr_".$type."_".$id."' style='display: none;'>";
          
          echo "<td>";
          echo "<a href='' id='pw_".$type."_".$id."_link'>";
          echo "<img id='pw_".$type."_".$id."' src='' width='640px' height='480px'>";
          echo "</a>";
          echo "</td>";
        echo "</tr>\n";
      }
?>  
    </table>
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
    $xml .= "<url_prefix>/mopsr/</url_prefix>";

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
}

handleDirect("plot_window");

