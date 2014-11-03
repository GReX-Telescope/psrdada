<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class corr_plot extends mopsr_webpage 
{
  var $inst = 0;

  var $verbose = false;

  var $nchan = 40;

  var $types = array ("dl", "sn");

  function corr_plot()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MOPSR Corr Plot";
    $this->callback_freq = 2000;
    $this->verbose = isset($_GET["verbose"]);
  }

  function javaScriptCallback()
  {
    return "corr_plot_request();";
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
          if ((imgs[i].id != "") && (imgs[i].id.indexOf("cp_") == 0) && (excluded.indexOf(imgs[i].id) == -1))
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
          if ((trs[i].id != "") && (trs[i].id.indexOf("cp_") == 0) && (excluded.indexOf(trs[i].id) == -1))
          {
            trs[i].style.display = "none";
          }
        }
      }

      function handle_corr_plot_request(cp_xml_request) 
      {
        if (cp_xml_request.readyState == 4)
        {
          var xmlDoc = cp_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;

            var chans = xmlObj.getElementsByTagName("chan");

            var i = 0;
            var img_id, tr_id;
            var excluded_imgs = new Array();
            var excluded_trs = new Array();
            for (i=0; i<chans.length; i++)
            {
              var chan = chans[i];
              var chan_name = chan.getAttribute("name");

              var j = 0;
              for (j=0; j<chan.childNodes.length; j++)
              {
                img = chan.childNodes[j];
                if (img.nodeType == 1)
                {
                  var type = img.getAttribute("type");
                  var imgurl = http_server + url_prefix + img.childNodes[0].nodeValue;
                  img_id = "cp_" + type + "_" + chan_name;
                  tr_id = "ch_" + chan_name;

                  excluded_imgs.push(img_id);
                  excluded_trs.push(tr_id);

                  try {
                    document.getElementById (tr_id).style.display = "";
                  } catch (e) {
                    alert("tr_id="+tr_id);
                  }

                  if (parseInt(img.getAttribute("width")) > 300)
                  {
                    document.getElementById (img_id + "_link").href = "javascript:popImage('"+imgurl+"')";
                    document.getElementById (img_id).src = imgurl;
                    document.getElementById (img_id).width = 160;
                    document.getElementById (img_id).height = 120;
                  }
                  else
                  {
                    document.getElementById (img_id).src = imgurl;
                    document.getElementById (img_id).height = 91;
                  }
                }
              }
            }
            reset_other_imgs(excluded_imgs);
            reset_other_trs(excluded_trs);
          }
        }
      }
                  
      function corr_plot_request() 
      {
        var host = "<?echo $this->inst->config["SERVER_HOST"];?>";
        var port = "<?echo $this->inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "corr_plot_window.lib.php?update=true&host="+host+"&port="+port;

        if (window.XMLHttpRequest)
          cp_xml_request = new XMLHttpRequest();
        else
          cp_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        cp_xml_request.onreadystatechange = function() {
          handle_corr_plot_request(cp_xml_request)
        };
        cp_xml_request.open("GET", url, true);
        cp_xml_request.send(null);
      }

    </script>

<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
  <center>
    <table border=1 cellspacing=0 cellpadding=2>

      <tr>
        <th>Channel</th>
        <th>Delays</th>
        <th>SNRs</th>
      </tr>
<?
    for ($ichan=0; $ichan<$this->nchan; $ichan++)
    {
      $id = sprintf("CH%02d", $ichan);
      echo "<tr id='ch_".$id."' style='display: none;'>\n";
      echo "<td>".$id."</td>\n";
      foreach ($this->types as $type)
      {
        echo "<td>";
        echo "<a href='' id='cp_".$type."_".$id."_link'>";
        echo "<img id='cp_".$type."_".$id."' src='' width='200px' height='150px'>";
        echo "</a>";
        echo "</td>\n";
      }
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

    $xml = "<corr_plot_update>";
    $xml .= "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .= "<url_prefix>/mopsr/</url_prefix>";

    $data = "";
    $response = "initial";

    if ($result == "ok") 
    {
      $xml .= "<images>";
      $bytes_written = socketWrite($socket, "corr_img_info\r\n");
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

    $xml .= "</corr_plot_update>";

    header('Content-type: text/xml');
    echo $xml;
  }
}

handleDirect("corr_plot");

