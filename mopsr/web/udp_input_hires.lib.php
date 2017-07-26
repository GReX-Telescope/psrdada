<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class udp_input_hires extends mopsr_webpage 
{
  var $inst = 0;

  var $pfb_id;

  var $pfb_name;

  var $img_size = "120x90";

  var $inputs = array (0, 1, 2, 3, 4, 5, 6, 7);

  var $plot_types = array("wf", "hg", "ts", "bp");

  var $plot_titles = array("wf" => "Waterfall", "ts" => "TimeSeries", "bp" => "Bandpass", "hg" => "Histogram");

  var $update_secs = 5;

  function udp_input_hires()
  {
    mopsr_webpage::mopsr_webpage();
    $this->pfb_id = $_GET["pfb_id"];
    $this->pfb_name = $_GET["pfb_name"];
    $this->title = "UDP Monitor [".$this->pfb_name."]";
    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();

    $this->host = "unknown";
    for ($ipwc=0; $ipwc<$this->inst->config["NUM_PWC"]; $ipwc++)
    {
      if ($this->inst->config["PWC_PFB_ID_".$ipwc] == $this->pfb_id)
      {
        $this->host = $this->inst->config["PWC_".$ipwc];
      }
    }
  }

  function javaScriptCallback()
  {
    return "udp_input_hires_update_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      function popImage(URL) {
        day = new Date();
        id = day.getTime();
        eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
      }

      function reset_others(excluded) 
      {
        var imgs = document.getElementsByTagName('img');
        var i=0;
        for (i=0; i< imgs.length; i++) 
        {
          if ((excluded.indexOf(imgs[i].id) == -1) && (imgs[i].id.indexOf("_mgt") == -1))
          {
            imgs[i].src = "/images/blankimage.gif";
          }
        }
      }

      function handle_udp_input_hires_update_request(xml_request) 
      {

        if (xml_request.readyState == 4)
        {
          var xmlDoc = xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;
            var img_prefix  = xmlObj.getElementsByTagName("img_prefix")[0].childNodes[0].nodeValue;

            var error_message;
            var error_element = xmlObj.getElementsByTagName("error")[0];
            try {
              error_message = error_element.childNodes[0].nodeValue;
              if (error_message != "")
              {
                document.getElementById("pfb_monitor_error").innerHTML = error_message;
                if (error_message != "None")
                {
                  return;
                }
              }
            } catch (e) {
              // do nothing
            }


            var i, j;
            var excluded = Array();
            var udp_input_hiress = xmlObj.getElementsByTagName ("udp_input_hires");

            // parse XML for the UDP Input (should only be 1)
            for (i=0; i<udp_input_hiress.length; i++)
            {
              var udp_input_hires = udp_input_hiress[i];
              var udp_input_hires_id = udp_input_hires.getAttribute("id");
              var udp_input_hires_host = udp_input_hires.getAttribute("host");

              // boards will have params and plots
              var nodes = udp_input_hires.childNodes;
              for (j=0; j<nodes.length; j++)
              {
                var inputs = nodes[j].childNodes;
                for (k=0; k<inputs.length; k++)
                {
                  var input = inputs[k];
                  var input_id = input.getAttribute("id");

                  // for each image in this input
                  var children = input.childNodes;
                  for (l=0; l<children.length; l++)
                  {
                    child = children[l];
                    if (child.nodeType == 1)
                    {
                      var type = child.getAttribute("type");
                      var img_id = input_id + "_" + type;
                      var imgurl = http_server + "/" + url_prefix + "/" + img_prefix + "/" + child.childNodes[0].nodeValue;
                      var linkurl = http_server + "/" + url_prefix + "/udp_plot.php?pfb_id="+udp_input_hires_id+"&host="+udp_input_hires_host+"&pfb_input_id="+input_id+"&type="+type+"&res=1024x768";
                      if (child.getAttribute("width") < 300)
                      {
                        excluded.push(img_id);
                        document.getElementById (img_id).src = imgurl;
                        document.getElementById (img_id + "_link").href = "javascript:popImage('"+linkurl+"')";
                      }
                      else
                      {
                      }
                    }
                  }
                }
              }
            }
            reset_others (excluded);
          }
        }
      }
                  
      function udp_input_hires_update_request() 
      {
        var url = "udp_input_hires.lib.php?update=true&pfb_id=<?echo $this->pfb_id?>";

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_udp_input_hires_update_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

      function udp_input_hires_action_request (key) 
      {
        var value = document.getElementById(key + "_new").value;
        var url = "udp_input_hires.lib.php?action=true&pfb_id=<?echo $this->pfb_id?>&key="+key+"&value="+value;
        var xml_request;

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_udp_input_hires_update_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

    </script>

    <style type='text/css'>

      .left {
        text-align: left;
      }

      table.udp_input_hires tr {
        padding: 0;
      }

    </style>
<?
  }

  function printHTML() 
  {
    $this->openBlockHeader("UDP Input from ".$this->pfb_name." on server ".$this->host);
    list ($xres, $yres) = split("x", $this->img_size);

    echo "<table class='udp_input_hires'>\n";
    echo " <tr>\n";

    echo "  <th>Input</th>\n";
    foreach ($this->plot_types as $plot)
    {
      echo "  <th>".$this->plot_titles[$plot]."</th>\n";
    }

    echo "  <th width='30px'>&nbsp;</th>\n";

    echo "  <th>Input</th>\n";
    foreach ($this->plot_types as $plot)
    {
      echo "  <th>".$this->plot_titles[$plot]."</th>\n";
    }
    echo " </tr>\n";

    $half_inputs = count($this->inputs)/2;
    for ($i=0; $i<$half_inputs; $i++)      
    {
      echo " <tr>\n";

      $input = $this->inputs[$i];
      echo "  <td id='".$input."_id'>".$input."</td>\n";
      foreach ($this->plot_types as $plot)
      {
        echo "  <td>";
        echo "    <a id='".$input."_".$plot."_link'>";
        echo "      <img id='".$input."_".$plot."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px'/>";
        echo "   </a>";
        echo "  </td>\n";
      }

      echo "  <td></td>\n";

      $input = $this->inputs[$i + $half_inputs];
      echo "  <td id='".$input."_id'>".$input."</td>\n";
      foreach ($this->plot_types as $plot)
      {
        echo "  <td>";
        echo "    <a id='".$input."_".$plot."_link'>";
        echo "      <img id='".$input."_".$plot."' src='/images/blackimage.gif' width='".$xres."px' height='".$yres."px'/>";
        echo "   </a>";
        echo "  </td>\n";
      }

      echo " </tr>\n";
    }
    echo "</table>\n";

    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {
    # since this will only be done infrequently and not by many http clients 
    # (I hope!), do a filesystem lookup!

    $xml_reply = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml_reply .= "<udp_input_hires_update>";
    $xml_reply .=   "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml_reply .=   "<url_prefix>mopsr</url_prefix>";
    $xml_reply .=   "<img_prefix>monitor/udp</img_prefix>";

    $xml_reply .=   "<udp_input_hires id='".$this->pfb_id."' host='".$this->host."'>";

    # get a listing of the images for this UDP input
    $cmd = "find ".$this->inst->config["SERVER_UDP_MONITOR_DIR"]." -name '2???-??-??-??:??:??.".$this->pfb_id.".*.??.?.*x*.png' -printf '%f\n' | sort -n";
    $images = Array();
    $lastline = exec($cmd, $images, $rval);
    $to_use = Array();
    $locked = Array();
    if (($rval == 0) && (count($images) > 0))
    {
      # use associative array to store only the most recent images of a input + type + resolution
      foreach ($images as $image)
      {
        list ($time, $rid, $input, $type, $lock, $res, $ext) = explode(".", $image);
        if (!array_key_exists($input, $to_use))
          $to_use[$input] = Array();
        $to_use[$input][$type.".".$res] = $image;
        $locked[$input] = ($lock == "L") ? "true" : "false";
      }

      $xml_reply .= "<inputs>";
      foreach ($this->inputs as $input)
      {
        $xml_reply .= "<input id='".$input."' locked='".$locked[$input]."'>";

        if (array_key_exists($input, $to_use))
        {
          foreach (array_keys($to_use[$input]) as $key)
          {
            list ($type, $res) = explode(".", $key);
            list ($xres, $yres) = explode("x", $res);
            $xml_reply .= "<plot type='".$type."' width='".$xres."' height='".$yres."'>".$to_use[$input][$key]."</plot>";
          }
        }
        $xml_reply .= "</input>";
      }
      $xml_reply .= "</inputs>";
    }
    else
    {
      $xml_reply .= "<error return_value='".$rval."'>".$lastline."</error>";
    }

    $xml_reply .= "</udp_input_hires>";
    $xml_reply .= "</udp_input_hires_update>";

    header('Content-type: text/xml');
    echo $xml_reply;
  }
}

handleDirect("udp_input_hires");

