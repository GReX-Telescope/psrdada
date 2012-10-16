<?PHP

include_once("dada_webpage.lib.php");
include_once("instrument.lib.php");

class admin extends dada_webpage 
{

  var $config = array();
  var $fields = array();
  var $machines = array();

  function admin()
  {
    dada_webpage::dada_webpage();

    $this->config = instrument::configFileToHash("/home/dada/linux_64/share/dada.cfg");

    for ($i=0; $i<$this->config["NUM_HOSTS"]; $i++)
    {
      array_push($this->machines, $this->config["HOST_".$i]);
    }

    $this->fields["service_tag"] = "Service Tag";
    $this->fields["bios"] = "BIOS";
    $this->fields["rac_fw"] = "RAC F/W";
    $this->fields["raid_fw"] = "RAID F/W";
    $this->fields["raid_drv"] = "RAID Drv";
    $this->fields["omsa_ver"] = "OMSA Ver";
    $this->fields["os"] = "O/S";
    $this->fields["kernel"] = "Kernel";
    $this->fields["disk0_id"] = "Disk0 ID";
    $this->fields["disk0_fw"] = "Disk0 F/W";
    $this->fields["disk1_id"] = "Disk1 ID";
    $this->fields["disk1_fw"] = "Disk2 F/W";

  }

  function printJavaScriptHead()
  {
?>
    <style type="text/css">
      td {
        padding-left: 5px;
      }

      th {
        background-color: #cae2ff;
      }
    </style>
    <script type='text/javascript'> 

      function popUp(URL, type) {

        var to = "toolbar=1";
        var sc = "scrollbars=1";
        var l  = "location=1";
        var st = "statusbar=1";
        var mb = "menubar=1";
        var re = "resizeable=1";

        options = to+","+sc+","+l+","+st+","+mb+","+re
        eval("page" + type + " = window.open(URL, '" + type + "', '"+options+",width=1024,height=768');");
      }

      function handle_admin_request(ad_xml_request) 
      {
        if (ad_xml_request.readyState == 4) 
        {
          var xmlDoc = ad_xml_request.responseXML;
          if (xmlDoc != null)
          {
            var xmlObj = xmlDoc.documentElement;
            var host_report = xmlObj.getElementsByTagName("host_report");
            var host = host_report[0].getAttribute("host");
            var children = host_report[0].childNodes;
            for (i=0; i<children.length; i++) 
            {
              node = children[i];
              if (node.nodeType == 1)
              {
                key = node.nodeName;
                val = node.childNodes[0].nodeValue;
                document.getElementById(host+"_"+key).innerHTML = val;
              }
            }
          }
        }
      }

      function admin_request(host) 
      {
        var ad_xml_request;
        var url = "admin.lib.php?update=true&host="+host;

        if (window.XMLHttpRequest)
          ad_xml_request = new XMLHttpRequest()
        else
          ad_xml_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        ad_xml_request.onreadystatechange = function() 
        {
          handle_admin_request(ad_xml_request)
        }

        ad_xml_request.open("GET", url, true)
        ad_xml_request.send(null)

      }

      function get_admin()
      {
<?
        for ($i=0; $i<$this->config["NUM_HOSTS"]; $i++)
        {
          echo "         admin_request('".$this->config["HOST_".$i]."');\n";
        }
?>
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML()
  {

    $keys = array_keys($this->fields);

    $this->openBlockHeader("Machines");
?>
    <table width="100%">
<?
    
    echo "      <tr>\n";
    echo "        <th>Host</th>\n";
    echo "        <th>RAC</th>\n";
    echo "        <th>OMSA</th>\n";
    for ($i=0; $i<count($keys); $i++) 
    {
      $k = $keys[$i];
      echo "        <th>".$this->fields[$k]."</th>\n";
    }
    echo "      </tr>\n";

    for ($i=0; $i<count($this->machines); $i++) 
    {
      $m = $this->machines[$i];
      echo "      <tr>\n";
      echo "        <td>".$this->machines[$i]."</td>\n";

      if ($this->config["RAC_".$i] != "")
        echo "        <td><a href='https://".$this->config["RAC_".$i]."/'>link</a></td>\n";
      else
        echo "        <td>-</td>\n";

      if (array_key_exists("OMSA_".$i, $this->config) && $this->config["OMSA_".$i] != "")
        echo "        <td><a href='https://".$this->config["OMSA_".$i]."/'>link</a></td>\n";
      else
        echo "        <td>-</td>\n";

      for ($j=0; $j<count($keys); $j++) 
      {
        echo "        <td id='".$this->config["HOST_".$i]."_".$keys[$j]."'></td>\n";
      }
      echo"       </tr>\n";
    }
?>
    </table>
    <script type='text/javascript'>
      get_admin();
    </script>
<?
    $this->closeBlockHeader();
    
  }

  function printUpdateHTML($host, $port)
  {
    $cmd = "/usr/bin/ssh ".$host." 'om_custom_report.csh'";
    $array=array();
    $last = exec($cmd, $array, $rval);
    $results = array();

    # produce the xml
    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<admin_report>";
    $xml .= "<host_report host='".$host."'>";
    for ($i=0; $i<count($array); $i++) {
      $xml .= $array[$i];
    }
    $xml .= "</host_report>";
    $xml .= "</admin_report>";

    header('Content-type: text/xml');
    echo $xml;
  }
}
handledirect("admin");


