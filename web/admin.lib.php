<?PHP

include("dada_webpage.lib.php");
include("instrument.lib.php");

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

      function handle_admin_request(ad_http_request) 
      {
        if (ad_http_request.readyState == 4) {

          var response = String(ad_http_request.responseText)
          var lines = response.split("\n");
          var bits;

          bits = lines[0].split(":::");
          host = bits[1];
          for (i=1; i<lines.length; i++)
          {
            if (lines[i].length > 3) 
            {
              bits = lines[i].split(":::");
              document.getElementById(host+"_"+bits[0]).innerHTML = bits[1];
            }
          }
        }
      }

      function admin_request(host) 
      {
        var ad_http_request;
        var url = "admin.lib.php?update=true&host="+host;

        if (window.XMLHttpRequest)
          ad_http_request = new XMLHttpRequest()
        else
          ad_http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        ad_http_request.onreadystatechange = function() 
        {
          handle_admin_request(ad_http_request)
        }

        ad_http_request.open("GET", url, true)
        ad_http_request.send(null)

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
        echo "        <td></td>\n";

      if ($this->config["OMSA_".$i] != "")
        echo "        <td><a href='https://".$this->config["OMSA_".$i]."/'>link</a></td>\n";
      else
        echo "        <td></td>\n";

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

    $cmd = "ssh ".$host." 'om_custom_report.csh'";
    $array=array();
    $last = exec($cmd, $array, $rval);
    $results = array();

    $string = "host:::".$host."\n";

    for ($i=0; $i<count($array); $i++) {
      $line = $array[$i];
      $arr = split(" ",$line, 2);
      $key = $arr[0];
      $val = $arr[1];
      $string .= $key.":::".$val."\n";
    }

    echo $string;
    flush();

  }

}
handledirect("admin");


