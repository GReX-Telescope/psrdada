<?PHP
include("caspsr_webpage.lib.php");
include("definitions_i.php");
include("functions_i.php");
include($instrument.".lib.php");

class rac_info extends caspsr_webpage
{

  var $dells = array();
  var $gpus = array();
  var $fields = array();
  var $rac_links = array();
  var $omsa_links = array();

  function rac_info()
  {
    caspsr_webpage::caspsr_webpage();

    $inst = new caspsr();
    $this->config = $inst->config;

    for ($i=0; $i<$this->config["NUM_PWC"]; $i++) 
    {
      array_push($this->gpus, $this->config["PWC_".$i]);
    }

    array_push($this->dells, "srv0");
    for ($i=0; $i<$this->config["NUM_DEMUX"]; $i++) 
    {
      array_push($this->dells, $this->config["DEMUX_".$i]);
    }
    $this->dells = array_unique($this->dells);

 
    $this->fields["service_tag"] = "Service Tag";
    $this->fields["bios"] = "BIOS";
    $this->fields["rac_fw"] = "RAC F/W";
    $this->fields["raid_fw"] = "RAID F/W";
    $this->fields["raid_drv"] = "RAID Drv";
    $this->fields["omsa_ver"] = "OMSA Ver";
    $this->fields["os"] = "O/S";
    $this->fields["disk0_id"] = "Disk0 ID";
    $this->fields["disk0_fw"] = "Disk0 F/W";
    $this->fields["disk1_id"] = "Disk1 ID";
    $this->fields["disk1_fw"] = "Disk2 F/W";

    $this->rac_links = array("demux0" => "http://192.168.0.32/", "demux1" => "http://192.168.0.33/", "gpu0" => "http://192.168.0.34", "gpu1" => "http://192.168.0.35", "gpu2" => "http://192.168.0.36/", "gpu3" => "http://192.168.0.37/");
    $this->omsa_links = array("srv0" => "https://srv0:1311/", "demux0" => "https://demux0:1311/", "demux1" => "https://demux1:1311/");


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

      function handle_rac_info_request(ri_http_request) 
      {
        if (ri_http_request.readyState == 4) {

          var response = String(ri_http_request.responseText)
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

      function rac_info_request(host, type) 
      {
        var ri_http_request;
        var url = "admin.lib.php?update=true&host="+host+"&type="+type;

        if (window.XMLHttpRequest)
          ri_http_request = new XMLHttpRequest()
        else
          ri_http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        ri_http_request.onreadystatechange = function() 
        {
          handle_rac_info_request(ri_http_request)
        }

        ri_http_request.open("GET", url, true)
        ri_http_request.send(null)

      }

      function get_rac_info()
      {
        rac_info_request("srv0", "dell");
        rac_info_request("demux0", "dell");
        rac_info_request("demux1", "dell");
        rac_info_request("gpu0", "dell");
        rac_info_request("gpu1", "dell");
        rac_info_request("gpu2", "dell");
        rac_info_request("gpu3", "dell");
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML()
  {

    $keys = array_keys($this->fields);

    $this->openBlockHeader("CASPSR Admin");
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

    for ($i=0; $i<count($this->dells); $i++) 
    {
      echo "      <tr>\n";
      echo "        <td>".$this->dells[$i]."</td>\n";
      if ($this->rac_links[$this->dells[$i]] != "")
        echo "        <td><a href='".$this->rac_links[$this->dells[$i]]."'>link</a></td>\n";
      else
        echo "        <td></td>\n";
      if ($this->omsa_links[$this->dells[$i]] != "")
        echo "        <td><a href='".$this->omsa_links[$this->dells[$i]]."'>link</a></td>\n";
      else
        echo "        <td></td>\n";

      for ($j=0; $j<count($keys); $j++) 
      {
        echo "        <td id='".$this->dells[$i]."_".$keys[$j]."'></td>\n";
      }
      echo"       </tr>\n";
    }

    for ($i=0; $i<count($this->gpus); $i++) 
    {
      echo "      <tr>\n";
      echo "        <td>".$this->gpus[$i]."</td>\n";
      if ($this->rac_links[$this->gpus[$i]] != "")
        echo "        <td><a href='".$this->rac_links[$this->gpus[$i]]."'>link</a></td>\n";
      else
        echo "        <td></td>\n";
      if ($this->omsa_links[$this->gpus[$i]] != "")
        echo "        <td><a href='".$this->omsa_links[$this->gpus[$i]]."'>link</a></td>\n";
      else
        echo "        <td></td>\n";

      for ($j=0; $j<count($keys); $j++) 
      {
        echo "        <td id='".$this->gpus[$i]."_".$keys[$j]."'></td>\n";
      }
      echo"       </tr>\n";
    }
?>
    </table>
    <script type='text/javascript'>
      get_rac_info();
    </script>
<?
    $this->closeBlockHeader();
    
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];

    $cmd = "ssh ".$host." '/usr/local/bin/om_custom_report.csh'";
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
handledirect("rac_info");

