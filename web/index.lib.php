<?PHP

include("dada_webpage.lib.php");
include("instrument.lib.php");

define ("DADA_CFG", "/home/dada/linux_64/share/dada.cfg");

class index extends dada_webpage 
{
  var $config = array();

  function index()
  {
    dada_webpage::dada_webpage();
    if (file_exists(DADA_CFG))
    {
      $this->config = instrument::configFileToHash(DADA_CFG);
    }
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("PSRDADA : Configured Backends");

    if (array_key_exists("BACKENDS", $this->config)) 
    {
      $backends = explode(" ", $this->config["BACKENDS"]);

      echo "<table>\n";
      for ($i=0; $i<count($backends); $i++)
      { 
        $url = "http://".$_SERVER["HTTP_HOST"]."/".$backends[$i]."/";
        $logo = "http://".$_SERVER["HTTP_HOST"]."/".$backends[$i]."/images/".$backends[$i]."_logo.png";
  
        echo "  <tr>\n";
        echo "    <td>";
        echo "<a style='border: none;' href='".$url."'><img border=0; src='".$logo."'></a>\n";
        echo "    </td>\n";
        echo "  </tr>\n";
      }
      echo "</table>\n";
    }
    else 
    {
      echo "Please create the ".DADA_CFG." file with the required backend specs<BR>\n";
    }
  

    $this->closeBlockHeader();

    echo "<br><br>\n";

    $this->openBlockHeader("PSRDADA : Management / Administration");
    
    echo "<p><a href='http://".$_SERVER["HTTP_HOST"]."/ganglia/'>Ganglia</a></p>\n";

    echo "<p><a href='http://psrdada.sf.net/support.shtml'>PSRDADA Support</a></p>\n";

    echo "<p><a href='admin.lib.php?single=true'>RAC, OMSA and Firmware Information</a></p>\n";

    echo "<p><a href='/nagios/'>NAGIOS Cluster Monitor</a></p>\n";

    echo "<p><a href='http://www.parkes.atnf.csiro.au/cgi-bin/monitoring/weather_mon.cgi'>Parkes Weather Monitoring</a></p>\n";

    $this->closeBlockHeader();

  }

}

handleDirect("index");

