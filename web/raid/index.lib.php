<?PHP

include("../dada_webpage.lib.php");
include("../instrument.lib.php");

define (DADA_CFG, "/home/dada/linux_64/share/dada.cfg");

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
    $this->config["BACKENDS"] = "apsr bpsr";
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("PSRDADA : Configured Archival Pipelines");

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

  }

}

handleDirect("index");

