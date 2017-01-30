<?PHP

//
// Allows control of individual polarisations in the beam configuration
//


ini_set('display_errors',1);
error_reporting(E_ALL);

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

// size of the multibeam graphic
define ("CELL_HEIGHT", "50");
define ("CELL_WIDTH",  "100");

class beam_control extends bpsr_webpage 
{
  var $inst;
  var $beam = array();
  var $hosts = array();
  var $active = array();

  function beam_control()
  {
    bpsr_webpage::bpsr_webpage();

    $this->title = "BPSR | Beam Control";

    # every time we load page, or request update/action, this list should be updated
    $this->inst = new bpsr();
    $this->active =  $this->inst->configFileToHash(BEAMS_FILE);

    # get the host configuration of each ibob
    for ($i=0; $i<$this->inst->roach["NUM_ROACH"]; $i++)
    {
      $b = (int) $this->inst->roach["BEAM_".$i];
      $this->beams[$b] = $this->inst->roach["BEAM_".$i];
      $this->hosts[$b] = $this->inst->roach["ROACH_".$i];
    }
  }

  function javaScriptCallback()
  {
    return "bc_update_request();";
  }

  function printJavaScriptHead()
  {
?>
    <style>
      .notset_beam {
        background: lightgrey;
      }
    
      .active_beam {
        background: lightgreen;
      }

      .inactive_beam {
        background: grey;
      }

      table.multibeam {
        border-spacing: 10px;
      }

      table.multibeam td {
        padding: 5px;
        vertical-align: middle;
        text-align: center;
      }


    </style>

    <script type='text/javascript'>  

      var all_beams = Array("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13");
      var outer_beams = Array("02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13");
      var bc_xml_request;

      // handle the response from an beam_control request
      function handle_bc_xml_request( bc_xml_request) 
      {
        if ( bc_xml_request.readyState == 4)
        {
          var xmlDoc = bc_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;
            var beams = xmlObj.getElementsByTagName("beam");

            var beam_number = "";
            var pwc_id = "";
            var host = "";
            var state = "";
            var checkbox = "";
    
            for (i=0; i<beams.length; i++) 
            {
              node = beams[i];
              if (node.nodeType == 1)
              {
                beam_number = node.getAttribute("number");
                pwc_id = node.getAttribute("pwc_id");
                host = node.getAttribute("host");
                pol  = node.getAttribute("pol");
                state = node.childNodes[0].nodeValue;

                // set host
                document.getElementById(beam_number + "_host").innerHTML = host;

                checkbox = beam_number + "_p" + pol + "_active"; 

                // get state class element for this beam
                checked = document.getElementById(checkbox).checked;  
 
                if (state == "on") 
                {
                  if (checked != true)
                  {
                    document.getElementById(checkbox).checked = true;
                  }
                }
                else 
                {
                  if (checked == true)
                  {
                    document.getElementById(checkbox).checked = false;
                  }
                }
              }
            }
          }
        } 
      }

      // generate an observation info request
      function bc_update_request() 
      {
        var url = "beam_control.lib.php?update=true&host=na&port=na";

        if (window.XMLHttpRequest)
          bc_xml_request = new XMLHttpRequest();
        else
          bc_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        bc_xml_request.onreadystatechange = function() {
          handle_bc_xml_request ( bc_xml_request)
        };
        bc_xml_request.open("GET", url, true);
        bc_xml_request.send(null);
      }

      function bc_action_request() 
      {
        // get all the beams' checkbox states
        var url = "beam_control.lib.php?action=change";
        for (i=0; i<all_beams.length; i++)
        {
          beam = all_beams[i];
          cell = document.getElementById(beam+"_cell");

          state = document.getElementById(beam+"_p0_active").checked;
          if (state == true)
            url += "&"+beam+"_p0_active=on"
          else
            url += "&"+beam+"_p0_active=off"
          
          state = document.getElementById(beam+"_p1_active").checked;
          if (state == true)
            url += "&"+beam+"_p1_active=on"
          else
            url += "&"+beam+"_p1_active=off"

          state = document.getElementById(beam+"_pt_active").checked;
          if (state == true)
            url += "&"+beam+"_pt_active=on"
          else
            url += "&"+beam+"_pt_active=off"
        }

        if (window.XMLHttpRequest)
          bc_xml_request = new XMLHttpRequest();
        else
          bc_xml_request = new ActiveXObject("Microsoft.XMLHTTP");
  
        bc_xml_request.onreadystatechange = function() 
        {
          handle_bc_xml_request(bc_xml_request);
        }

        bc_xml_request.open("GET", url, true);
        bc_xml_request.send(null);
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("Beam Configuration");
?>
    <table border=0>
    <tr>
      <td width='<?echo (CELL_WIDTH*5+50)?>px'>

    <?
      echo "<table class='multibeam' border='0px'>\n";

      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(13);
        $this->echoBlank();
        $this->echoBeam(12);
        $this->echoBlank();
      echo "  </tr>\n";

      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank(); 
        $this->echoBeam(6);
        $this->echoBlank();
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(7);
        $this->echoBeam(5);
        $this->echoBlank();
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBeam(8);
        $this->echoBeam(1);
        $this->echoBeam(11);
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBeam(2);
        $this->echoBeam(4);
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(3);
        $this->echoBlank();
      echo "  </tr>\n";
      
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(9);
        $this->echoBeam(10);
        $this->echoBlank();
      echo "  </tr>\n";
      
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBlank();
        $this->echoBlank();
      echo "  </tr>\n";
      
      echo "  <tr>\n";
      echo "    <td colspan=5>\n";
      echo "      <input type='submit' value='Apply Changes' onClick='bc_action_request()'>\n";
      echo "    </td>\n";
      echo "  </tr>\n";
      
      echo "</table>\n";
    ?>
        </td>
        <td valign='top' width='350px'>
    <p>
      These controls will affect the level setting for each polarisation of each beam.
      A checked box means the level setting algorithm will try to set optimal levels
      for each pol. If the box is unchecked, then the levels for that pol will be set to
      0. This can be used to disable a bad polarisation.
    </p>
    <p>
      To use, check/uncheck polarsations and click Apply Changes
    </p>
    <p>
      <b>Any changes are persistent and will only affect future observations, the current observation will be unchanged.</b>
    </p>

    <p>
      <b>The Trans box check means that transient events from this beam will be plotted in the Transient Summary and searched for FRBs</b>
    </p>

        </td>
      </tr>
    </table>
<?
    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {
    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<beam_states num_pwc='".$this->inst->config["NUM_PWC"]."' num_roach='".$this->inst->roach["NUM_ROACH"]."'>\n";

    for ($i=0; $i<$this->inst->roach["NUM_ROACH"]; $i++)
    {
      $b = $this->inst->roach["BEAM_".$i];

      $state = "off";
      $beam = "BEAM_".$b."_p0";
      if (($i < $this->inst->config["NUM_PWC"]) && (array_key_exists($beam, $this->active)))
        $state = $this->active[$beam];
      $xml .= "<beam number='".$b."' pwc_id='".$i."' pol='0' host='".$this->hosts[(int)$b]."'>".$state."</beam>\n";

      $state = "off";
      $beam = "BEAM_".$b."_p1";
      if (($i < $this->inst->config["NUM_PWC"]) && (array_key_exists($beam, $this->active)))
        $state = $this->active[$beam];
      $xml .= "<beam number='".$b."' pwc_id='".$i."' pol='1' host='".$this->hosts[(int)$b]."'>".$state."</beam>\n";

      $state = "off";
      $beam = "BEAM_".$b."_pt";
      if (($i < $this->inst->config["NUM_PWC"]) && (array_key_exists($beam, $this->active)))
        $state = $this->active[$beam];
      $xml .= "<beam number='".$b."' pwc_id='".$i."' pol='t' host='".$this->hosts[(int)$b]."'>".$state."</beam>\n";
    }

    $xml .= "</beam_states>";
    header('Content-type: text/xml');
    echo $xml;
  }

  #
  # Handles the modification of the beam configuration
  #
  function printActionHTML($data) 
  {
    $xml = "";
    $tmp_file = BEAMS_FILE.".tmp";

    if (file_exists($tmp_file))
      unlink($tmp_file);

    # basically re-write the BEAMS_ACTIVE file, will take affect at start of next observation only
    $fptr = fopen($tmp_file, "w");
    if (!$fptr)
    {
      $xml  = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
      $xml .= "<beam_states>Error opening tmp file<beam_states>\n";;
    }

    fwrite($fptr, "# BPSR Active Beams File\n");
    fwrite($fptr, "# generated by beam_control.lib.php on ".date(DADA_TIME_FORMAT, time())."\n");
    for ($i=0; $i<$this->inst->roach["NUM_ROACH"]; $i++)
    {
      $b = $this->inst->roach["BEAM_".$i];
      $state = "on";
      $beam = $b."_p0_active";
      if (isset($data[$beam]) && ($data[$beam] == "off"))
        $state = $data[$beam];
      fwrite($fptr, instrument::headerFormat("BEAM_".$b."_p0", $state)."\n");

      $state = "on";
      $beam = $b."_p1_active";
      if (isset($data[$beam]) && ($data[$beam] == "off"))
        $state = $data[$beam];
      fwrite($fptr, instrument::headerFormat("BEAM_".$b."_p1", $state)."\n");

      $state = "on";
      $beam = $b."_pt_active";
      if (isset($data[$beam]) && ($data[$beam] == "off"))
        $state = $data[$beam];
      fwrite($fptr, instrument::headerFormat("BEAM_".$b."_pt", $state)."\n");
    }

    fclose($fptr);

    rename($tmp_file, BEAMS_FILE);

    //sleep(1);

    if ($xml == "")
    {
      # refresh the active array with new value(s)
      $this->active =  $this->inst->configFileToHash(BEAMS_FILE);

      # now print the updated values
      $this->printUpdateHTML($data);
    }
    else
    {
      header('Content-type: text/xml');
      echo $xml;
    }
  } 

  function echoBlank() 
  {
    echo "    <td width='".CELL_WIDTH."px' height='".CELL_HEIGHT."px'></td>\n";
  }

  function echoBeam($beam) 
  {
    $b = sprintf("%02d",$beam);
    $active = false;

    echo "    <td id='".$b."_cell' rowspan=2 class='notset_beam' width='".(CELL_HEIGHT*2)."px'>\n";
    echo "      <b>Beam ".$b."</b>\n";
    echo "      <span id='".$b."_host'></span><br/>\n";
    echo "      Pol 0: <input id='".$b."_p0_active' type='checkbox' name='".$b."_p0_active'".($active ? " checked" : "")."><br>\n";
    echo "      Pol 1: <input id='".$b."_p1_active' type='checkbox' name='".$b."_p1_active'".($active ? " checked" : "")."><br>\n";
    echo "      Trans: <input id='".$b."_pt_active' type='checkbox' name='".$b."_pt_active'".($active ? " checked" : "")."><br>\n";
    echo "    </td>\n";
  }

}

if (array_key_exists("update", $_GET) && ($_GET["update"] == "true"))
{
  $obj = new beam_control();
  $obj->printUpdateHTML($_GET);
}
else if (array_key_exists("action", $_GET) && ($_GET["action"] == "change"))
{
  $obj = new beam_control();
  $obj->printActionHTML($_GET);
}
else 
{
  handleDirect("beam_control");
}
