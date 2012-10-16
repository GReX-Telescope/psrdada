<?PHP

ini_set('display_errors',1);
error_reporting(E_ALL);

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

// size of the multibeam graphic
define ("CELL_HEIGHT", "50");
define ("CELL_WIDTH",  "100");

class beam_config extends bpsr_webpage 
{
  var $inst;
  var $beam = array();
  var $hosts = array();
  var $active = array();

  function beam_config()
  {
    bpsr_webpage::bpsr_webpage();

    $this->title = "BPSR | Beam Configuration";

    # every time we load page, or request update/action, this list should be updatede
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

      // check/uncheck the outer beams
      function checkOuter(val) {
        for (i=0; i<outer_beams.length; i++) {
          document.getElementById(outer_beams[i]+"_active").checked = val;
        }
      }

      // handle the response from an beam_config request
      function handle_bc_xml_request( bc_xml_request) 
      {
        if ( bc_xml_request.readyState == 4)
        {
          var xmlDoc = bc_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;
            //var beam_states = xmlObj.getElementsByTagName("beam_states")[0].childNodes[0].nodeValue;
            var beams = xmlObj.getElementsByTagName("beam");

            var beam_number = "";
            var pwc_id = "";
            var host = "";
            var state = "";
    
            for (i=0; i<beams.length; i++) 
            {
              node = beams[i];
              if (node.nodeType == 1)
              {
                beam_number = node.getAttribute("number");
                pwc_id = node.getAttribute("pwc_id");
                host = node.getAttribute("host");
                state = node.childNodes[0].nodeValue;

                // set host
                document.getElementById(beam_number + "_host").innerHTML = host;

                // get state class element for this beam
                current_state = document.getElementById(beam_number + "_cell").className;  
 
                if (state == "on") 
                {
                  document.getElementById(beam_number + "_cell").className = "active_beam";
                  if (current_state != "active_beam") 
                    document.getElementById(beam_number + "_active").checked = true;
                }
                else 
                {
                  document.getElementById(beam_number + "_cell").className = "inactive_beam";
                  if (current_state != "inactive_beam")
                    document.getElementById(beam_number + "_active").checked = false;
                }
              }
            }
          }
        } 
      }

      // generate an obsevartaion info request
      function bc_update_request() 
      {
        var url = "beam_config.lib.php?update=true&host=na&port=na";

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
        var url = "beam_config.lib.php?action=change";
        for (i=0; i<all_beams.length; i++)
        {
          beam = all_beams[i];
          cell = document.getElementById(beam+"_cell");
          state = document.getElementById(beam+"_active").checked;
          if (state == true)
            url += "&"+beam+"_active=on"
          else
            url += "&"+beam+"_active=off"
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
        $this->echoBeam(13, $this->active["BEAM_13"], $this->hosts[13]);
        $this->echoBlank();
        $this->echoBeam(12, $this->active["BEAM_12"], $this->hosts[12]);
        $this->echoBlank();
      echo "  </tr>\n";

      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank(); 
        $this->echoBeam(6, $this->active["BEAM_06"], $this->hosts[6]);
        $this->echoBlank();
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(7, $this->active["BEAM_07"], $this->hosts[7]);
        $this->echoBeam(5, $this->active["BEAM_05"], $this->hosts[5]);
        $this->echoBlank();
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBeam(8, $this->active["BEAM_08"], $this->hosts[8]);
        $this->echoBeam(1, $this->active["BEAM_01"], $this->hosts[1]);
        $this->echoBeam(11, $this->active["BEAM_11"], $this->hosts[11]);
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBeam(2, $this->active["BEAM_02"], $this->hosts[2]);
        $this->echoBeam(4, $this->active["BEAM_04"], $this->hosts[4]);
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(3, $this->active["BEAM_03"], $this->hosts[3]);
        $this->echoBlank();
      echo "  </tr>\n";
      
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(9, $this->active["BEAM_09"], $this->hosts[9]);
        $this->echoBeam(10, $this->active["BEAM_10"], $this->hosts[10]);
        $this->echoBlank();
      echo "  </tr>\n";
      
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBlank();
        $this->echoBlank();
      echo "  </tr>\n";
      
      echo "  <tr>\n";
      echo "    <td colspan=5>\n";
      echo "      <input type='button' value='Check Outer' onClick='checkOuter(true)'>\n";
      echo "      <input type='button' value='Uncheck Outer' onClick='checkOuter(false)'>\n";
      echo "      <input type='submit' value='Apply Changes' onClick='bc_action_request()'>\n";
      echo "    </td>\n";
      echo "  </tr>\n";
      
      echo "</table>\n";
    ?>
        </td>
        <td valign='top' width='350px'>
    <p>The ACTIVE beams are coloured green, to change:
    <ul>
      <li>Check all the boxes of the beams to wish to have active</li>
      <li>Click APPPLY CHANGES</li>
      <li>After a few seconds the colours of the relevant beams will change to GREEN or GREY</li>
    </ul>
    <b>Any changes will only affect future observations, the current observation will be unchanged.</b>
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
      if (($i < $this->inst->config["NUM_PWC"]) && (array_key_exists("BEAM_".$b, $this->active)))
        $state = $this->active["BEAM_".$b];

      $xml .= "<beam number='".$b."' pwc_id='".$i."' host='".$this->hosts[(int)$b]."'>".$state."</beam>\n";
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
    fwrite($fptr, "# generated by beam_config.lib.php on ".date(DADA_TIME_FORMAT, time())."\n");
    for ($i=0; $i<$this->inst->roach["NUM_ROACH"]; $i++)
    {
      $b = $this->inst->roach["BEAM_".$i];
      $state = "on";
      if (isset($data[$b."_active"]) && ($data[$b."_active"] == "off"))
        $state = $data[$b."_active"];
      fwrite($fptr, instrument::headerFormat("BEAM_".$b, $state)."\n");
    }

    fclose($fptr);

    rename($tmp_file, BEAMS_FILE);

    sleep(1);

    if ($xml == "")
    {
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

    echo "    <td id='".$b."_cell' rowspan=2 class='notset_beam' width='".(CELL_HEIGHT*2)."px' id>\n";
    echo "      <b>Beam ".$b."</b>\n";
    echo "      <input id='".$b."_active' type='checkbox' name='".$b."_active'".($active ? " checked" : "")."><br>\n";
    echo "      <span id='".$b."_host'></span>\n";
    echo "    </td>\n";
  }

}

if (array_key_exists("update", $_GET) && ($_GET["update"] == "true"))
{
  $obj = new beam_config();
  $obj->printUpdateHTML($_GET);
}
else if (array_key_exists("action", $_GET) && ($_GET["action"] == "change"))
{
  $obj = new beam_config();
  $obj->printActionHTML($_GET);
}
else 
{
  handleDirect("beam_config");
}
