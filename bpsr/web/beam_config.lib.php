<?PHP

include("bpsr.lib.php");
include("bpsr_webpage.lib.php");

// size of the multibeam graphic
define (CELL_HEIGHT, 50);
define (CELL_WIDTH,  100);

class beam_config extends bpsr_webpage 
{

  function beam_config()
  {
    bpsr_webpage::bpsr_webpage();
  }

  function javaScriptCallback()
  {
    return "beam_config_request();";
  }

  function printJavaScriptHead()
  {
    $inst = new bpsr();
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

      var outer_beams = Array("02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13");

      // check/uncheck the outer beams
      function checkOuter(val) {
        for (i=0; i<outer_beams.length; i++) {
          document.getElementById(outer_beams[i]+"_active").checked = val;
        }
      }

      // handle the response from an beam_config request
      function handle_beam_config_request( bc_http_request) 
      {
        if ( bc_http_request.readyState == 4) {
          var response = String(bc_http_request.responseText)

          if (response.indexOf("Could not connect to") == -1) 
          {

            var lines = response.split("\n");
            var bits;

            for (i=0; i<lines.length; i++) 
            {
              bits = lines[i].split(":::");
              if (bits[0]) 
              {
                beam = bits[0];
                host = bits[1];
                active = bits[2];

                document.getElementById(beam+"_host").innerHTML = host;
                current_state = document.getElementById(beam+"_cell").className;  
 
                if (active == "1") 
                {
                  document.getElementById(beam+"_cell").className = "active_beam";
                  if (current_state != "active_beam") 
                    document.getElementById(beam+"_active").checked = true;
                }
                else 
                {
                  document.getElementById(beam+"_cell").className = "inactive_beam";
                  if (current_state != "inactive_beam")
                    document.getElementById(beam+"_active").checked = false;
                }
              }
            }
          }
        }
      }

      // generate an obsevartaion info request
      function beam_config_request() 
      {
        var url = "beam_config.lib.php?update=true&host=na&port=na";

        if (window.XMLHttpRequest)
          bc_http_request = new XMLHttpRequest();
        else
          bc_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        bc_http_request.onreadystatechange = function() {
          handle_beam_config_request( bc_http_request)
        };
        bc_http_request.open("GET", url, true);
        bc_http_request.send(null);
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

    <form name='change_beams' action='beam_config.lib.php' target='beam_config_frame' method='post'>
    <?

      echo "<table class='multibeam' border='0px'>\n";

      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(13, $active[13], $hosts[13]);
        $this->echoBlank();
        $this->echoBeam(12, $active[12], $hosts[12]);
        $this->echoBlank();
      echo "  </tr>\n";

      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank(); 
        $this->echoBeam(6, $active[6], $hosts[6]);
        $this->echoBlank();
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(7, $active[7], $hosts[7]);
        $this->echoBeam(5, $active[5], $hosts[5]);
        $this->echoBlank();
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBeam(8, $active[8], $hosts[8]);
        $this->echoBeam(1, $active[1], $hosts[1]);
        $this->echoBeam(11, $active[11], $hosts[11]);
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBeam(2, $active[2], $hosts[2]);
        $this->echoBeam(4, $active[4], $hosts[4]);
      echo "  </tr>\n";
        
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(3, $active[3], $hosts[3]);
        $this->echoBlank();
      echo "  </tr>\n";
      
      echo "  <tr height=".CELL_HEIGHT.">\n";
        $this->echoBlank();
        $this->echoBeam(9, $active[9], $hosts[9]);
        $this->echoBeam(10, $active[10], $hosts[10]);
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
      echo "      <input type='submit' value='Apply Changes'>\n";
      echo "    </td>\n";
      echo "  </tr>\n";
      
      echo "  <tr>\n";
      echo "    <td colspan=5>\n";
    ?>
    <p>Currently active beams are colored green. To change, use the checkbox. 
    Note that you can only apply changes if BPSR is currently in an idle state,
    since making beam changes requires some of the daemons to be reloaded.
    This will be done automatically if you click Apply changes.</p>
    <?
      echo "    </td>\n";
      echo "  </tr>\n";
      echo "</table>\n";
    ?>
            <input type='hidden' name='action' value='change'>
          </form>
        </td>
        <td align='left' style='vertical-align: top;'>
          <iframe name='beam_config_frame' width='400px' height='500px' src='beam_reconfig.php' background='white'></iframe>
        </td>
      </tr>
    </table>


<?
    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {

    # force a re-read of the bpsr configuration
    $inst = new bpsr();

    $beams = array();
    $hosts = array();
    $active = array();

    # get the host configuration of each ibob
    for ($i=0; $i<$inst->ibobs["NUM_IBOB"]; $i++)
    {
      $beams[$i] = $inst->ibobs["BEAM_".$i];
      $hosts[$i] = $inst->ibobs["10GbE_CABLE_".$i];
    }

    for ($i=0; $i<$inst->ibobs["NUM_IBOB"]; $i++)
    {
      $is_active = false;
      for ($j=0; $j<$inst->config["NUM_PWC"]; $j++)
      {
        if ($inst->ibobs["10GbE_CABLE_".$i] == $inst->config["PWC_".$j])
        {
          $is_active = true;
        }
      }
      $active[$i] = $is_active;
    }

    $string = "";
    for ($i=0; $i<$inst->ibobs["NUM_IBOB"]; $i++)
    {
      $string .= $beams[$i].":::".$hosts[$i].":::".$active[$i]."\n";
    }

    echo rtrim($string);
  }

  #
  # Handles the modification of the beam configuration
  #
  function printActionHTML() 
  {
    $inst = new bpsr();

    # get the list of configurable ibobs
    $ibobs = array();
    $hosts = array();

    for ($i=0; $i<$inst->ibobs["NUM_IBOB"]; $i++)
    {
      $ibobs[$i] = $inst->ibobs["BEAM_".$i];
      $hosts[$i] = $inst->ibobs["10GbE_CABLE_".$i];
    }

    $j = 0;
    $new_pwcs = array();
    for ($i=0; $i<count($ibobs); $i++)
    {
      if (isset($_POST[$ibobs[$i]."_active"]) && ($_POST[$ibobs[$i]."_active"] == "on"))
      {
        $new_pwcs[$j] = $hosts[$i];
        $j++;
      }
    }
    $num_pwcs = count($new_pwcs);

    header("Cache-Control: no-cache, must-revalidate"); // HTTP/1.1
    header("Expires: Mon, 26 Jul 1997 05:00:00 GMT");   // Date in the past

    echo "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n";
    echo "<html xmlns=\"http://www.w3.org/1999/xhtml\">\n";
    echo "<body>\n";
    echo "<pre>\n";
    # Stop BPSR backend

    $script = "source /home/dada/.bashrc; bpsr_reconfigure.pl -s 2>&1";
    passthru($script, $rval);
    if ($rval != 0) {
      exit;
    }

    echo "Adjust BPSR configuration\n";
    # adjust the bpsr_pwcs.cfg file
    $fptr = fopen(PWC_FILE, "w");
    if (!$fptr) {
      echo "ERROR: could not open PWC config file [".PWC_FILE."] for writing<BR>\n";
      exit;
    }

    fwrite($fptr, "# BPSR PWC Config File\n");
    fwrite($fptr, "# generated by bpsr_config.lib.php on ".date(DADA_TIME_FORMAT, time())."<BR>\n");
    fwrite($fptr, instrument::headerFormat("NUM_PWC", count($new_pwcs))."\n");

    for ($i=0; $i<count($new_pwcs); $i++) {
      fwrite($fptr, instrument::headerFormat("PWC_".$i, $new_pwcs[$i])."\n");
    }

    fclose($fptr);

    # Force a sleep here to ensure all socket connections are properly closed
    sleep(10);

    # Start BPSR backend
    $script = "source /home/dada/.bashrc; bpsr_reconfigure.pl -i 2>&1";
    system($script, $rval);
    if ($rval != 0) {
      exit;
    }


    echo "Done!\n";
    echo "</pre>\n";
    echo "</body>\n";
    echo "</html>\n";

    #echo "<script type='text/javascript'>\n";
    #echo "  parent.document.location='beam_config.php'\n";
    #echo "</script>\n";

  } 

  function echoBlank() 
  {
    echo "    <td width='".CELL_WIDTH."px' height='".CELL_HEIGHT."px'></td>\n";
  }

  function echoBeam($beam) 
  {
    $b = sprintf("%02d",$beam);

    echo "    <td id='".$b."_cell' rowspan=2 class='notset_beam' width='".(CELL_HEIGHT*2)."px' id>\n";
    echo "      <b>Beam ".$b."</b>\n";
    echo "      <input id='".$b."_active' type='checkbox' name='".$b."_active'".($active ? " checked" : "")."><br>\n";
    echo "      <span id='".$b."_host'></span>\n";
    echo "    </td>\n";
  }

}

if ($_GET["update"] == "true")
{
  $obj = new beam_config();
  $obj->printUpdateHTML($_GET);
}
else if ($_POST["action"] == "change") 
{
  $obj = new beam_config();
  $obj->printActionHTML();
}
else 
{
  handleDirect("beam_config");
}
