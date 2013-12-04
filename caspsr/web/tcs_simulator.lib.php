<?PHP

include_once("caspsr_webpage.lib.php");
include_once("definitions_i.php");
include_once("functions_i.php");
include_once($instrument.".lib.php");

class tcs_simulator extends caspsr_webpage 
{

  var $groups = array();
  var $psrs = array();
  var $psr_keys = array();
  var $inst = 0;

  function tcs_simulator()
  {
    caspsr_webpage::caspsr_webpage();

    array_push($this->css, "/caspsr/buttons.css");
    $this->inst = new caspsr();
    $this->groups = getProjects("caspsr");
  }

  function printJavaScriptHead()
  {

    $this->psrs = $this->inst->getPsrcatPsrs();
    $this->psr_keys = array_keys($this->psrs);
    array_push ($this->psr_keys, "J1705-1908");
    $this->psrs["J1705-1908"]["RAJ"] = "17:05:27.1399999999905";
    $this->psrs["J1705-1908"]["DECJ"] = "-19:08:02.5899999999903";

?>
    <script type='text/javascript'>
<?
      echo "      var ras = {";
      for ($i=0; $i<count($this->psr_keys); $i++) {
        $psr = $this->psr_keys[$i];
        $raj = $this->psrs[$psr]["RAJ"];
        if ($i != 0)
          echo ",";
        echo "'".$psr."':'".$raj."'";
      }
      echo "};\n";

      echo "      var decs = {";
       for ($i=0; $i<count($this->psr_keys); $i++) {
        $psr = $this->psr_keys[$i];
        $decj = $this->psrs[$psr]["DECJ"];
        if ($i != 0)  
          echo ",";
        echo "'".$psr."':'".$decj."'";
      }
      echo "};\n";
?>

      function startButton() {

        document.getElementById("COMMAND").value = "START";

        var i = 0;
        var psr = "";
        var mode = "";

        updateRADEC();

        i = document.getElementById("src_list").selectedIndex;
        psr = document.getElementById("src_list").options[i].value;

        i = document.getElementById("mode_list").selectedIndex;
        mode = document.getElementById("mode_list").options[i].value;

        if ((mode == "CAL") || (mode == "LEVCAL")) {
          document.getElementById("src").value = psr + "_R";
        } else {
          document.getElementById("src").value = psr;
        }

        document.tcs.submit();

      }

      function stopButton() {
        document.getElementById("COMMAND").value = "STOP";
        document.tcs.submit();
      }

      function updateRADEC() {
        var i = document.getElementById("src_list").selectedIndex;
        var psr = document.getElementById("src_list").options[i].value;
        var psr_ra = ras[psr];
        var psr_dec= decs[psr];
        document.getElementById("ra").value = psr_ra;
        document.getElementById("dec").value = psr_dec;
      }
    </script>

<?
  }

  /*************************************************************************************************** 
   *
   * HTML for this page 
   *
   ***************************************************************************************************/
  function printHTML()
  {
?>
<html>
<head>
  <title>CASPSR | TCS Simulator</title>
  <link rel='shortcut icon' href='/caspsr/images/caspsr_favicon.ico'/>
<?
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
    for ($i=0; $i<count($this->ejs); $i++)
      echo "   <script language='javascript' type='text/javascript' src='".$this->ejs[$i]."'></script>\n";

    $this->printJavaScriptHead();
?>
</head>

<body>
<?
  $this->printJavaScriptBody();
?>
  <div class='PageBackgroundSimpleGradient'>
  </div>
  <div class='Main'>
      <div style='text-align: left; vertical-align: middle; padding-left: 10px'>
        <img src="/caspsr/images/caspsr_logo_200x60.png" width=200 height=60>
      </div>
<?
        $this->printMainHTML();
?>
  </div>
</body>
</html>
<?
  }


  /* HTML for this page */
  function printMainHTML() 
  {
    $this->openBlockHeader("TCS Simulator");
?>
    <form name="tcs" target="tcs_interface" method="GET">
    <table border=0 cellpadding=5 cellspacing=0 width='100%'>
      <tr>

        <td>band</td>
        <td>
          <select name="band">
            <option value="-400">-400</option>
            <option value="400">400</option>
            <option value="64">64</option>
          </select>
        </td>

        <td width=20px>&nbsp;</td>

        <td>observer</td>
        <td><input type="text" name="observer" size="4" value="test"></td>
      
        <td width=20px>&nbsp;</td>

        <td>nbeam</td>
        <td><input type="text" name="nbeam" size="2" value="1"></td>

      </tr>
      <tr>

        <td>src</td>
        <td>
          <input type="hidden" id="src" name="src" value="">
          <select id="src_list" name="src_list" onChange='updateRADEC()'>
<? 
          for ($i=0; $i<count($this->psr_keys); $i++) {
            echo "          <option value='".$this->psr_keys[$i]."'>".$this->psr_keys[$i]."</option>\n";
          }
?>
          </select>
        </td>

        <td width=20px>&nbsp;</td>

        <td>ra</td>
        <td><input type="text" id="ra" name="ra" size="12" value="04:37:00.00" readonly></td>

        <td width=20px>&nbsp;</td>

        <td>dec</td>
        <td><input type="text" id="dec" name="dec" size="12" value="-47:35:00.0" readonly></td>

      </tr>
      <tr>

        <td>receiver</td>
        <td>
          <select name="receiver">
            <option value="MULTI">MULTI</option>
            <option value="1050CM">1050CM</option>
          </select>
        </td>

        <td width=20px>&nbsp;</td>

        <td>pid</td>
        <td>
          <select name="pid">
<?          for ($i=0; $i<count($this->groups); $i++) {
              echo "            <option value=".$this->groups[$i].">".$this->groups[$i]."</option>\n";
            } 
?>
          </select>
        </td>

        <td width=20px>&nbsp;</td>

        <td>procfil</td>
        <td>
          <select name="procfil">
            <option value="dspsr.gpu">dspsr.gpu</option>
            <option value="dspsr.50cmgpu">dspsr.50cmgpu</option>
            <option value="dspsr.nosk">dspsr.nosk</option>
            <option value="dspsr.single.gpu">dspsr.single.gpu</option>
            <option value="dspsr.cpu">dspsr.cpu</option>
            <option value="dspsr.skfb">dspsr.skfb</option>
            <option value="dspsr.aj">dspsr.aj</option>
            <option value="counter.test">counter.test</option>
            <option value="dbnull.caspsr">dbnull.caspsr</option>
            <option value="dbdisk.caspsr">dbdisk.caspsr</option>
            <option value="caspsr.20cmdbib">caspsr.20cmdbib</option>
            <option value="caspsr.50cmdbib">caspsr.50cmdbib</option>
          </select>
        </td>

      </tr>
      <tr>

        <td>obsval</td>
        <td><input type="text" name="obsval" size="16" value="120.000"> [s]</td>

        <td>&nbsp;</td>
      
        <td>obsunit</td>
        <td><input type="text" name="obsunit" size="16" value="SECONDS"></td>

        <td>&nbsp;</td>

        <td>refbeam</td>
        <td><input type="text" name="refbeam" size="2" value="1"></td>
      
      </tr>

      <tr>
        <td>mode</td>
        <td>
          <select id="mode_list" name="mode_list">
            <option value="PSR" selected>PSR</option>
            <option value="CAL">CAL</option>
            <option value="LEVCAL">LEVCAL</option>
          </select>

        <td>&nbsp;</td>

        <td>freq</td>
        <td><input type="text" name="freq" size="16" value="1382.000000"></td>



        <td colspan=6>&nbsp;</td>
      </tr>

      <tr>
        <td colspan=11><hr></td>
      </tr>
      
      <tr>
        <td colspan=11>
          <div class="btns" style='text-align: center'>
            <a href="javascript:startButton()"  class="btn" > <span>Start</span> </a>
            <a href="javascript:stopButton()"  class="btn" > <span>Stop</span> </a>
          </div>
        </td>
    </table>
    <input type="hidden" id="COMMAND" name="COMMAND" value="">
    </form>
<?
    $this->closeBlockHeader();

    $this->openBlockHeader("TCS Interface");
?>
    <iframe name="tcs_interface" src="" width=100% frameborder=0 height='350px'></iframe>
<?
    $this->closeBlockHeader();
  }

  function printTCSResponse($get)
  {

    // Open a connection to the TCS interface script
    $host = $this->inst->config["TCS_INTERFACE_HOST"];
    $port = $this->inst->config["TCS_INTERFACE_PORT"];
    $sock = 0;

    echo "<html>\n";
    echo "<head>\n";
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
?>
</head>
<body>
<div class="BlockContent-body">
  <div>
<table border=0>
 <tr>
  <th>Command</th>
  <th>Response</th>
 </tr>
<?
    list ($sock,$message) = openSocket($host,$port,2);
    if (!($sock)) {
      $this->printTR("Error: opening socket to TCS interface [".$host.":".$port."]: ".$message, "");
      $this->printTF();
      $this->printFooter();
      return;
    }

    # if we have a STOP command try and stop the tcs interface
    if ($get["COMMAND"] == "STOP") {
      $cmd = "STOP\r\n";
      socketWrite($sock,$cmd);
      $result = rtrim($this->sRead($sock));
      # $ignore = $this->sRead($sock);
      $this->printTR($cmd,$result);
      $this->printTF();
      $this->printFooter();
      return;
    }   

    # otherwise its a START command
    $keys = array_keys($get);
    for ($i=0; $i<count($keys); $i++) {

      $k = $keys[$i];
      if (($k != "COMMAND") && ($k != "src_list") && ($k != "mode_list")) {
        if ($get[$k] != "") {
          $cmd = $k." ".$get[$k]."\r\n";
          socketWrite($sock,$cmd);
          $result = rtrim($this->sRead($sock),"\r\n");
          $ignore = $this->sRead($sock);
          if ($result != "ok") {
            $this->printTR("[".$cmd."] failed ", $result);
            $this->printTR("START aborted", "");
            return;
          }
          $this->printTR($cmd,$result);
        } else {
          $this->printTR($k, "Ignoring as value was empty");
        }
      }
    }

    # Issue START command to server_tcs_interface 
    $cmd = "START\r\n";
    socketWrite($sock,$cmd);
    $result = rtrim($this->sRead($sock),"\r\n");
    $this->printTR($cmd,$result);
    $ignore = $this->sRead($sock);
    if ($result != "ok") {
      $this->printTR("START command failed", $result.": ".rtrim($this->sRead($sock)));
      $this->printTF();
      $this->printFooter();
      return;
    } else {
      $this->printTR("Sent START to nexus", "ok");
    }

    # now wait for the UTC_START to come back to us
    $result = rtrim($this->sRead($sock));
    $this->printTR("", $result);
    $this->printTF();
    $this->printFooter();
    return;
  }

  function printTR($left, $right) {
    echo " <tr>\n";
    echo "  <td>".$left."</td>\n";
    echo "  <td>".$right."</td>\n";
    echo " </tr>\n";
    echo '<script type="text/javascript">self.scrollBy(0,100);</script>';
    flush();
  }

  function printFooter() {
    echo "  </div>\n";
    echo "</div>\n";
    echo "</body>\n";
    echo "</html>\n";
  }

  function printTF() {
    echo "</table>\n";
  }

  function getRAForSource($source)
  {
    $cmd = "psrcat -all -c \"PSRJ RAJ DECJ\" ".$source." -nohead | grep -v \"*             *\" | awk '{print $2,$4, $7}'";
    $script = "source /home/dada/.bashrc; ".$cmd." 2>&1";
    $string = exec($script, $output, $return_var);

    $ra = "00:00:00.000";
    if ($rval == 0 && count($output) == 1 && strpos("WARNING", $output[0]) !== FALSE)
    {
      $bits = split(" ", $output[0]);
      $ra = $bits[1];
    }
    return $ra;
  }

  function getDECForSource($source)
    {
    $cmd = "psrcat -all -c \"PSRJ RAJ DECJ\" ".$source." -nohead | grep -v \"*             *\" | awk '{print $2,$4, $7}'";
    $script = "source /home/dada/.bashrc; ".$cmd." 2>&1";
    $string = exec($script, $output, $return_var);

    $dec = "00:00:00.000";
    if ($rval == 0 && count($output) == 1 && strpos("WARNING", $output[0]) !== FALSE)
    {
      $bits = split(" ", $output[0]);
      $dec = $bits[2];
    }
    return $dec;
  }


  function handleRequest()
  {
    $action = isset($_GET["action"]) ? $_GET["action"] : "";
    $source = isset($_GET["source"]) ? $_GET["source"] : "";
    if (($action == "get_ra") && ($source != ""))
      echo $this->getRAForSource($source);
    else if (($action == "get_dec") && ($source != ""))
      echo $this->getDECForSource($source);
    else if (isset($_GET["COMMAND"]))
      $this->printTCSResponse($_GET);
    else
      $this->printHTML($_GET);
  }

  function sRead($handle)
  {
    list ($result, $response) = socketRead($handle);
    return $response;  
  }
  

}
$obj = new tcs_simulator();
$obj->handleRequest($_GET);
