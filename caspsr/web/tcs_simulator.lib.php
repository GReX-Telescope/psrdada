<?PHP

include("caspsr_webpage.lib.php");
include("definitions_i.php");
include("functions_i.php");
include($instrument.".lib.php");

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

        i = document.getElementById("SOURCE_LIST").selectedIndex;
        psr = document.getElementById("SOURCE_LIST").options[i].value;

        i = document.getElementById("MODE_LIST").selectedIndex;
        mode = document.getElementById("MODE_LIST").options[i].value;
        document.getElementById("MODE").value = mode;

        if ((mode == "CAL") || (mode == "LEVCAL")) {
          document.getElementById("SOURCE").value = psr + "_R";
        } else {
          document.getElementById("SOURCE").value = psr;
        }

        document.tcs.submit();

      }

      function stopButton() {
        document.getElementById("COMMAND").value = "STOP";
        document.tcs.submit();
      }

      function updateRADEC() {
        var i = document.getElementById("SOURCE_LIST").selectedIndex;
        var psr = document.getElementById("SOURCE_LIST").options[i].value;
        var psr_ra = ras[psr];
        var psr_dec= decs[psr];
        document.getElementById("RA").value = psr_ra;
        document.getElementById("DEC").value = psr_dec;
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

        <td>CALFREQ</td>
        <td><input type="text" name="CALFREQ" size="12" value="11.123000"></td>

        <td width=20px>&nbsp;</td>

        <td>CFREQ</td>
        <td><input type="text" name="CFREQ" size="12" value="1382.00000"></td>

        <td width=20px>&nbsp;</td>

        <td>BANDWIDTH</td>
        <td><select name="BANDWIDTH">
              <option value="-400.0000">-400.0000</option>
              <option value="400.0000">400.0000</option>
              <option value="64">64</option>
            </select>

        <td width=20px>&nbsp;</td>

        <td>NDIM</td>
        <td><input type="text" name="NDIM" size="2" value="1" readonly></td>
      
      </tr>
      <tr>

        <td>SOURCE</td>
        <td>
          <input type="hidden" id="SOURCE" name="SOURCE" value="">
          <select id="SOURCE_LIST" name="SOURCE_LIST" onChange='updateRADEC()'>
<? 
          for ($i=0; $i<count($this->psr_keys); $i++) {
            echo "          <option value='".$this->psr_keys[$i]."'>".$this->psr_keys[$i]."</option>\n";
          }
?>
          </select>
        </td>

        <td width=20px>&nbsp;</td>

        <td>RA</td>
        <td><input type="text" id="RA" name="RA" size="12" value="04:37:00.00" readonly></td>

        <td width=20px>&nbsp;</td>

        <td>DEC</td>
        <td><input type="text" id="DEC" name="DEC" size="12" value="-47:35:00.0" readonly></td>

        <td width=20px>&nbsp;</td>

        <td>NBIT</td>
        <td><input type="text" name="NBIT" size="2" value="8" readonly></td>
      
      </tr>
      <tr>

        <td>RECEIVER</td>
        <td><input type="text" name="RECEIVER" size="12" value="MULTI" readonly></td>

        <td width=20px>&nbsp;</td>

        <td>PID</td>
        <td>
          <select name="PID">
<?          for ($i=0; $i<count($this->groups); $i++) {
              echo "            <option value=".$this->groups[$i].">".$this->groups[$i]."</option>\n";
            } 
?>
          </select>
        </td>

        <td width=20px>&nbsp;</td>

        <td>PROCFIL</td>
        <td>
          <select name="PROCFIL">
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

        <td width=20px>&nbsp;</td>

        <td>NPOL</td>
        <td><input type="text" name="NPOL" size="2" value="2" readonly></td>
      
      </tr>
      <tr>

        <td>MODE</td>
        <td>
          <input type="hidden" id="MODE" name="MODE" value="">
          <select id="MODE_LIST" name="MODE_LIST">
            <option value="PSR">PSR</option>
            <option value="CAL">CAL</option>
            <option value="LEVCAL">LEVCAL</option>
          </select>
        </td>

        <td width=20px>&nbsp;</td>

        <td>LENGTH</td>
        <td><input type="text" name="LENGTH" size="5" value=""> [s]</td>

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
      $result = rtrim(socketRead($sock));
      # $ignore = socketRead($sock);
      $this->printTR($cmd,$result);
      $this->printTF();
      $this->printFooter();
      return;
    }   

    # otherwise its a START command
    $keys = array_keys($get);
    for ($i=0; $i<count($keys); $i++) {

      $k = $keys[$i];
      if (($k != "COMMAND") && ($k != "SOURCE_LIST") && ($k != "MODE_LIST")) {
        if ($get[$k] != "") {
          $cmd = $k." ".$get[$k]."\r\n";
          socketWrite($sock,$cmd);
          $result = rtrim(socketRead($sock),"\r\n");
          $ignore = socketRead($sock);
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
    $result = rtrim(socketRead($sock),"\r\n");
    $this->printTR($cmd,$result);
    $ignore = socketRead($sock);
    if ($result != "ok") {
      $this->printTR("START command failed", $result.": ".rtrim(socketRead($sock)));
      $this->printTF();
      $this->printFooter();
      return;
    } else {
      $this->printTR("Sent START to nexus", "ok");
    }

    # now wait for the UTC_START to come back to us
    $result = rtrim(socketRead($sock));
    $this->printTR("", $result);
    $this->printTF();
    $this->printFooter();
    return;
  }

  function printTR($left, $right) {
    echo " <tr\n";
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



  function handleRequest()
  {

    if (isset($_GET["COMMAND"])) {
      $this->printTCSResponse($_GET);
    } else {
      $this->printHTML($_GET);
    }
  }

}
$obj = new tcs_simulator();
$obj->handleRequest($_GET);
