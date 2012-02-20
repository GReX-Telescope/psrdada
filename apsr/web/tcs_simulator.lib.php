<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

$_GET["single"] = "false";
$_GET["show_buttons"] = "false";
include("state_banner.lib.php");
$_GET["single"] = "true";

class tcs_simulator extends apsr_webpage 
{

  var $groups = array();
  var $psrs = array();
  var $psr_keys = array();
  var $inst = 0;
  var $pdfb_configs = array("pdfb3_1024_1024_1024", "pdfb3_1024_256_1024", "pdfb3_1024_256_2048", "pdfb3_1024_256_512", "pdfb3_1024_64_1024", "pdfb3_128_64_1024", "pdfb3_128_64_512", "pdfb3_2048_1024_1024", "pdfb3_2048_1024_2048", "pdfb3_2048_256_1024", "pdfb3_256_256_1024", "pdfb3_256_256_2048", "pdfb3_256_64_1024", "pdfb3_256_64_512", "pdfb3_512_1024_1024", "pdfb3_512_1024_2048", "pdfb3_512_256_1024", "pdfb3_512_256_2048", "pdfb3_512_64_1024", "pdfb3_512_64_512", "pdfb4_1024_1024_1024", "pdfb4_1024_256_1024", "pdfb4_256_1024_1024", "pdfb4_512_1024_1024", "pdfb4_512_256_512");


  function tcs_simulator()
  {
    apsr_webpage::apsr_webpage();

    $this->title = "APSR | TCS Simulator";
    $this->inst = new apsr();
    $this->groups = $this->inst->getPIDS();
  
    $this->state_banner = new state_banner();
  }

  function javaScriptCallback()
  {
    return $this->state_banner->javaScriptCallback();
  }

  function printJavaScriptHead()
  {

    $this->psrs = $this->inst->getPsrcatPsrs();
    $this->psr_keys = array_keys($this->psrs);

?>
    <style type='text/css'>

      td.key {
        text-align: right;
      }
 
      td.val {
        padding-right: 20px;
        text-align: left;
      } 
    </style>

    <script type='text/javascript'>

      var ras = new Array();
      var decs = new Array();
<?
      for ($i=0; $i<count($this->psr_keys); $i++) {
        $psr  = $this->psr_keys[$i];
        $raj  = $this->psrs[$psr]["RAJ"];
        $decj = $this->psrs[$psr]["DECJ"];
        echo "ras['".$psr."'] = '".$raj."';\n";
        echo "decs['".$psr."'] = '".$decj."';\n";
      }
?>
      function startButton() 
      {

        document.getElementById("command").value = "start";

        var i = 0;
        var psr = "";
        var mode = "";

        updateRADEC();

        i = document.getElementById("SOURCE_LIST").selectedIndex;
        psr = document.getElementById("SOURCE_LIST").options[i].value;
        document.getElementById("SOURCE").value = psr;

        i = document.getElementById("MODE_LIST").selectedIndex;
        mode = document.getElementById("MODE_LIST").options[i].value;
        document.getElementById("MODE").value = mode;

        if ((mode == "CAL") || (mode == "LEVCAL")) {
          document.getElementById("SOURCE").value = psr + "_R";
        }
        document.tcs.submit();
      }

      function stopButton() {
        document.getElementById("command").value = "stop";
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
    $this->state_banner->printJavaScriptHead();
  }

  /*************************************************************************************************** 
   *
   * HTML for this page 
   *
   ***************************************************************************************************/
  function printHTML()
  {
    $this->state_banner->printHTML();

    echo "<br>\n";

    $this->openBlockHeader("TCS Simulator");
?>
    <form name="tcs" target="tcs_interface" method="GET">
    <table border=0 cellpadding=5 cellspacing=0 width='100%'>
      <tr>

        <td class='key'>CALFREQ</td>
        <td class='val'><input type="text" name="CALFREQ" size="5" value="11.123"> [MHz]</td>

        <td class='key'>CFREQ</td>
        <td class='val'><input type="text" name="CFREQ" size="12" value="1382.00000"> [MHz]</td>

        <td class='key'>BANDWIDTH</td>
        <td class='val'>
          <select name='BANDWIDTH'>
            <option value="-256.0000">-256.0000</option>
            <option value="64.0000">64.0000</option>
            <option value="1024.0000">1024.0000</option>
          </select>
        </td>

        <td class='key'>NDIM</td>
        <td class='val'><input type="text" name="NDIM" size="2" value="2" readonly> +</td>

      </tr>
      <tr>

        <td class='key'>SOURCE</td>
        <td class='val'>
          <input type="hidden" id="SOURCE" name="SOURCE">
    <?
          echo "<select id='SOURCE_LIST' name='SOURCE_LIST' onChange='updateRADEC()'>\n";
          for ($i=0; $i<count($this->psr_keys); $i++) {
            $psr = $this->psr_keys[$i];
            if ($psr == "J0437-4715") {
              echo "<option value='".$psr."' selected>".$psr."</option>\n";
            } else {
             echo "<option value='".$psr."'>".$psr."</option>\n";
            }
          }
          echo "</select>\n";
    ?>
        </td>

        <td class='key'>RA</td>
        <td class='val'><input type="text" id="RA" name="RA" size="12" value="04:37:00.00" readonly> *</td>

        <td class='key'>DEC</td>
        <td class='val'><input type="text" id="DEC" name="DEC" size="12" value="-47:35:00.0" readonly> *</td>

        <td class='key'>NBIT</td>
        <td class='val'>
          <select name="NBIT">
            <option value=2>2</option>
            <option value=4>4</option>
            <option value=8 selected>8</option>
          </select>
        </td>
      
      </tr>
      <tr>

        <td class='key'>RECEIVER</td>
        <td class='val'>
          <select name="RECEIVER">
            <option value="MULTI">MULTI</option>
            <option value="1050CM">1050CM</option>
          </select> *
        </td>

        <td class='key'>PID</td>
        <td class='val'>
          <select name="PID">
    <?      for ($i=0; $i<count($this->groups); $i++) 
            {
              $g = $this->groups[$i];
              $selected = "";
              if ($g == "P999")
                $selected = " selected";
              echo "        <option value=".$g.$selected.">".$g."</option>\n";
            } ?>
          </select>
        </td>

        <td class='key'>PROC FILE</td>
        <td class='val'>
          <select name="PROCFIL">
            <option value="dspsr.select">dspsr.select</option>
            <option value="dspsr.1sec">dspsr.1sec</option>
            <option value="dspsr.multi">dspsr.multi</option>
            <option value="dspsr.giant">dspsr.giant</option>
            <option value="dspsr.single">dspsr.single</option>
            <option value="dspsr.singleF">dspsr.singleF</option>
            <option value="dspsr.singleM">dspsr.singleM</option>
            <option value="apsr.scratch">apsr.scratch</option>
            <option value="P778.disk">P778.disk</option>
          </select>
        </td>

        <td class='key'>NPOL</td>
        <td class='val'><input type="text" name="NPOL" size="2" value="2" readonly> +</td>

     </tr>
     <tr>

        <td class='key'>MODE</td>
        <td class='val'>
          <input type="hidden" id="MODE" name="MODE" value="">
          <select id="MODE_LIST" name="MODE_LIST">
            <option value="PSR">PSR</option>
            <option value="CAL">CAL</option>
            <option value="LEVCAL">LEVCAL</option>
          </select>
        </td>

        <td class='key'>CONFIG</td>
        <td class='val'>
          <select name="CONFIG">
    <?
            for ($i=0; $i<count($this->pdfb_configs); $i++) {
              echo "      <option value='".$this->pdfb_configs[$i]."'>".$this->pdfb_configs[$i]."</option>\n";
            }
    ?>
          </select> *
        </td>

        <td colspan='4'>
           <font size="-1">* has no effect on APSR<br/>
                           + constant for all APSR modes</font>
        </td>
      </tr>

      <tr>
        <td colspan=8><hr></td>
      </tr>

      <tr>
        <td colspan=8 align='center'>
          <input type="button" value="Start" onClick="startButton()">
          <input type="button" value="Stop" onClick="stopButton()">
        </td>
      </tr>
    </table>
    <input type="hidden" id="command" name="command" value="">
    </form>
<?
    $this->closeBlockHeader();

    echo "<br/>\n";

    // have a separate frame for the output from the TCS interface
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
    echo "</head>\n";
?>
</head>
<body style="background-image: none; background: #FFFFFF;">
<table border=0 width="100%">
 <tr>
  <th align="left" width="50%">Command</th>
  <th align="left" width="50%">Response</th>
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
    if ($get["command"] == "stop") {
      $cmd = "stop\r\n";
      socketWrite($sock,$cmd);
      $result = rtrim(socketRead($sock));
      $this->printTR($cmd,$result);
      $this->printTF();
      $this->printFooter();
      socket_close($sock);
      return;
    }   

    # otherwise its a START command
    $keys = array_keys($get);
    for ($i=0; $i<count($keys); $i++) {

      $k = $keys[$i];
      if (($k != "command") && ($k != "SOURCE_LIST") && ($k != "MODE_LIST")) {
        if ($get[$k] != "") {
          $cmd = $k." ".$get[$k]."\r\n";
          socketWrite($sock,$cmd);
          $result = rtrim(socketRead($sock),"\r\n");
          if ($result != "ok") {
            $this->printTR("HEADER command failed ", $result.": ".rtrim(socketRead($sock)));
            $this->printTR("START aborted", "");
            socket_close($sock);
            return;
          }
          $this->printTR($cmd,$result);
        } else {
          $this->printTR($k, "Ignoring as value was empty");
        }
      }
    }

    # Issue START command to server_tcs_interface 
    $cmd = "start\r\n";
    socketWrite($sock,$cmd);
    $result = rtrim(socketRead($sock),"\r\n");
    $this->printTR(strtoupper($cmd),$result);
    if ($result != "ok") {
      $this->printTR("START command failed", $result.": ".rtrim(socketRead($sock)));
      $this->printTF();
      $this->printFooter();
      socket_close($sock);
      return;
    }

    # wait for the next modulo 10 second time
    $time_unix = time();
    if ($time_unix % 10 != 0) {
      $sleep_time = 10-($time_unix % 10);
      $this->printTR("Waiting ".$sleep_time." secs for next 10 second boundary", "ok");
      sleep($sleep_time);
    }

    # now wait 5 seconds before issuing the BAT/SET_UTC_START command
    #$this->printTR("Waiting 5 seconds to issue SET_UTC_START", "ok");
    sleep(5);

    # generate a UTC_START time in the future
    $time_unix = time();

    # ensure the UTC_START is modulo 10
    if ($time_unix % 10 != 0) {
      $time_unix += (10 - ($time_unix % 10));
    }

    $utc_start = gmdate(DADA_TIME_FORMAT,$time_unix);

    # Issue SET_UTC_START command
    $cmd = "SET_UTC_START ".$utc_start."\r\n";
    socketWrite($sock,$cmd);
    $result = rtrim(socketRead($sock));

    if ($result != "ok") {
      $result .= "<BR>\n".rtrim(socketRead($sock));
      $this->printTR("SET_UTC_START [".$cmd."] failed: ", $result);
      $this->printTR(rtrim(socketRead($sock)),"");
      $this->stopCommand($sock);
      exit(-1);
    } else {
      $this->printTR($cmd, "ok");
    }

    $this->printTF();
    $this->printFooter();
    socket_close($sock);
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
    echo "</body>\n";
    echo "</html>\n";
  }

  function printTF() {
    echo "</table>\n";
  }

  # Issue the STOP command
  function stopCommand($socket) {

    $cmd = "STOP\r\n";
    socketWrite($socket,$cmd);

    $result = rtrim(socketRead($socket));
    if ($result != "ok") {
      $this->printTR("\"$cmd\" failed",$result.": ".rtrim(socketRead($socket)));
    } else {
      $this->printTR("Sent \"".$cmd."\" to nexus","ok");
    }
    return $result;
  }
}

if (isset($_GET["command"])) {
  $obj = new tcs_simulator();
  $obj->printTCSResponse($_GET);
} else {
  handleDirect("tcs_simulator");
}



