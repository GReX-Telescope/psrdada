<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class tcs_simulator extends bpsr_webpage 
{
  var $groups = array();
  var $psrs = array();
  var $psr_keys = array();
  var $valid_psrs = array();
  var $inst = 0;

  function tcs_simulator()
  {
    bpsr_webpage::bpsr_webpage();

    $this->title = "BPSR | TCS Simulator";
    $this->inst = new bpsr();
    $this->groups = $this->inst->getPIDS();
    array_push($this->groups, "P999");
    array_push($this->groups, "P456");

    $this->valid_psrs = array("J0437-4715", "J0534+2200", "J0610-2100", "J0613-0200",
                              "J0711-6830", "J0737-3039A", "J0742-2822", "J0835-4510",
                              "J0900-3144", "J0904-7459", "J1001-5939", "J1017-7156",
                              "J1018-7154", "J1022+1001", "J1024-0719", "J1045-4509",
                              "J1103-5355", "J1125-5825", "J1125-6014", "J1141-6545",
                              "J1226-6202", "J1431-4717", "J1439-5501", "J1525-5544",
                              "J1546-4552", "J1600-3053", "J1603-7202", "J1643-1224",
                              "J1713+0747", "J1718-3718", "J1730-2304", "J1732-5049",
                              "J1744-1134", "J1824-2452", "J1857+0943", "J1909-3744",
                              "J1933-6211", "J1939+2134", "J2124-3358", "J2129-5721",
                              "J2145-0750", "J2241-5236");

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
      var ras = { 'default':'00:00:00.00'<?
      for ($i=0; $i<count($this->psr_keys); $i++)
      {
        $p = $this->psr_keys[$i];
        if (in_array($p, $this->valid_psrs))
        {
          echo ",'".$p."':'".$this->psrs[$p]["RAJ"]."'";
        }
      }
      ?>};

      var decs = { 'default':'00:00:00.00'<?
      for ($i=0; $i<count($this->psr_keys); $i++)
      {
        $p = $this->psr_keys[$i];
        if (in_array($p, $this->valid_psrs))
        {
          echo ",'".$p."':'".$this->psrs[$p]["DECJ"]."'";
        }
      }
      ?>};


      function startButton() {

        document.getElementById("command").value = "start";

        var i = 0;
        var psr = "";

        updateRADEC();

        i = document.getElementById("src_list").selectedIndex;
        psr = document.getElementById("src_list").options[i].value;

        document.getElementById("src").value = psr;

        document.tcs.submit();

      }

      function stopButton() {
        document.getElementById("command").value = "stop";
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
    $this->openBlockHeader("TCS Simulator");
?>
    <form name="tcs" target="tcs_interface" method="GET">
    <table border=0 cellpadding=5 cellspacing=0 width='100%'>
      <tr>

        <td class='key'>PROC FILE</td>
        <td class='val'>
          <select name="procfil">
            <option value="BPSR.NULL">BPSR.NULL</option>
            <option value="SURVEY.MULTIBEAM">SURVEY.MULTIBEAM</option>
            <option value="THEDSPSR">THEDSPSR</option>
            <option value="BPSR_256CH_64US_1B">BPSR_256CH_64US_1B</option>
            <option value="BPSR_256CH_128US_1B">BPSR_256CH_128US_1B</option>
            <option value="BPSR_128CH_64US_1B">BPSR_128CH_64US_1B</option>
            <option value="BPSR_128CH_512US_1B">BPSR_128CH_512US_1B</option>
            <option value="BPSR_128CH_256US_1B">BPSR_128CH_256US_1B</option>
            <option value="BPSR_128CH_128US_1B">BPSR_128CH_128US_1B</option>
            <option value="BPSR_128CH_1024US_1B">BPSR_128CH_1024US_1B</option>
            <option value="BPSR.SCRATCH">BPSR.SCRATCH</option>
          </select>
        </td>

        <td class='key'>BAND</td>
        <td class='val'><input type="text" name="band" value="-400.00" size="12" readonly></td>

        <td class='key'>ACC LEN</td>
        <td class='val'><input type="text" name="acclen" size="2" value="25" readonly></td>

        <td class='key'>TSCRUNCH</td>
        <td class='val'><input type="text" name="tscrunch" size="1" value="1"></td>

      </tr>
      <tr>

        <td class='key'>SOURCE</td>
        <td class='val'>
          <input type="hidden" id="src" name="src" value="">
          <select id="src_list" name="src_list" onChange='updateRADEC()'>
            <option value='G302.9-37.3'>G302.9-37.3</option>
<?
          for ($i=0; $i<count($this->psr_keys); $i++)
          {
            $p = $this->psr_keys[$i];
            if (in_array($p, $this->valid_psrs))
            {
              echo "            <option value='".$p."'>".$p."</option>\n";
            }
          }
?>
          </select>
        </td>

        <td class='key'>RA</td>
        <td class='val'><input type="text" id="ra" name="ra" size="12" value="04:37:00.00" readonly></td>


        <td class='key'>NBIT</td>
        <td class='val'><input type="text" name="nbit" size="2" value="2"></td>

        <td class='key'>CHANAV *</td>
        <td class='val'><input type="text" name="chanav" size="2" value="0" readonly></td>
      
      </tr>
      <tr>

        <td class='key'>PID</td>
        <td class='val'>
          <select name="pid">
<?          for ($i=0; $i<count($this->groups); $i++) {
              $pid = $this->groups[$i];
              if ($pid == "P999")
                echo "            <option value=".$pid." selected>".$pid."</option>\n";
              else
                echo "            <option value=".$pid.">".$pid."</option>\n";
            } 
?>
          </select>
        </td>

        <td class='key'>DEC</td>
        <td class='val'><input type="text" id="dec" name="dec" size="12" value="-47:35:00.0" readonly></td>

        <td class='key'>NPROD *</td>
        <td class='val'><input type="text" name="nprod" size="2" value="1" readonly></td>

        <td class='key'>FTMAX *</td>
        <td class='val'><input type="text" name="ftmax" size="2" value="0" readonly></td>
      
      </tr>
      <tr>

        <td class='key'>LENGTH</td>
        <td class='val'><input type="text" name="length" size="5" value=""> [s]</td>

        <td class='key'>FREQ</td>
        <td class='val'><input type="text" name="freq" size="12" value="1382.00" readonly></td>

        <td class='key'>TCONST *</td>
        <td class='val'><input type="text" name="tconst" size="6" value="1.0000" readonly></td>

        <td class='key'>FSCRUNCH *</td>
        <td class='val'><input type="text" name="fscrunch" size="1" value="1"></td>

      </tr>
  
      <tr>

        <td class='key'>REF_BEAM</td>
        <td class='val'><input type="text" name="refbeam" size="5" value="1"></td>

        <td class='key'>NBEAM</td>
        <td class='val'><input type="text" name="nbeam" size="2" value="13"></td>

        <td class='key'>OBSERVER</td>
        <td class='val'><input type="text" name="observer" size="6" value="TEST"></td>

        <td class='key'></td>
        <td class='val'></td>

      </tr>
  
      <tr>
        <td colspan=8><hr></td>
      </tr>
      
      <tr>
        <td colspan=4>
          <div class="btns" style='text-align: center'>
            <a href="javascript:startButton()"  class="btn" > <span>Start</span> </a>
            <a href="javascript:stopButton()"  class="btn" > <span>Stop</span> </a>
          </div>
        </td>
        <td colspan=4 style='text-align: right;'>
          <font size="-1">* has no effect on BPSR, for future use</font>
        </td>
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
<body>
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
    if ($get["command"] == "stop") {
      $cmd = "stop\r\n";
      socketWrite($sock,$cmd);
      list ($result, $response) = socketRead($sock);
      $this->printTR($cmd, $response);
      $this->printTF();
      $this->printFooter();
      socket_close($sock);
      return;
    }   

    # otherwise its a START command
    $keys = array_keys($get);
    for ($i=0; $i<count($keys); $i++) {

      $k = $keys[$i];
      if (($k != "command") && ($k != "src_list")) {
        if ($get[$k] != "") {
          $cmd = $k." ".$get[$k]."\r\n";
          socketWrite($sock, $cmd);
          list ($result, $response) = socketRead($sock);
          if ($response != "ok") 
          {
            $this->printTR("HEADER command failed ", $response);
            $this->printTR("START aborted", "");
            socket_close($sock);
            return;
          }
          $this->printTR($cmd, $result);
        } else {
          $this->printTR($k, "Ignoring as value was empty");
        }
      }
    }

    # Issue START command to server_tcs_interface 
    $cmd = "start\r\n";
    socketWrite($sock,$cmd);
    list ($result, $response) = socketRead($sock);
    $this->printTR($cmd, $response);
    if ($response != "ok") {
      $this->printTR("START command failed", $response);
      $this->printTF();
      $this->printFooter();
      socket_close($sock);
      return;
    } else {
      $this->printTR("Sent START to nexus", "ok");
    }

    # now wait for the UTC_START to come back to us
    $this->printTR("Waiting for UTC_START to be reported", "...");
    list ($result, $response) = socketRead($sock);
    $this->printTR($result, "ok");

    $this->printTF();
    $this->printFooter();
    socket_close($sock);
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
    echo "</body>\n";
    echo "</html>\n";
  }

  function printTF() {
    echo "</table>\n";
  }

}

if (isset($_GET["command"])) {
  $obj = new tcs_simulator();
  $obj->printTCSResponse($_GET);
} else {
  handleDirect("tcs_simulator");
}



