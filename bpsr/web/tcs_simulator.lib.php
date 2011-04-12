<?PHP

include("bpsr.lib.php");
include("bpsr_webpage.lib.php");

class tcs_simulator extends bpsr_webpage 
{

  var $groups = array();
  var $psrs = array();
  var $psr_keys = array();
  var $inst = 0;

  function tcs_simulator()
  {
    bpsr_webpage::bpsr_webpage();

    $this->title = "BPSR | TCS Simulator";
    $this->inst = new bpsr();
    $this->groups = $this->inst->getPIDS();
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

      var ras = { 'J0437-4715':'04:37:00.00','G302.9-37.3':'00:52:00.666' };
      var decs = { 'J0437-4715':'-47:35:00.0','G302.9-37.3':'-79:47:53.58' };

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
            <option value="SURVEY.MULTIBEAM">SURVEY.MULTIBEAM</option>
            <option value="THEDSPSR">THEDSPSR</option>
          </select>
        </td>

        <td class='key'>BAND</td>
        <td class='val'><input type="text" name="band" value="-400.00" size="12" readonly></td>

        <td class='key'>ACC LEN</td>
        <td class='val'><input type="text" name="acclen" size="2" value="25" readonly></td>

        <td class='key'>TSCRUNCH *</td>
        <td class='val'><input type="text" name="tscrunch" size="1" value="1" readonly></td>

      </tr>
      <tr>

        <td class='key'>SOURCE</td>
        <td class='val'>
          <input type="hidden" id="src" name="src" value="">
          <select id="src_list" name="src_list" onChange='updateRADEC()'>
            <option value='J0437-4715'>J0437-4715</option>
            <option value='G302.9-37.3'>G302.9-37.3</option>
          </select>
        </td>

        <td class='key'>RA</td>
        <td class='val'><input type="text" id="ra" name="ra" size="12" value="04:37:00.00" readonly></td>


        <td class='key'>NBIT</td>
        <td class='val'><input type="text" name="nbit" size="2" value="8" readonly></td>

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
        <td class='val'><input type="text" name="fscrunch" size="1" value="1" readonly></td>

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
      if (($k != "command") && ($k != "src_list")) {
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
    $this->printTR($cmd,$result);
    if ($result != "ok") {
      $this->printTR("START command failed", $result.": ".rtrim(socketRead($sock)));
      $this->printTF();
      $this->printFooter();
      socket_close($sock);
      return;
    } else {
      $this->printTR("Sent START to nexus", "ok");
    }

    # now wait for the UTC_START to come back to us
    $this->printTR("Waiting for UTC_START to be reported", "...");
    $result = rtrim(socketRead($sock));
    $this->printTR($result, "ok");

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

}

if (isset($_GET["command"])) {
  $obj = new tcs_simulator();
  $obj->printTCSResponse($_GET);
} else {
  handleDirect("tcs_simulator");
}



