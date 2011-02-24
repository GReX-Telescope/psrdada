<?PHP

include("../definitions_i.php");
include("../functions_i.php");
include("../".$instrument.".lib.php");

$inst = new $instrument();

$groups = $inst->getProjectGroups();
$psrs   = $inst->getPsrcatPsrs();
$keys   = array_keys($psrs);

$pdfb_configs = array("pdfb3_1024_1024_1024", "pdfb3_1024_256_1024", "pdfb3_1024_256_2048", "pdfb3_1024_256_512", "pdfb3_1024_64_1024", "pdfb3_128_64_1024", "pdfb3_128_64_512", "pdfb3_2048_1024_1024", "pdfb3_2048_1024_2048", "pdfb3_2048_256_1024", "pdfb3_256_256_1024", "pdfb3_256_256_2048", "pdfb3_256_64_1024", "pdfb3_256_64_512", "pdfb3_512_1024_1024", "pdfb3_512_1024_2048", "pdfb3_512_256_1024", "pdfb3_512_256_2048", "pdfb3_512_64_1024", "pdfb3_512_64_512", "pdfb4_1024_1024_1024", "pdfb4_1024_256_1024", "pdfb4_256_1024_1024", "pdfb4_512_1024_1024", "pdfb4_512_256_512");

?>
<html>
<head>
<?
  $inst->print_head_int("APSR TCS Simulator | Interface", 0);
?>
  <script type="text/javascript">

function startButton() {

  document.getElementById("COMMAND").value = "START";

  var i = 0;
  var psr = "";
  var mode = "";

  i = document.getElementById("SOURCE_LIST").selectedIndex;
  psr = document.getElementById("SOURCE_LIST").options[i].value;

  i = document.getElementById("MODE_LIST").selectedIndex;
  mode = document.getElementById("MODE_LIST").options[i].value;

  document.getElementById("SOURCE").value = psr;
  document.getElementById("MODE").value = mode;

  if ((mode == "CAL") || (mode == "LEVCAL")) {
    document.getElementById("SOURCE").value = psr + "_R";
  }

  document.tcs.submit();

}

function stopButton() {
  document.getElementById("COMMAND").value = "STOP";
  document.tcs.submit();
}

function updateRADEC() {

  var ras = new Array();
  var decs = new Array();
  <?
  for ($i=0; $i<count($keys); $i++) {
    $psr = $keys[$i];
    $raj = $psrs[$psr]["RAJ"];
    $decj = $psrs[$psr]["DECJ"];
    echo "ras['".$psr."'] = '".$raj."';\n";
    echo "decs['".$psr."'] = '".$decj."';\n";
  }
  ?>
  var i = document.getElementById("SOURCE_LIST").selectedIndex;
  var psr = document.getElementById("SOURCE_LIST").options[i].value;
  var psr_ra = ras[psr];
  var psr_dec= decs[psr];
  document.getElementById("RA").value = psr_ra;
  document.getElementById("DEC").value = psr_dec;

}

  </script>

</head>
<body>

<center>

<form name="tcs" action="simulator.php" target="tcs_interface" method="GET">
<table border=0 cellpadding=5 cellspacing=0>
  <tr>

    <td>CALFREQ</td>
    <td><input type="text" name="CALFREQ" size="10" value="11"></td>

    <td width=20px>&nbsp;</td>

    <td>CFREQ</td>
    <td><input type="text" name="CFREQ" size="12" value="1382.00000"></td>

    <td width=20px>&nbsp;</td>

    <td>BANDWIDTH</td>
    <td><select name="BANDWIDTH">
          <option value="-256.0000">-256.0000</option>
          <option value="64.0000">64.0000</option>
          <option value="1024.0000">1024.0000</option>
        </select>

    <td width=20px>&nbsp;</td>

    <td>NDIM</td>
    <td><input type="text" name="NDIM" size="2" value="2" readonly></td>

  </tr>
  <tr>

    <td>SOURCE</td>
    <td>
      <input type="hidden" id="SOURCE" name="SOURCE">
<?
      echo "<select id='SOURCE_LIST' name='SOURCE_LIST' onChange='updateRADEC()'>\n"; 
      for ($i=0; $i<count($keys); $i++) {
        $psr = $keys[$i]; 
        if ($psr == "J0437-4715") {
          echo "<option value='".$psr."' selected>".$psr."</option>\n";
        } else {
         echo "<option value='".$psr."'>".$psr."</option>\n";
        }
      }
      echo "</select>\n";
?>

    <td width=20px>&nbsp;</td>

    <td>RA *</td>
    <td><input type="text" id="RA" name="RA" size="12" value="04:37:00.00" readonly></td>

    <td width=20px>&nbsp;</td>

    <td>DEC *</td>
    <td><input type="text" id="DEC" name="DEC" size="12" value="-47:35:00.0" readonly></td>

    <td width=20px>&nbsp;</td>

    <td>NBIT</td>
    <td>
      <select name="NBIT">
        <option value=2>2</option>
        <option value=4>4</option>
        <option value=8 selected>8</option>
      </select>
    </td>
  </tr>
  <tr>

    <td>RECEIVER *</td>
    <td><input type="text" name="RECEIVER" size="12" value="MULTI" readonly></td>

    <td width=20px>&nbsp;</td>

    <td>PID</td>
    <td>
      <select name="PID">
<?      for ($i=0; $i<count($groups); $i++) {
          $selected = "";
          if ($groups[$i] == "P999")  
            $selected = " selected";
          echo "        <option value=".$groups[$i].$selected.">".$groups[$i]."</option>\n";
        } ?>
      </select>
    </td>

    <td width=20px>&nbsp;</td>

    <td>PROCFIL</td>
    <td>
      <select name="PROCFIL">
        <option value="dspsr.select">dspsr.select</option>
        <option value="dspsr.1sec">dspsr.1sec</option>
        <option value="dspsr.multi">dspsr.multi</option>
        <option value="dspsr.giant">dspsr.giant</option>
        <option value="dspsr.single">dspsr.single</option>
        <option value="dspsr.singleF">dspsr.singleF</option>
        <option value="dspsr.singleM">dspsr.singleM</option>
        <option value="apsr.scratch">apsr.scratch</option>
        <option value="apsr.scratch">apsr.scratch</option>
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

    <td>CONFIG *</td>
    <td>
      <select name="CONFIG"> 
<?
        for ($i=0; $i<count($pdfb_configs); $i++) {
          echo "      <option value='".$pdfb_configs[$i]."'>".$pdfb_configs[$i]."</option>\n";
        }
?>
      </select>
    </td>

    <td width=20px>&nbsp;</td>

    <td colspan=5>* has no effect on APSR backend</td>


  </tr>
  <tr>
    <td colspan=11><hr></td>
  </tr>

  <tr>
    <td colspan=4>&nbsp;</td>
    <td>
      <div class="btns">
        <a href="javascript:startButton()"  class="btn" > <span>Start</span> </a>
        <a href="javascript:stopButton()"  class="btn" > <span>Stop</span> </a>
      </div>
    </td>
    <td colspan=6>&nbsp;</td>
</table>
<input type="hidden" id="COMMAND" name="COMMAND" value="">
</form>
  
</center>

</body>
</html>
