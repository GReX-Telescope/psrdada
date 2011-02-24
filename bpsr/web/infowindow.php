<?PHP

include ("bpsr.lib.php");
$inst = new bpsr();

?>
<html>
<? 
  $inst->open_head();
  $inst->print_head_int("BPSR Info Window", 0);
?>

<script type="text/javascript">

  var url="/bpsr/infoupdate.php"

  function looper() {
    request()
    setTimeout('looper()',10000)
  }

  function request() {

    if (window.XMLHttpRequest)
      http_request = new XMLHttpRequest();
    else
      http_request = new ActiveXObject("Microsoft.XMLHTTP");

    http_request.onreadystatechange = function() {
      handle_data(http_request)
    };
    http_request.open("GET", url, true);
    http_request.send(null);

  }

  function handle_data(http_request) {
    if (http_request.readyState == 4) {
      var response = String(http_request.responseText)
      var lines = response.split(";;;")

      for (i=0; i<lines.length; i++) {
        var values = lines[i].split(":::");
        
        if ((values[0]) && (document.getElementById(values[0]))) {
          document.getElementById(values[0]).innerHTML = values[1]
        }
      }
    }
  }
</script>
<?
  $inst->close_head();
?>
<body onload="looper()">
<? 
?>
  <table border=0 cellspacing=0 cellpadding=5 width=100%>
  <tr>
    <td>
      <table cellpadding=0 cellspacing=0 border=0>
        <tr>
          <td align="right" class="smalltext"><b>Source</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="SOURCE"></span></td>
        </tr>
        <tr>
          <td align="right" class="smalltext"><b>RA</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="RA"></span></td>
        </tr>
        <tr>
          <td align="right" class="smalltext"><b>DEC</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="DEC"></span></td>
        </tr>
      </table>
    </td>
    <td width=5>&nbsp;</td>
    <td>
      <table cellpadding=0 cellspacing=0 border=0>
        <tr>
          <td align="right" class="smalltext"><b>CFREQ</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="CFREQ"></span> MHz</td>
        </tr>
        <tr>
          <td align="right" class="smalltext"><b>Bandwith</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="BANDWIDTH"></span> MHz</td>
        </tr>
        <tr>
          <td align="right" class="smalltext"><b>ACC LEN</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="ACC_LEN"></span></td>
        </tr>
      </table>
    </td>
    <td width=5>&nbsp;</td>
    <td>
      <table cellpadding=0 cellspacing=0 border=0>
        <tr>
          <td align="right" class="smalltext"><b>Num PWCs</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="NUM_PWC"></span></td>
        </tr>
        <tr>
          <td align="right" class="smalltext"><b>Project ID</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="PID"></span></td>
        </tr>
        <tr>
          <td align="right" class="smalltext"><b>UTC START</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="UTC_START"></span></td>
        </tr>
      </table>
    </td>
  </tr>
  </table>
</body>
</html>

