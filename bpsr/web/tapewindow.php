<?PHP
include("definitions_i.php");
include("functions_i.php");

$config = getConfigFile(SYS_CONFIG, TRUE);

?>
<html>
<? include("header_i.php"); ?>

<script type="text/javascript">

  //soundManager.url = '/sounds/sm2-swf-movies/'; // directory where SM2 .SWFs live

  // disable debug mode after development/testing..
  //soundManager.debugMode = false;

  //soundManager.waitForWindowLoad = true;
  //soundManager.onload = function() {
    // SM2 has loaded - now you can create and play sounds!
    //soundManager.createSound('changetape','/sounds/tapechange.mp3');
  //}

  var url="/bpsr/tapeupdate.php?control_dir=/nfs/control/bpsr"

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

          if (values[1] == "Insert Tape") {

            var html = "    <div class=\"btns\">\n"
            html += "<a href=\"/bpsr/tapeupdate.php?control_dir=/nfs/control/bpsr"
            html += "&tapeinserted="+values[2]+"&location="+values[0]+"\" class=\"btn\" >"
            html += " <span>Load "+values[2]+"</span></a>\n";
            html += "</div>\n"

            document.getElementById(values[0]).innerHTML = html
            document.getElementById(values[0]+"_td").style.backgroundColor = "red"
            //soundManager.play('changetape');
          } else {
            document.getElementById(values[0]+"_td").style.backgroundColor = ""
          }
        }
      }
    }
  }

</script>

<body onload="looper()">
<? 
?>
  <table cellpadding=0 cellspacing=0 border=0 width=100%>
    <tr>
      <td width=3>&nbsp;</td>
      <td id="XFER_td">
        <table> <tr>
          <td align="right" class="smalltext"><b>XFER</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="XFER"></span></td>
        </tr> </table>
      </td>
      <td id="SWIN_td">
        <table> <tr>
          <td align="right" class="smalltext"><b>SWIN</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="SWIN"></span></td>
        </tr> </table>
      </td>
      <td id="PARKES_td">
        <table> <tr>
          <td align="right" class="smalltext"><b>PARKES</b></td>
          <td width=5>&nbsp;</td>
          <td align="left" class="smalltext"><span class="smalltext" id ="PARKES"></span></td>
        </tr> </table>
      </td>
    </tr>
  </table>
</body>
</html>

