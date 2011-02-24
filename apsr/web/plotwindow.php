<?PHP
include("definitions_i.php");
include("functions_i.php");

$config = getConfigFile(SYS_CONFIG);
$conf = getConfigFile(DADA_CONFIG,TRUE);
$spec = getConfigFile(DADA_SPECIFICATION, TRUE);

?>
<html>
<head>
  <? echo STYLESHEET_HTML; ?>
  <? echo FAVICO_HTML?>
  <script type="text/javascript">

  // This URL will return the names of the 5 current timestamped images();
  var url = "plotupdate.php?results_dir=<?echo $config["SERVER_RESULTS_DIR"]?>";
  
  var npsrs = 0;
  var utc_start = "";
  var psrs = new Array();

  function popWindow(URL) {
    day = new Date();
    id = day.getTime();
    eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=1,scrollbars=1,location=1,statusbar=0,menubar=1,resizable=1,width=900,height=700');");
  }

  function popImage(URL) {
    day = new Date();
    id = day.getTime();
    eval("page" + id + " = window.open(URL, '" + id + "', 'toolbar=0,scrollbars=0,location=1,statusbar=0,menubar=0,resizable=1,width=1080,height=800');");
  }


  function handle_data(http_request) {

    if (http_request.readyState == 4) {
      var response = String(http_request.responseText)
      var lines = response.split("\n");

      var img
      var link
      var pos = 0
      var i = 0
      var j = 0
      var base = 0;
      var psr

      /* the current select HTML obj */
      var label_psrs = document.getElementById("single_psr");
      var psrs_select = document.getElementById("psrs");
      var utc_start_span = document.getElementById("utc_start");
      var selected_psr = psrs_select.selectedIndex;
      var custom_plots = document.getElementById("custom_plot");

      if ((response.indexOf("Could not connect to") == -1) &&
          (lines.length >= 9) ) {
    
        var rebuild_select = 0;
        psrs = new Array();

        /* parse the utc_start */
        values = lines[0].split(":::");
        if (values[1] != utc_start) {
          rebuild_select = 1;
        }
        utc_start = values[1];
        custom_plots.onclick = new Function("popWindow('custom_plot.php?basedir=<?echo $config["SERVER_RESULTS_DIR"]?>&utc_start="+utc_start+"')");

        /* parse the number of pulsars */ 
        values = lines[1].split(":::");
        if (npsrs != values[1]) {
          rebuild_select = 1;
        }
        npsrs = values[1];

        /* parse the data from each PSR, updating the global array */  
        for (i=0; i<npsrs; i++) {
          base = 2 + (i*7);
          values = lines[base].split(":::");
          psrs[i] = values[1]; 
          if ((!(psrs_select.options[i])) || (psrs[i] != psrs_select.options[i].value)) {
            rebuild_select = 1;
          }
        }

        if (rebuild_select == 1) { 

          //alert("rebuilding select");

          /* destroy and rebuild the select element */
          if (npsrs > 1) {
            psrs_select.options.length = 0;
            for (i=0; i < npsrs; i++) {
              psrs_select.options[i] = new Option(psrs[i],psrs[i]);
            }
            psrs_select.selectedIndex = 0;
            psrs_select.style.display = "inline";
            label_psrs.style.display = "none";
            utc_start_span.innerHTML = utc_start

          /* hide the select element, and just display a simple text field */
          } else {
            psrs_select.style.display = "none";
            label_psrs.style.display = "inline";
            label_psrs.innerHTML = psrs[0];
            utc_start_span.innerHTML = utc_start
          }
        } else {
          //alert("not rebuilding select");
        }

        /* extract the images for the currently selected pulsar */
        if (npsrs == 1) {
          selected_psr = 0
        } else {
          selected_psr = psrs_select.selectedIndex;
        }

        /* line the lines array for the selected PSR */
        base = (selected_psr*7) + 2;

        for (i=1; i<=6; i++) {

          values = lines[base+i].split(":::");
          parts = values[0].split("_");

          img = document.getElementById(parts[0]);
          link = document.getElementById(parts[0]+"_a");

          if (parts[1] == "240x180") {
            if (img.src != values[1]) {
              img.src = values[1]
            }

          // Hi res image for a "click"
          } else {

            if (values[1].indexOf("blankimage") > 0) {
              //link.href="javascript:void(0)";
              img.border=0
            } else {
              link.href="javascript:popImage('"+values[1]+"')";
              img.border=2
            }
          }
        } 
      }
    }
  }

  function request() {
    if (window.XMLHttpRequest)
      http_request = new XMLHttpRequest()
    else
      http_request = new ActiveXObject("Microsoft.XMLHTTP");

    http_request.onreadystatechange = function() {
      handle_data(http_request)
    }
    http_request.open("GET", url, true)
    http_request.send(null)
  }

  function looper() {
    request()
    setTimeout('looper()',4000)
  }

                                                                                                                 

</script>
</head>

<body onload="looper()">
<? 
?>
  <center>
  PSR: <span id="single_psr"></span>
  <select id="psrs" onchange="request()"></select>&nbsp;&nbsp;&nbsp;
  <span id="utc_start"></span>&nbsp;&nbsp;&nbsp;
  <span><input id="custom_plot" type=button value="View Custom Plots" onClick="popWindow('custom_plot.php')"></span>
  <br> 
  </center>
  <table border=0 width="100%" cellspacing=0 cellpadding=5>
  <tr>
    <td align="center" width=240px height=180px>
      <font class="smalltext">Total Intensity vs Phase</font><br>
      <a id="pvfl_a" href="about:blank">
        <img id="pvfl" src="/images/blankimage.gif" alt="No Data Available">
      </a>
    </td>
    <td align="center" width=240px height=180px>
      <font class="smalltext">Phase vs Time</font><br>
      <a id="pvt_a" href="about:blank">
        <img id="pvt" src="/images/blankimage.gif"alt="No Data Available">
      </a>
    </td>
    <td align="center" width=240px height=180px>
      <font class="smalltext">Phase vs Frequency</font><br>
      <a id="pvfr_a" href="about:blank">
        <img id="pvfr" src="/images/blankimage.gif" alt="No Data Available">
      </a>
    </td>
  </tr>
  </table>
</body>
</html>

