<?PHP
include("definitions_i.php");
include("functions_i.php");

if ((isset($_GET["change"])) && ($_GET["change"] == "true")) {
  
  // Sleep for 5 seconds to ensure all browser sessions get a chance
  // to change instrument
  sleep(5);

  if (file_exists(CHANGE_INSTRUMENT_FILE)) {
    unlink (CHANGE_INSTRUMENT_FILE);
  }
}

?>
<html>
<head>

  <script>

  var url="/instrument_update.php"

  function looper() {
    request()
    setTimeout('looper()',2000)
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

      // If we need to refresh 
      if (response == "change") {
        document.location = "instrument.php?change=true";
        parent.banner.document.location = "statebanner.php";
        parent.statuswindow.document.location = "statuswindow.php";
        parent.infowindow.document.location = "infowindow.php";
        parent.controlwindow.document.location = "controlwindow.php";
        parent.plotwindow.document.location = "plotwindow.php";
        parent.logheader.document.location = "logheader.php";
        parent.logwindow.document.location = "machine_status.php?machine=nexus";
      }

    }
  }
  </script>

</head>
<body onload="looper()">
</body>
</html>
