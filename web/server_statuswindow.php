<?PHP 

/*
 * Monitors WARN and ERROR files for the server daemons specified 
 * in the relevant CONFIG file
 */

include("definitions_i.php");
include("functions_i.php");

?>

<html>
<? include("header_i.php"); ?>

<script type="text/javascript">
  
  var url = "statusupdate.php";

  /* cause the javascript to loop */
  function looper() {
    request()
    setTimeout('looper()',5000)
  }

  /* send the request for new data */
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
 
  /* handle the response from request() */ 
  function handle_data(http_request) {
    if (http_request.readyState == 4) {
      var response = String(http_request.responseText)
      var lines = response.split(";;;")

      var values = lines[0].split(":::")
      var num_pwc = values[1]

      var i

      var loglengthIndex = parent.parent.logheader.document.getElementById("loglength").selectedIndex
      var loglength = parent.parent.logheader.document.getElementById("loglength").options[loglengthIndex].value

      for (i=0; i<num_pwc; i++) {

        j = 1 + i*7;

        values = lines[j].split(":::")
        var pwc_name = values[1];
        values = lines[(j+1)].split(":::")
        var pwc_status = values[1];
        values = lines[(j+2)].split(":::")
        var pwc_message = values[1];
        values = lines[(j+3)].split(":::")
        var src_status = values[1];
        values = lines[(j+4)].split(":::")
        var src_message = values[1];
        values = lines[(j+5)].split(":::")
        var sys_status = values[1];
        values = lines[(j+6)].split(":::")
        var sys_message = values[1];

        /* Change the link to point at correct log */
        document.getElementById("a_"+pwc_name+"_pwc");

        if (pwc_status == <?echo STATUS_OK?>) {
          /* Update the image */
          document.getElementById("img_"+pwc_name+"_pwc").src = "/images/green_light.png"
          document.getElementById("img_"+pwc_name+"_pwc").title = "";
          document.getElementById("a_"+pwc_name+"_pwc").href = "logwindow.php?machine="+pwc_name+"&loglevel=all&loglength="+loglength+"&daemon=<?echo INSTRUMENT?>_pwc_monitor&autoscroll=true"
        }
        if (pwc_status == <?echo STATUS_WARN?>) {
          /* Update the image */
          document.getElementById("img_"+pwc_name+"_pwc").src = "/images/yellow_light.png"
          document.getElementById("img_"+pwc_name+"_pwc").title = pwc_message
          document.getElementById("a_"+pwc_name+"_pwc").href = "logwindow.php?machine="+pwc_name+"&loglevel=warn&loglength="+loglength+"&daemon=<?echo INSTRUMENT?>_pwc_monitor&autoscroll=true"
        }
        if (pwc_status == <?echo STATUS_ERROR?>) {
          /* Update the image */
          document.getElementById("img_"+pwc_name+"_pwc").src = "/images/red_light.png"
          document.getElementById("img_"+pwc_name+"_pwc").title = pwc_message
          document.getElementById("a_"+pwc_name+"_pwc").href = "logwindow.php?machine="+pwc_name+"&loglevel=error&loglength="+loglength+"daemon=<?echo INSTRUMENT?>_pwc_monitor&autoscroll=true"
        }

        if (src_status == <?echo STATUS_OK?>) {
          /* Update the image */
          document.getElementById("img_"+pwc_name+"_src").src = "/images/green_light.png"
          document.getElementById("img_"+pwc_name+"_src").title = "";
          document.getElementById("a_"+pwc_name+"_src").href = "logwindow.php?machine="+pwc_name+"&loglevel=all&loglength="+loglength+"&daemon=<?echo INSTRUMENT?>_src_monitor&autoscroll=true"
        }
        if (src_status == <?echo STATUS_WARN?>) {
          /* Update the image */
          document.getElementById("img_"+pwc_name+"_src").src = "/images/yellow_light.png"
          document.getElementById("img_"+pwc_name+"_src").title = src_message
          document.getElementById("a_"+pwc_name+"_src").href = "logwindow.php?machine="+pwc_name+"&loglevel=warn&loglength="+loglength+"&daemon=<?echo INSTRUMENT?>_src_monitor&autoscroll=true"
        }
        if (src_status == <?echo STATUS_ERROR?>) {
          /* Update the image */
          document.getElementById("img_"+pwc_name+"_src").src = "/images/red_light.png"
          document.getElementById("img_"+pwc_name+"_src").title = src_message
          document.getElementById("a_"+pwc_name+"_src").href = "logwindow.php?machine="+pwc_name+"&loglevel=error&loglength="+loglength+"&daemon=<?echo INSTRUMENT?>_src_monitor&autoscroll=true"
        }

        if (sys_status == <?echo STATUS_OK?>) {
          /* Update the image */
          document.getElementById("img_"+pwc_name+"_sys").src = "/images/green_light.png"
          document.getElementById("img_"+pwc_name+"_sys").title = "";
          document.getElementById("a_"+pwc_name+"_sys").href = "logwindow.php?machine="+pwc_name+"&loglevel=all&loglength="+loglength+"&daemon=<?echo INSTRUMENT?>_sys_monitor&autoscroll=true"
        }
        if (sys_status == <?echo STATUS_WARN?>) {
          /* Update the image */
          document.getElementById("img_"+pwc_name+"_sys").src = "/images/yellow_light.png"
          document.getElementById("img_"+pwc_name+"_sys").title = sys_message
          document.getElementById("a_"+pwc_name+"_sys").href = "logwindow.php?machine="+pwc_name+"&loglevel=warn&loglength="+loglength+"&daemon=<?echo INSTRUMENT?>_sys_monitor&autoscroll=true"
        }
        if (sys_status == <?echo STATUS_ERROR?>) {
          /* Update the image */
          document.getElementById("img_"+pwc_name+"_sys").src = "/images/red_light.png"
          document.getElementById("img_"+pwc_name+"_sys").title = sys_message
          document.getElementById("a_"+pwc_name+"_sys").href = "logwindow.php?machine="+pwc_name+"&loglevel=error&loglength="+loglength+"&daemon=<?echo INSTRUMENT?>_sys_monitor&autoscroll=true"
        }
      }
    }
  }

  var activeMachine = "nexus";
                                                                                   
  function setActiveMachine(machine) {
                                                                                   
    var machine

    var oldmachineIndex = parent.parent.logheader.document.getElementById("active_machine").selectedIndex
    var oldmachine = parent.parent.logheader.document.getElementById("active_machine").options[oldmachineIndex].value
    var loglengthIndex = parent.parent.logheader.document.getElementById("loglength").selectedIndex
    var loglength = parent.parent.logheader.document.getElementById("loglength").options[loglengthIndex].value

    /* Unselect the currently selected machine */
    document.getElementById("td_"+oldmachine+"_txt").className = "notselected"
    document.getElementById("td_"+oldmachine+"_pwc").className = "notselected"
    document.getElementById("td_"+oldmachine+"_src").className = "notselected"
    document.getElementById("td_"+oldmachine+"_sys").className = "notselected"
   
    Select_Value_Set(parent.parent.logheader.document.getElementById("active_machine"), machine);                                                                               
    /* Select the new machine */
    document.getElementById("td_"+machine+"_txt").className = "selected"
    document.getElementById("td_"+machine+"_pwc").className = "selected"
    document.getElementById("td_"+machine+"_src").className = "selected"
    document.getElementById("td_"+machine+"_sys").className = "selected"
  }

  function Select_Value_Set(SelectObject, Value) {
    for(index = 0; index < SelectObject.length; index++) {
     if(SelectObject[index].value == Value)
       SelectObject.selectedIndex = index;
     }
  }


</script>
</head>
<body onload="looper()">
<?PHP

$config = getConfigFile(SYS_CONFIG);
$server_daemons = split(" ",$config["SERVER_DAEMONS"]);

?>
<table border=0 cellspacing=0 cellpadding=0>
<?
$j = 0;
for ($i=0; $i<count($server_daemons); $i++) {

  $d = $server_daemons[$i];

  if ($j == 0) {
    echo "  <tr>\n";
  }

  $link_id = "a_".$d;
  $img_id = "img_".$d;
  $td_id = "td_".$d;

  echo "    <td id=\"".$td_id."\">".statusLight(1, "OK", $link_id, $img_id)." ".$d."</td>\n";

  if ($j == 0) {
    echo "  </tr>\n";
  }
  $j++;

  if ($j == 2) {
    $j=0;
  }
}
?>

</table>

</body>
</html>
<?

function statusLight($status, $message, $linkid, $imgid) {

  $daemon = "";
  if (strstr($linkid, "nexus_pwc") !== FALSE) {
    $daemon = "&daemon=pwc";
  }
  if (strstr($linkid, "nexus_src") !== FALSE) {
    $daemon = "&daemon=src";
  }
  if (strstr($linkid, "nexus_sys") !== FALSE) {
    $daemon = "&daemon=sys";
  }

  if ($status == STATUS_OK) {
    $url = "logwindow.php?machine=nexus&loglevel=all".$daemon;
    $title = "OK";
  } else if ($status == STATUS_WARN) {
    $url = "logwindow.php?machine=nexus&loglevel=warn".$daemon;
    $title = "Warning";
  } else if ($status == STATUS_ERROR) {
    $url = "logwindow.php?machine=nexus&loglevel=error".$daemon;
    $title = "Error";
  } else {
    $url = "logwindow.php?machine=nexus&loglevel=all".$daemon;
    $title = "Message";
  }
                                                                                   
  $string = '<a target="logwindow" id="'.$linkid.'" href="'.$url.'" '.
            'onClick="setActiveMachine(\'nexus\')"';
    
  $string .= ' TITLE="'.$title.': '.$message.'">';

  #$string .= ' TITLE="header=['.$title.'] body=['.$message.'] '.
  #            'cssbody=[ttbody] cssheader=[ttheader]">';

  $string .= '
       <img id="'.$imgid.'" border="none" width="15px" height="15px" src="/images/';
                                                                                   
  if ($status == STATUS_OK) {
    return $string."green_light.png\" alt=\"OK\">\n      </a>\n";
  } else if ($status == STATUS_WARN) {
    return $string."yellow_light.png\" alt=\"WARN\">\n      </a>\n";
  } else {
    return $string."red_light.png\" alt=\"ERROR\">\n      </a>\n";
  }
                                                                                   
}
?>
