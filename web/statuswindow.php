<?PHP 

include("definitions_i.php");
include("functions_i.php");

?>

<html>
<? include("header_i.php"); ?>

<script type="text/javascript">
  
  var url = "statusupdate.php";

  function looper() {

    request()
    setTimeout('looper()',5000)

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

$pwc_config = getConfigFile(SYS_CONFIG);
$pwc_status = getAllStatuses($pwc_config);

$nexus_status = STATUS_OK;
$nexus_message = "no probs mate";

?>
<table border=0 cellspacing=0 cellpadding=0>
  <tr>
    <td></td>
    <td width="30px" align="center" valign="top" class="selected" id="td_nexus_txt"><?echo statusLink("nexus","Nexus");?></td>

<?
// Headings

  for($i=0; $i<$pwc_config["NUM_PWC"]; $i++) {
    echo "    <td width=\"30\" align=\"center\" class=\"notselected\" id=\"td_".$pwc_config["PWC_".$i]."_txt\">".statusLink($pwc_config["PWC_".$i],$i)."</td>\n";
  }
?>

  </tr>
  <tr height="25px">
    <td valign="center" class="notselected">PWC:</td>
    <td align="center" class="selected" id="td_nexus_pwc">
      <?echo statusLight($nexus_status, $nexus_message, "nexus", "a_nexus_pwc", "img_nexus_pwc")?>
    </td>

  <?

// PWC status
for($i=0; $i<$pwc_config["NUM_PWC"]; $i++) {

  $machine = $pwc_config["PWC_".$i];
  $status = $pwc_status["PWC_".$i."_STATUS"];
  $message = $pwc_status["PWC_".$i."_MESSAGE"];
  $linkid = "a_".$machine."_pwc";
  $imgid = "img_".$machine."_pwc";
  $tdid = "td_".$machine."_pwc";

?>
    <td class="notselected" id="<?echo $tdid?>" height="20px">
<?    echo "        ".statusLight($status,$message,$machine,$linkid,$imgid);  ?> 
    </td>
<? } ?>

  </tr>
  <tr height="25px">
    <td valign="center" class="notselected">SRC:</td>
    <td align="center" class="selected" id="td_nexus_src">
      <?echo statusLight($nexus_status, $nexus_message, "nexus", "a_nexus_src", "img_nexus_src")?>
    </td>
<?
// SRC status
for($i=0; $i<$pwc_config["NUM_PWC"]; $i++) {

  $machine = $pwc_config["PWC_".$i];
  $status = $pwc_status["SRC_".$i."_STATUS"];
  $message = $pwc_status["SRC_".$i."_MESSAGE"];
  $linkid = "a_".$machine."_src";
  $imgid = "img_".$machine."_src";
  $tdid = "td_".$machine."_src";

?>
    <td class="notselected" align="center" id="<?echo $tdid?>">
<?    echo "        ".statusLight($status,$message,$machine,$linkid,$imgid);  ?>
    </td>
<? } ?>                                                                                    

  </tr>

  <tr height="25px">
    <td valign="center" class="notselected">SYS:</td>
    <td align="center" class="selected" id="td_nexus_sys">
      <?echo statusLight($nexus_status, $nexus_message, "nexus", "a_nexus_sys", "img_nexus_sys")?>
    </td>

<?
// SYS status
for($i=0; $i<$pwc_config["NUM_PWC"]; $i++) {
                                                                                                                                                            
  $machine = $pwc_config["PWC_".$i];
  $status = $pwc_status["SYS_".$i."_STATUS"];
  $message = $pwc_status["SYS_".$i."_MESSAGE"];
  $linkid = "a_".$machine."_sys";
  $imgid = "img_".$machine."_sys";
  $tdid = "td_".$machine."_sys";
                                                                                                                                                            
?>
    <td class="notselected" align="center" id="<?echo $tdid?>">
<?    echo "        ".statusLight($status,$message,$machine,$linkid,$imgid);  ?>
    </td>
<? } ?>
                                                                                                                                                            
  </tr>

</table>

</body>
</html>
<?

function statusLink($machine, $linktext) {

  $url = "machine_status.php?machine=".$machine;

  $string = '<a target="logwindow" href="'.$url.'" '.
            'onClick="setActiveMachine(\''.$machine.'\')">'.
            $linktext.'</a>';

  //return $string;
  return $linktext;

}

function statusLight($status, $message, $machine, $linkid, $imgid) {

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
    //$url = "machine_status.php?machine=".$machine;
    $url = "logwindow.php?machine=".$machine."&loglevel=all".$daemon;
    $title = "OK: ".$machine;
  } else if ($status == STATUS_WARN) {
    $url = "logwindow.php?machine=".$machine."&loglevel=warn".$daemon;
    $title = "Warning: ".$machine;
  } else if ($status == STATUS_ERROR) {
    $url = "logwindow.php?machine=".$machine."&loglevel=error".$daemon;
    $title = "Error: ".$machine;
  } else {
    $url = "logwindow.php?machine=".$machine."&loglevel=all".$daemon;
    $title = "Message";
  }
                                                                                   
  $string = '<a target="logwindow" id="'.$linkid.'" href="'.$url.'" '.
            'onClick="setActiveMachine(\''.$machine.'\')"';
    
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
