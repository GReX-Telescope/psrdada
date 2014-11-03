<?PHP

//
// Web interface to Ewan/Duncan's TCC
//

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class dcw_simulator extends mopsr_webpage 
{
  var $inst = 0;
  var $valid_srcs;

  function dcw_simulator()
  {
    mopsr_webpage::mopsr_webpage();

    $this->callback_freq = 5000;
    $this->title = "MOPSR | DCW Simulator";
    $this->inst = new mopsr();
    $this->valid_srcs = array("J0437-4715", "J0534+2200", "J0610-2100", "J0613-0200",
                              "J0711-6830", "J0737-3039A", "J0742-2822", "J0835-4510",
                              "J0900-3144", "J0904-7459", "J0953+0755", "J1001-5939",
                              "J1018-7154", "J1022+1001", "J1024-0719", "J1045-4509",
                              "J1103-5355", "J1125-5825", "J1125-6014", "J1141-6545",
                              "J1226-6202", "J1431-4717", "J1439-5501", "J1525-5544",
                              "J1546-4552", "J1600-3053", "J1603-7202", "J1643-1224",
                              "J1644-4559", "J1713+0747", "J1717-4045", "J1718-3718", "J1730-2304",
                              "J1732-5049", "J1744-1134", "J1824-2452", "J1857+0943",
                              "J1909-3744", "J1933-6211", "J1939+2134", "J2124-3358",
                              "J2129-5721", "J2145-0750", "J2241-5236", "J1456-6843",
                              "CenA",       "3C273",      "J0630-2834", "J1017-7156",
                              "J0738-4042", "J0630-2834", "J0738-4042", "J0742-2822",
                              "J0835-4510", "J0837-4135", "J0953+0755", "J1136+1551",
                              "J1453-6413", "J1456-6843", "J1559-4438", "J1644-4559",
                              "J1645-0317", "J1752-2806", "J1932+1059", "J1935+1616",
                              "HydraA",   
			      "CJ0252-7104",
			      "CJ0408-7507",
			      "CJ0408-6545", 
			      "CJ0519-4545",
			      "CJ0522-3627",
			      "CJ1020-4251", 
			      "CJ1154-3505", 
			      "CJ1218-4600", 
			      "CJ1248-4118", 
			      "CJ1325-4301",
			      "CJ1424-4913", 
			      "CJ1556-7914", 
			      "CJ1737-5632", 
                              "CJ1819-6345", 
			      "CJ1830-3602", 
			      "CJ1935-4620", 
			      "CJ1924-2833");
  }

  function javaScriptCallback()
  {
    return "dcw_simulator_request();";
  }


  function printJavaScriptHead()
  {
    $this->srcs = $this->inst->getPsrcatPsrs();

    $this->srcs["CenA"] = array();
    $this->srcs["CenA"]["RAJ"] = "13:25:27.6";
    $this->srcs["CenA"]["DECJ"] = "-43:01:09";

    $this->srcs["3C273"] = array();
    $this->srcs["3C273"]["RAJ"] = "12:29:06.7";
    $this->srcs["3C273"]["DECJ"] = "02:03:09.0";

    $this->srcs["HydraA"] = array();
    $this->srcs["HydraA"]["RAJ"] = "09:18:06";
    $this->srcs["HydraA"]["DECJ"] = "-12:05:45";

    $this->srcs["CJ0252-7104"] = array();
    $this->srcs["CJ0252-7104"]["RAJ"] = "02:52:46.3";
    $this->srcs["CJ0252-7104"]["DECJ"] = "-71:04:36.2";

    $this->srcs["CJ0408-6545"] = array();
    $this->srcs["CJ0408-6545"]["RAJ"] = "04:08:20.28";
    $this->srcs["CJ0408-6545"]["DECJ"] = "-65:45:08.1";

    $this->srcs["CJ0408-7507"] = array();
    $this->srcs["CJ0408-7507"]["RAJ"] = "04:08:48.5";
    $this->srcs["CJ0408-7507"]["DECJ"] = "-75:07:20.0";

    $this->srcs["CJ0519-4545"] = array();
    $this->srcs["CJ0519-4545"]["RAJ"] = "05:19:26.6";
    $this->srcs["CJ0519-4545"]["DECJ"] = "-45:45:58.9";

    $this->srcs["CJ0522-3627"] = array();
    $this->srcs["CJ0522-3627"]["RAJ"] = "05:22:57.8";
    $this->srcs["CJ0522-3627"]["DECJ"] = "-36:27:31.0";

    $this->srcs["CJ1020-4251"] = array();
    $this->srcs["CJ1020-4251"]["RAJ"] = "10:20:03.5";
    $this->srcs["CJ1020-4251"]["DECJ"] = "-42:51:33.0";

    $this->srcs["CJ1154-3505"] = array();
    $this->srcs["CJ1154-3505"]["RAJ"] = "11:54:21.9";
    $this->srcs["CJ1154-3505"]["DECJ"] = "-35:05:32.1";

    $this->srcs["CJ1218-4600"] = array();
    $this->srcs["CJ1218-4600"]["RAJ"] = "12:18:06.0";
    $this->srcs["CJ1218-4600"]["DECJ"] = "-46:00:29.0";

    $this->srcs["CJ1248-4118"] = array();
    $this->srcs["CJ1248-4118"]["RAJ"] = "12:48:49.8";
    $this->srcs["CJ1248-4118"]["DECJ"] = "-41:18:42.2";

    $this->srcs["CJ1305-4928"] = array();
    $this->srcs["CJ1305-4928"]["RAJ"] = "13:05:27.4";
    $this->srcs["CJ1305-4928"]["DECJ"] = "-49:28:06.3";

    $this->srcs["CJ1325-4301"] = array();
    $this->srcs["CJ1325-4301"]["RAJ"] = "13:25:24.0";
    $this->srcs["CJ1325-4301"]["DECJ"] = "-43:01:38.1";

    $this->srcs["CJ1424-4913"] = array();
    $this->srcs["CJ1424-4913"]["RAJ"] = "14:24:32.18";
    $this->srcs["CJ1424-4913"]["DECJ"] = "-49:13:17.16";

    $this->srcs["CJ1556-7914"] = array();
    $this->srcs["CJ1556-7914"]["RAJ"] = "15:56:57.8";
    $this->srcs["CJ1556-7914"]["DECJ"] = "-79:14:03.8";

    $this->srcs["CJ1737-5632"] = array();
    $this->srcs["CJ1737-5632"]["RAJ"] = "17:37:42.85";
    $this->srcs["CJ1737-5632"]["DECJ"] = "-56:32:46.0";

    $this->srcs["CJ1819-6345"] = array();
    $this->srcs["CJ1819-6345"]["RAJ"] = "18:19:35.0";
    $this->srcs["CJ1819-6345"]["DECJ"] = "-63:45:48.6";

    $this->srcs["CJ1830-3602"] = array();
    $this->srcs["CJ1830-3602"]["RAJ"] = "18:30:58.8";
    $this->srcs["CJ1830-3602"]["DECJ"] = "-36:02:30.3";

    $this->srcs["CJ1924-2833"] = array();
    $this->srcs["CJ1924-2833"]["RAJ"] = "19:24:50.2";
    $this->srcs["CJ1924-2833"]["DECJ"] = "-28:33:39.4";

    $this->srcs["CJ1935-4620"] = array();
    $this->srcs["CJ1935-4620"]["RAJ"] = "19:35:57.2";
    $this->srcs["CJ1935-4620"]["DECJ"] = "-46:20:43.1";

    $this->srcs["CJ2334-4125"] = array();
    $this->srcs["CJ2334-4125"]["RAJ"] = "23:34:26.1";
    $this->srcs["CJ2334-4125"]["DECJ"] = "-41:25:25.8";

    $this->src_keys = array_keys($this->srcs);
?>
    <style type='text/css'>

      table.tcc_status th {
        text-align: left;
        border-style: none none solid none;
        border-color: #005700 #005700 #005700 #005700;
        border-width: 0px 0px 1px 0px;
        color: #005700;
        padding: 4px;
        font-size: 12pt;
      }

      table.tcc_status td.key {
        text-align: right;
        font-size: 10pt;
        font-weight: bold;
        min-width: 100px;
      }

      table.tcc_status td.val {
        min-width: 80px;
        padding-right: 20px;
        text-align: left;
      }

    </style>

    <script type='text/javascript'>
      var ras = { 'default':'00:00:00.00'<?
      for ($i=0; $i<count($this->src_keys); $i++)
      {
        $p = $this->src_keys[$i];
        if (in_array($p, $this->valid_srcs))
        {
          echo ",'".$p."':'".$this->srcs[$p]["RAJ"]."'";
        }
      }
      ?>};

      var decs = { 'default':'00:00:00.00'<?
      for ($i=0; $i<count($this->src_keys); $i++)
      {
        $p = $this->src_keys[$i];
        if (in_array($p, $this->valid_srcs))
        {
          echo ",'".$p."':'".$this->srcs[$p]["DECJ"]."'";
        }
      }
      ?>};

      function pointButton() 
      {
        document.getElementById("command").value = "point";

        var i = 0;
        var src = "";

        i = document.getElementById("src_list").selectedIndex;
        src = document.getElementById("src_list").options[i].value;

        document.getElementById("source").value = src;

        if (document.getElementById("observer").value == "")
        {
          alert ("You must specify an observer ident");
        }
        else
        {
          document.dcw.submit();
        }
      }

      function stopButton() {
        document.getElementById("command").value = "stop";
        document.dcw.submit();
      }

      function windStowButton() {
        document.getElementById("command").value = "wind_stow";
        document.dcw.submit();
      }

      function maintenanceStowButton() {
        document.getElementById("command").value = "maintenance_stow";
        document.dcw.submit();
      }

      function updateRADEC() {
        var i = document.getElementById("src_list").selectedIndex;
        var src = document.getElementById("src_list").options[i].value;
        var src_ra = ras[src];
        var src_dec= decs[src];
        document.getElementById("ra").value = src_ra;
        document.getElementById("dec").value = src_dec;
      }

      function handle_dcw_simulator_request(ds_xml_request)
      {
        if (ds_xml_request.readyState == 4)
        {
          var xmlDoc = ds_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj = xmlDoc.documentElement;

            var error_element = xmlObj.getElementsByTagName("socket_error")[0];
            try {
              error_message = error_element.childNodes[0].nodeValue;
              if (error_message != "")
              {
                document.getElementById("error_string").innerHTML = error_message;
                if (error_message != "")
                {
                  return;
                }
              }
              else
              {
                document.getElementById("error_string").innerHTML = "";
              }
            } catch (e) {
              // do nothing
            }

            section = xmlObj.getElementsByTagName("overview");
            var nodes = section[0].children;
            var tag, val;
            for (i=0; i<nodes.length; i++)
            {
              if (nodes[i].nodeType == 1)
              {
                tag = nodes[i].nodeName;
                try {
                  val = nodes[i].childNodes[0].nodeValue  
                  document.getElementById(tag).innerHTML = val;
                } catch (e) {
                }
              }
            }

            var section = xmlObj.getElementsByTagName("interface");
            var nodes = section[0].children;
            for (i=0; i<nodes.length; i++)
            {
              if (nodes[i].nodeType == 1)
              {
                tag = nodes[i].nodeName;
                val = "";
                try {
                  val = nodes[i].childNodes[0].nodeValue
                  document.getElementById(tag).innerHTML = val;
                } catch (e) {
                }
              }
            }

            try {
              var section = xmlObj.getElementsByTagName("coordinates");
              var nodes = section[0].children;
              for (i=0; i<nodes.length; i++)
              {
                if (nodes[i].nodeType == 1)
                {
                  tag = nodes[i].nodeName;
                  val = "";
                  try {
                    val = nodes[i].childNodes[0].nodeValue
                    document.getElementById(tag).innerHTML = val;
                  } catch (e) {
                  }
                }
              }
            } catch (e) {

            }

            var t1 = "west_arm";
            var section = xmlObj.getElementsByTagName(t1);
            var subsections = section[0].children;

            var ns_on_target = false;
            var md_on_target = false;

            for (h=0;h<subsections.length; h++)
            {
              var t2 = subsections[h].nodeName;
              var nodes = subsections[h].children;
              for (i=0; i<nodes.length; i++)
              {
                if (nodes[i].nodeType == 1)
                {
                  tag = t1 + "_" + t2 + "_" + nodes[i].nodeName;
                  val = "";
                  try {
                    val = nodes[i].childNodes[0].nodeValue
                  } catch (e) {

                  }
                  if (nodes[i].nodeName == "on_target")
                  { 
                    if (val == "True")
                    {
                       document.getElementById(tag).style.backgroundColor = "lightgreen";
                    }
                    else
                    {
                      document.getElementById(tag).style.backgroundColor = "";
                    }
                  }

                  if (nodes[i].nodeName == "tilt")
                  {
                    target_tilt_tag = t2.substr(0,2).toUpperCase();
                    target_tilt = document.getElementById(target_tilt_tag).innerHTML;
                    tilt_remaining = parseFloat(target_tilt) - parseFloat(val);
                    newval = parseFloat(val).toFixed(4) + "  [" + tilt_remaining.toFixed(4) + " offset]";
                    val = newval;
                  }
                  document.getElementById(tag).innerHTML = val;
                }
              }
            }

            var t1 = "east_arm";
            var section = xmlObj.getElementsByTagName(t1);
            var subsections = section[0].children;
            for (h=0;h<subsections.length; h++)
            {
              var t2 = subsections[h].nodeName;
              var nodes = subsections[h].children;
              for (i=0; i<nodes.length; i++)
              {
                if (nodes[i].nodeType == 1)
                {
                  tag = t1 + "_" + t2 + "_" + nodes[i].nodeName;
                  val = "";
                  try {
                    val = nodes[i].childNodes[0].nodeValue
                  } catch (e) {
                  }
                  document.getElementById(tag).innerHTML = val;
                  if (nodes[i].nodeName == "on_target")
                  { 
                    if (val == "True")
                    {
                       document.getElementById(tag).style.backgroundColor = "lightgreen";
                    }
                    else
                    {
                      document.getElementById(tag).style.backgroundColor = "";
                    }
                  }
                  if (nodes[i].nodeName == "tilt")
                  {
                    target_tilt_tag = t2.substr(0,2).toUpperCase();
                    target_tilt = document.getElementById(target_tilt_tag).innerHTML;
                    tilt_remaining = parseFloat(target_tilt) - parseFloat(val);
                    newval = parseFloat(val).toFixed(4) + "  [" + tilt_remaining.toFixed(4) + " offset]";
                    val = newval;
                  }
                  document.getElementById(tag).innerHTML = val;

                }
              }
            }
          }
        }
      }

      function dcw_simulator_request()
      {
        var url = "dcw_simulator.lib.php?update=true";

        if (window.XMLHttpRequest)
          ds_xml_request = new XMLHttpRequest();
        else
          ds_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        ds_xml_request.onreadystatechange = function() {
          handle_dcw_simulator_request(ds_xml_request)
        };
        ds_xml_request.open("GET", url, true);
        ds_xml_request.send(null);
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
    $this->openBlockHeader("TCC Status");

?>
    <div id='socket_error'></div>

    <table border='0' cellpadding=5px>
    <tr>
    <td valign=top>

    <table class='tcc_status'>
      <tr><th colspan=2>Overview</th></tr>
      <tr><td class='key'>At Limits</td><td class='val' id='at_limits'></td></tr>
      <tr><td class='key'>At Hard Limits</td><td class='val' id='at_hard_limits'></td></tr>
      <tr><td class='key'>Has Coordinates</td><td class='val' id='has_coordinates'></td></tr>
      <tr><td class='key'>On Target</td><td class='val' id='on_target'></td></tr>
      <tr><td class='key'>Tracking</td><td class='val' id='tracking'</td></tr>
      <tr><td class='key'>Slewing</td><td class='val' id='slewing'></td></tr>
      <tr><td class='key'>High Wind</td><td class='val' id='high_wind'></td></tr>
      <tr><td class='key'>Error</td><td class='val' id='error_string'></td></tr>
    </table>

    </td>
    <td valign=top>
   
    <table width='100%' class='tcc_status'>
      <tr><th colspan=2>Interface</th></tr>
      <tr><td class='key'>Controlled By</td><td class='val' id='controlled_by'></td></tr>   
      <tr><td class='key'>Comment</td><td class='val' id='comment'></td></tr>   
      <tr><td class='key'>Locked</td><td class='val' id='locked'></td></tr>   
      <tr><td class='key'>Override Mode</td><td class='val' id='override_mode'></td></tr>   
      <tr><td class='key'>Admin Mode</td><td class='val' id='admin_mode'></td></tr>   
    </table>

    </td>
    <td valign='top'>

    <table width='100%' class='tcc_status'>
      <tr><th colspan=2>Coordinates</th></tr>
      <tr><td class='key'>RA</td><td class='val' id='RA'></td></tr>   
      <tr><td class='key'>Dec</td><td class='val' id='Dec'></td></tr>   
      <tr><td class='key'>HA</td><td class='val' id='HA'></td></tr>   
      <tr><td class='key'>Glat</td><td class='val' id='Glat'></td></tr>   
      <tr><td class='key'>Glon</td><td class='val' id='Glon'></td></tr>   
      <tr><td class='key'>Alt</td><td class='val' id='Alt'></td></tr>   
      <tr><td class='key'>Az</td><td class='val' id='Az'></td></tr>   
      <tr><td class='key'>NS</td><td class='val' id='NS'></td></tr>   
      <tr><td class='key'>MD</td><td class='val' id='MD'></td></tr>   
    </table>

    </td>
    <td valign='top'>

    <table width='100%' class='tcc_status'>
      <tr><th></th><th>West Arm</th><th>East Arm</th><tr>
      <tr><td class='key'>NS Tilt</td><td class='val' id='west_arm_ns_drive_tilt'></td><td class='val' id='east_arm_ns_drive_tilt'></td></tr>
      <tr><td class='key'>NS Status</td><td class='val' id='west_arm_ns_drive_status'></td><td class='val' id='east_arm_ns_drive_status'></td></tr>
      <tr><td class='key'>NS At Limit</td><td class='val' id='west_arm_ns_drive_at_limit'></td><td class='val' id='east_arm_ns_drive_at_limit'></td></tr>
      <tr><td class='key'>NS On Target</td><td class='val' id='west_arm_ns_drive_on_target'></td><td class='val' id='east_arm_ns_drive_on_target'></td></tr>
  
      <tr><td class='key'>MD Tilt</td><td class='val' id='west_arm_md_drive_tilt'></td><td class='val' id='east_arm_md_drive_tilt'></td></tr>
      <tr><td class='key'>MD Status</td><td class='val' id='west_arm_md_drive_status'></td><td class='val' id='east_arm_md_drive_status'></td></tr>
      <tr><td class='key'>MD At Limit</td><td class='val' id='west_arm_md_drive_at_limit'></td><td class='val' id='east_arm_md_drive_at_limit'></td></tr>
      <tr><td class='key'>MD On Target</td><td class='val' id='west_arm_md_drive_on_target'></td><td class='val' id='east_arm_md_drive_on_target'></td></tr>
    </table>

    </td>
    </tr>
    </table>

<?
    $this->closeBlockHeader();

    $this->openBlockHeader("DCW Simulator");
?>
    <form name="dcw" target="dcw_interface" method="GET">
    <table border=0 cellpadding=5 cellspacing=0>

      <tr>
        <td colspan=4></td>

        <td>RA / MD tilt</td>
        <td>DEC</td>

        <td colspan=3></td>

      </tr>

      <tr>

        <td class='key'>Source</td>
        <td class='val'>
          <input type="hidden" id="source" name="source" value="">
          <select id="src_list" name="src_list" onChange='updateRADEC()'>
<?
          for ($i=0; $i<count($this->src_keys); $i++)
          {
            $p = $this->src_keys[$i];
            if (in_array($p, $this->valid_srcs))
            {
              if ($p == "J0835-4510")
                echo "            <option value='".$p."' selected>".$p."</option>\n";
              else
                echo "            <option value='".$p."'>".$p."</option>\n";
            }
          }
?>
          </select>
        </td>

        <td class='space'>&nbsp;</td>

        <td class='key'>Position</td>
        <td class='val'><input type="text" id="ra" name="ra" size="12" value="08:35:20.61149"></td>
        <td class='val'><input type="text" id="dec" name="dec" size="12" value="-45:10:34.8751"></td>

        <td class='space'>&nbsp;</td>

        <td class='key'>Tracking</td>
        <td class=val'>
          <input type="radio" id="tracking" name="tracking" value="on" checked/>On<br/>
          <input type="radio" id="tracking" name="tracking" value="off"/>Off
        </td>

      </tr>

      <tr>

        <td class='key'>Observer</td>
        <td class='val'><input type="text" id="observer" name="observer" size="8" value=""></td>

        <td class='space'>&nbsp;</td>

        <td class='key'>Offset</td>
        <td class='val'><input type="text" id="ra_offset" name="ra_offset" size="12" value="0"> [min]</td>
        <td class='val'><input type="text" id="dec_offset" name="dec_offset" size="12" value="0"> [arcmin]</td>

        <td class='space'>&nbsp;</td>

        <td class='key'>System</td>
        <td class=val'>
          <input type="radio" id="system" name="system" value="equatorial" checked/>Equatorial<br/>
          <input type="radio" id="system" name="system" value="ewdec"/>EW Dec
        </td>

      </tr>

      <tr>
        <td colspan=10>
          <div class="btns" style='text-align: center'>
            <a href="javascript:pointButton()"  class="btn" > <span>Track</span> </a>
            <a href="javascript:stopButton()"  class="btn" > <span>Stop</span> </a>
            <a href="javascript:windStowButton()"  class="btn" > <span>Wind Stow</span> </a>
            <a href="javascript:maintenanceStowButton()"  class="btn" > <span>Maintenance Stow</span> </a>
          </div>
        </td>
      </tr>
    </table>

    <input type="hidden" id="command" name="command" value="">
    </form>
<?
    $this->closeBlockHeader();

    echo "<br/>\n";

    // have a separate frame for the output from the TCC interface
    $this->openBlockHeader("TCC");
?>
    <iframe name="dcw_interface" src="" width=100% frameborder=0 height='350px'></iframe>
<?
    $this->closeBlockHeader();
  }

  function printTCCResponse($get)
  {
    // Open a connection to the TCC interface script
    $host = "172.17.227.103";
    $port = 38012;
    $sock = 0;

    echo "<html>\n";
    echo "<head>\n";
    for ($i=0; $i<count($this->css); $i++)
    {
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
    }
    echo "</head>\n";

    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml .= "<tcc_request>";
    $xml .=   "<user_info>";
    $xml .=     "<name>".$get["observer"]."</name>";
    $xml .=     "<comment></comment>";
    $xml .=   "</user_info>";

    if ($get["command"] == "stop")
    {
      $xml .= "<tcc_command>";
      $xml .=   "<command>stop</command>";
      $xml .= "</tcc_command>";
    }
    else if ($get["command"] == "point")
    {
      $xml .= "<tcc_command>";
      $xml .=   "<command>point</command>";

      $xml .=   "<pointing units='hhmmss' epoch='2000' tracking='".$get["tracking"]."' system='".$get["system"]."'>";

      $ra  = $get["ra"];
      $dec = $get["dec"];
      if (strcmp($get["system"], "equatorial") == 0)
      {
        $ra = $this->inst->addToRA($get["ra"], $get["ra_offset"]);
        $dec = $this->inst->addToDEC($get["dec"], $get["dec_offset"]);
      }

      $xml .=     "<xcoord>".$ra."</xcoord>";
      $xml .=     "<ycoord>".$dec."</ycoord>";

      $xml .=   "</pointing>";
      $xml .= "</tcc_command>";
    }
    else if ($get["command"] == "wind_stow")
    {
      $xml .= "<tcc_command>";
      $xml .=   "<command>wind_stow</command>";
      $xml .= "</tcc_command>";
    }
    else if ($get["command"] == "maintenance_stow")
    {
      $xml .= "<tcc_command>";
      $xml .=   "<command>maintenance_stow</command>";
      $xml .= "</tcc_command>";
    }
    else
    {
      echo "ERROR: unrecognized command [".$get["command"]."]<BR>\n";
    }

    $xml .= "</tcc_request>";

?>
<body>
<table border=1 width='100%'>
 <tr>
  <th>Command</th>
  <th>Response</th>
 </tr>
<?
    list ($socket, $message) = openSocket($host, $port, 2);
    if (!$socket) 
    {
      $this->printTR("Error: opening socket to TCC [".$host.":".$port."]: ".$message, "");
      $this->printTF();
      $this->printFooter();
      return;
    }

    $html_cmd = str_replace("><", "&#62;\n&#60;", $xml);
    $html_cmd = str_replace("<", "&#60;", $html_cmd);
    $html_cmd = str_replace(">", "&#62;", $html_cmd);
    $html_cmd = str_replace("\n", "<br/>", $html_cmd);

    $xml = str_replace("\n", "", $xml);

    $this->printTR ("Sending", $html_cmd);

    socketWrite ($socket, $xml."\r\n");

    $flags = 0;
    $xml = "";

    $bytes_recv = socket_recv ($socket, $xml, 262144, $flags);
    if ($bytes_recv === false)
    {
      $xml .= "<?xml version='1.0' encoding='ISO-8859-1'?>";
      $xml .= "<tcc_reply>";
      $xml .= "<socket_error>".socket_strerror(socket_last_error($socket))."</socket_error>";
      $xml .= "</tcc_reply>";
    }
    else
    {
      
    }
    socket_close($socket);
      
    $html_response = str_replace("><", "&#62;\n&#60;", $xml);
    $html_response = str_replace("<", "&#60;", $html_response);
    $html_response = str_replace(">", "&#62;", $html_response);
    $html_response = str_replace("\n", "<br/>\n", $html_response);

    $this->printTR ("Received", $html_response);
    $this->printTF();
    $this->printFooter ();

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

  function printUpdateHTML() 
  {
    $host = "172.17.227.103";
    $port = 38013;

    error_reporting(E_ALL);

    // create a socket
    $socket = socket_create(AF_INET, SOCK_DGRAM, SOL_UDP);
    if ($socket === FALSE)
    {
      $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>";
      $xml .= "<tcc_status>";
      $xml .= "<socket_error>Failed to create socket: ".socket_strerror(socket_last_error($socket))."</socket_error>";
      $xml .= "</tcc_status>";
      header('Content-type: text/xml');
      echo $xml;
      return;
    }

    if (socket_set_option ($socket, SOL_SOCKET, SO_REUSEADDR, 1) === FALSE)
    {
      $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>";
      $xml .= "<tcc_status>";
      $xml .= "<socket_error>Failed to set socket option: ".socket_strerror(socket_last_error($socket))."</socket_error>";
      $xml .= "</tcc_status>";
    }
    else
    {
      // bind it to the specified port
      if (@socket_bind ($socket, $host, $port) === FALSE)
      {
        $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>";
        $xml .= "<tcc_status>";
        $xml .= "<socket_error>Failed to bind socket to ".$host.":".$port.": ".socket_strerror(socket_last_error($socket))."</socket_error>";
        $xml .= "</tcc_status>";
      }
      else
      {
        // create read/write sets for select, with a 1.5s timeout
        $read   = array($socket);
        $write  = NULL;
        $except = NULL;
        $timeout_secs = 1;

        #echo "socket_select with ".$timeout_secs." timeout<BR/>\n";
        $num_changed_sockets = socket_select($read, $write, $except, $timeout_secs);
        #echo "number of changed sockets=".$num_changed_sockets."<BR>\n";

        // no packet received in 3 seconds
        if ($num_changed_sockets === false)
        {
          $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>";
          $xml .= "<tcc_status>";
          $xml .= "<socket_error>Failed to select on socket: ".socket_strerror(socket_last_error($socket))." s</socket_error>";
          $xml .= "</tcc_status>";
        }
        else if ($num_changed_sockets == 0)
        {
          $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>";
          $xml .= "<tcc_status>";
          $xml .= "<socket_error>No UDP packet received within ".$timeout_secs." s</socket_error>";
          $xml .= "</tcc_status>";
        }
        else
        {
          $from = '';
          $port = 0;
          $flags = 0;
          $bytes_recvd = @socket_recvfrom($socket, $xml, 1500, $flags, $from, $port);
        }
      }
    }
    socket_close($socket);

    header('Content-type: text/xml');
    echo $xml;
  }
};

if (isset($_GET["command"])) {
  $obj = new dcw_simulator();
  $obj->printTCCResponse($_GET);
} else {
  handleDirect("dcw_simulator");
}
