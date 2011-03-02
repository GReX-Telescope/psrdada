<?PHP

include("caspsr_webpage.lib.php");
include("definitions_i.php");
include("functions_i.php");
include($instrument.".lib.php");

class results extends caspsr_webpage 
{

  var $filter_types = array("", "SOURCE", "CFREQ", "BANDWIDTH", "PID", "UTC_START", "PROC_FILE");
  var $cfg = array();

  function results()
  {
    caspsr_webpage::caspsr_webpage();
    $inst = new caspsr();
    $this->cfg = $inst->config;
  }

  function javaScriptCallback()
  {
    return "results_request();";
  }

  function printJavaScriptHead($get)
  {
?>
    <style type="text/css">
      .processing {
        background-color: #FFFFFF;
      }

      .finished {
        background-color: #cae2ff;
      }
    </style>

    <script type='text/javascript'>

      var offset = <?echo $get["offset"]?>;
  
      function getLength()
      {
        var i = document.getElementById("displayLength").selectedIndex;
        var length = document.getElementById("displayLength").options[i].value;
        return parseInt(length);
      }

      function getOffset()
      {
        return offset;
      }

      function setOffset(value)
      {
        offset = value;
      }

      /* If a page reload is required */
      function changeLength() {

        var i = document.getElementById("displayLength").selectedIndex;
        var length = document.getElementById("displayLength").options[i].value;

        var show_inline;
        if (document.getElementById("inlineimages").checked)
          show_inline = "true";
        else
          show_inline = "false";

        var offset = getOffset()

        var URL = "results.lib.php?offset="+offset+"&length="+length+"&inlineimages="+show_inline;

        i = document.getElementById("filter_type").selectedIndex;
        var filter_type = document.getElementById("filter_type").options[i].value;

        var filter_value = document.getElementById("filter_value").value;

        if ((filter_value != "") && (filter_type != "")) {
          URL = URL + "&filter_type="+filter_type+"&filter_value="+filter_value;
        }
        document.location = URL;
      }

      function toggle_images()
      {
        var i = document.getElementById("displayLength").selectedIndex;
        var length = document.getElementById("displayLength").options[i].value;
        var img;
        var show_inline

        if (document.getElementById("inlineimages").checked) {
          show_inline = true;
          document.getElementById("IMAGE_TR").innerHTML = "IMAGE";
        } else {
          show_inline = false;
          document.getElementById("IMAGE_TR").innerHTML = "";
        }

        for (i=0; i<length; i++) {
          img = document.getElementById("img_"+i);
          if (show_inline) 
            img.style.display = "";
          else
            img.style.display = "none";
        }
      }

      function results_update_request() 
      {
        var ru_http_requset;

        // get the offset 
        var offset = getOffset();
        var length = getLength();

        var url = "results.lib.php?update=true&offset="+offset+"&length="+length;

        // check if inline images has been specified
        if (document.getElementById("inlineimages").checked)
          show_inline = "true";
        else
          show_inline = "false";
        url = url + "&inlineimages="+show_inline;
        
        // check if a filter has been specified
        var u, filter_type, filter_vaule;
        i = document.getElementById("filter_type").selectedIndex;
        filter_type = document.getElementById("filter_type").options[i].value;
        filter_value = document.getElementById("filter_value").value;
        if ((filter_value != "") && (filter_type != "")) {
          url = URL + "&filter_type="+filter_type+"&filter_value="+filter_value;
        }

        if (window.XMLHttpRequest)
          ru_http_request = new XMLHttpRequest()
        else
          ru_http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        ru_http_request.onreadystatechange = function() 
        {
          handle_results_update_request(ru_http_request)
        }

        ru_http_request.open("GET", url, true)
        ru_http_request.send(null)

      }

      function handle_results_update_request(xml_request) 
      {
        if (xml_request.readyState == 4) {

          var xmlDoc=xml_request.responseXML;
          var xmlObj=xmlDoc.documentElement; 

          var i, j, k, result, key, value, span, this_result;

          var results = xmlObj.getElementsByTagName("result");

          // for each result returned in the XML DOC
          for (i=0; i<results.length; i++) {

            result = results[i];
            this_result = new Array();

            for (j=0; j<result.childNodes.length; j++) 
            {

              // if the child node is an element
              if (result.childNodes[j].nodeType == 1) {
                key = result.childNodes[j].nodeName;
                // if there is a text value in the element
                if (result.childNodes[j].childNodes.length == 1) {
                  value = result.childNodes[j].childNodes[0].nodeValue;
                } else {
                  value = "";
                }
                this_result[key] = value;
              }
            }

            utc_start = this_result["UTC_START"];

            for ( key in this_result) {
              value = this_result[key];

              if (key == "SOURCE") {

                // ignore

              } else if (key == "IMG") {

                var url = "result.lib.php?utc_start="+utc_start;
                var link = document.getElementById("link_"+i);
                link.href = url;
                link.onmouseover = new Function("Tip(\"<img src='results/"+utc_start+"/"+value+"' width=241 height=181>\")");
                link.onmouseout = new Function("UnTip()");
                link.innerHTML = this_result["SOURCE"];

                try {
                  document.getElementById("img_"+i).src = "results/"+utc_start+"/"+value;
                } catch (e) {
                  // do nothing
                }

              } else if (key == "processing") {
                if (value == 1)
                  document.getElementById("row_"+i).className = "processing";
                else
                  document.getElementById("row_"+i).className = "finished";
  
              } else {
                try {
                  span = document.getElementById(key+"_"+i);
                  span.innerHTML = value;
                } catch (e) {
                  // do nothing 
                }
              }
            } // end for
          }
          
          // update then showing_from and showing_to spans
          var offset = getOffset();
          var length = getLength();
          document.getElementById("showing_from").innerHTML = offset;
          document.getElementById("showing_to").innerHTML = (offset + length);
        }
      }

    </script>
<?
  }

  function printJavaScriptBody()
  {
?>
    <!--  Load tooltip module for images as tooltips, hawt -->
    <script type="text/javascript" src="/js/wz_tooltip.js"></script>
<?
  }

  function printSideBarHTML($get) 
  {
     $this->openBlockHeader("Search Parameters");
?>
    <div>
      <table>
        <tr>
          <td>Filter</td>
          <td>
            <select name="filter_type" id="filter_type">
<?
            for ($i=0; $i<count($this->filter_types); $i++) {
              $t = $this->filter_types[$i];
              echoOption($t, $t, FALSE, $get["filter_type"]);
            }
?>
            </select>
          </td>
        </tr>
        <tr>
          <td>For</td>
          <td><input name="filter_value" id="filter_value" value="<?echo $get["filter_value"]?>" onChange="changeLength()"></td>
        </tr>
      </table>
    </div>

    <br/>

    <div>
      Inline Images <input type=checkbox id="inlineimages" name="inlineimages" onChange="toggle_images()"<? if($get["inlineimages"] == "true") echo " checked";?>>
    </div>

    <br/>

    <div>
      Num Results:
      <select name="displayLength" id="displayLength" onChange='changeLength()'>
  <?
        echoOption("20", "20", FALSE, $get["length"]);
        echoOption("50", "50", FALSE, $get["length"]);
        echoOption("100", "100", FALSE, $get["length"]);
  ?>
      </select>
    </div>

    <br/>

    <div>
    <table>
      <tr><td colspan=2 align=center>Legend</td></tr>
      <tr><td>CFREQ</td><td>Centre frequency of the observation [MHz]</td></tr>
      <tr><td>BW</td><td>Total bandwidth [MHz]</td></tr>
      <tr><td>LENGTH</td><td>Total intergration received [seconds]</td></tr>
      <tr><td>White</td><td>Newer results, may still be updated</td></tr>
      <tr><td>Blue</td><td>Finalised results, no new archives received for 5 minutes</td></tr>
    </table>

    </div>
  <?
     $this->closeBlockHeader();
    }

  /*************************************************************************************************** 
   *
   * HTML for this page 
   *
   ***************************************************************************************************/
  function printHTML($get) 
  {
    if (!isset($get["length"])) $get["length"] = 20;
    if (!isset($get["offset"])) $get["offset"] = 0;
    if (!isset($get["inlineimages"])) $get["inlineimages"] = "false";
    if (!isset($get["filter_type"])) $get["filter_type"] = "";
    if (!isset($get["filter_value"])) $get["filter_value"] = "";

?>
<html>
<head>
  <title>CASPSR | Results</title>
<?
    echo "    <link rel='shortcut icon' href='/caspsr/images/caspsr_favicon.ico'/>\n";
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
    for ($i=0; $i<count($this->ejs); $i++)
      echo "   <script type='text/javascript' src='".$this->ejs[$i]."'></script>\n";
?>
    
  <script type='text/javascript'>;
    function poll_server()
    {
      setTimeout('poll_server()', 10000);
    }
  </script>
<?

  $this->printJavaScriptHead($get);
?>
</head>

<body onload='poll_server()'>
<?
  $this->printJavaScriptBody();
?>
  <div class='PageBackgroundSimpleGradient'>
  </div>
  <div class='Main'>
    <div class="contentLayout">
      <div class="sidebar1">
        <div style='text-align: center; vertical-align: middle;'>
          <img src="/caspsr/images/caspsr_logo_200x60.png" width=200 height=60>
        </div>
<?
        $this->printSideBarHTML($get);
?>
      </div>
    </div>
<?
    flush();
?>
    <div class="content">
<?
        $this->printMainHTML($get);
?>
    </div>
  </div>
</body>
</html>
<?
  }

  function printMainHTML($get)
  {

    $this->openBlockHeader("Recent Results");

    $basedir = $this->cfg["SERVER_RESULTS_DIR"];
    $archive_ext = ".ar";
    $cmd = "";

    if (($get["filter_type"] == "") && ($get["filter_value"] == "")) {
      $cmd = "find ".$basedir." -maxdepth 2 -name 'obs.info' | wc -l";
    } else {
      if ($get["filter_type"] == "UTC_START") {
        $cmd = "find ".$basedir."/*".$get["filter_value"]."* -maxdepth 1 -name 'obs.info' | wc -l";
      } else {
        $cmd = "find ".$basedir." -maxdepth 2 -type f -name obs.info | xargs grep ".$get["filter_type"]." | grep ".$get["filter_value"]." | wc -l";
      }
    }
    $total_num_results = exec($cmd);

    $results = $this->getResultsArray($this->cfg["SERVER_RESULTS_DIR"], $this->cfg["SERVER_ARCHIVE_DIR"],  
                                      $archive_ext, 0, $get["length"], $get["filter_type"], $get["filter_value"]);

    ?>
    <div style="text-align: right; padding-bottom:10px;">
      <span style="padding-right: 10px">
        <a href="javascript:newest()">&#171; Newest</a>
      </span>
      <span style="padding-right: 10px">
        <a href="javascript:newer()">&#8249; Newer</a>
      </span>
      <span style="padding-right: 10px">
        Showing <span id="showing_from">0</span> - <span id="showing_to"><?echo (min($get["length"], $total_num_results))?></span> of <?echo $total_num_results?>
      </span>
      <span style="padding-right: 10px">
        <a href="javascript:older()">Older &#8250;</a>
      </span>
      <span style="padding-right: 10px">
        <a href="javascript:oldest()">Oldest &#187;</a>
      </span>
    </div>

    <script type="text/javascript">

      var total_num_results = <?echo $total_num_results?>;

      function newest()
      {
        setOffset(0);
        results_update_request();    
      }

      function newer()
      {
        var length = getLength();
        var offset = getOffset();
        if ((offset - length) < 0)
          offset = 0;
        else
          offset -= length;
        setOffset(offset);
        results_update_request();

      }

      function older() 
      {
        var length = getLength();
        var offset = getOffset();
        if ((offset + length > total_num_results) < 0)
          offset = total_num_results - length;
        else
          offset += length;
        setOffset(offset);
        results_update_request();
      }

      function oldest()
      {
        var length = getLength();
        var offset = getOffset();
        if ((total_num_results - length) < 0)
          offset = 0;
        else
          offset = total_num_results - length;
        setOffset(offset);
        results_update_request();
      }

    </script>

    <table width="100%">
      <tr>
<?
      if ($get["inlineimages"] == "true")
        echo "        <th id='IMAGE_TR'>IMAGE</th>\n";
      else 
        echo "        <th id='IMAGE_TR'></th>\n";
?>
        <th>SOURCE</th>
        <th>UTC START</th>
        <th>CFREQ</th>
        <th>BW</th>
        <th>SNR</th>
        <th>LENGTH</th>
        <th>PID</th>
        <th>PROC_FILE</th>
        <th class="trunc">Annotation</th>
      </tr>
<?

        $keys = array_keys($results);
        rsort($keys);
        $result_url = "result.lib.php";
        $results_dir = "results";

        for ($i=0; $i < count($keys); $i++) {

          $k = $keys[$i];
          $r = $results[$k];
          
          #$data = getObservationImages($basedir."/".$keys[$i]);
          
          $url = $result_url."?utc_start=".$k;
          $mousein = "onmouseover=\"Tip('<img src=\'".$results_dir."/".$k."/".$r["IMG"]."\' width=241 height=181>')\"";
          $mouseout = "onmouseout=\"UnTip()\"";
  
          /* If archives have been finalised and its not a brand new obs */
          echo "  <tr id='row_".$i."' class='".(($results[$keys[$i]]["processing"] === 1) ? "processing" : "finished")."'>\n";

          /* IMAGE */
          if ($get["inlineimages"] == "true") 
            $style = "";
          else
            $style = "display: none;";
          echo "    <td><img style='".$style."' id='img_".$i."' src=".$results_dir."/".$k."/".$r["IMG"]." width=64 height=48>\n";
          
          
          /* SOURCE */
          echo "      <td ".$bg_style."><a id='link_".$i."' href='".$url."' ".$mousein." ".$mouseout.">".$r["SOURCE"]."</a></td>\n";

          /* UTC_START */
          echo "    <td ".$bg_style."><span id='UTC_START_".$i."'>".$k."</span></td>\n";

          /* CFREQ */
          echo "    <td ".$bg_style."><span id='CFREQ_".$i."'>".$r["CFREQ"]."</span></td>\n";

          /* BW */
          echo "    <td ".$bg_style."><span id='BANDWIDTH_".$i."'>".$r["BANDWIDTH"]."</span></td>\n";

          /* SNR */
          echo "    <td ".$bg_style."><span id='SNR_".$i."'>".$r["SNR"]."</span></td>\n";

          /* INTERGRATION LENGTH */
          echo "    <td ".$bg_style."><span id='INT_".$i."'>".$r["INT"]."</span></td>\n";

          /* PID */
          echo "    <td ".$bg_style."><span id='PID_".$i."'>".$r["PID"]."</span></td>\n";

          /* PROC_FILE */
          echo "    <td ".$bg_style."><span id='PROC_FILE_".$i."'>".$r["PROC_FILE"]."</span></td>\n";

          /* ANNOTATION */
          echo "    <td ".$bg_style." class=\"trunc\"><div><span id='annotation_".$i."'>".$r["annotation"]."</span></div></td>\n";

          echo "  </tr>\n";

        }
?>
    </table>
<?
    $this->closeBlockHeader();
  }

  /*************************************************************************************************** 
   *
   * Prints raw text to be parsed by the javascript XMLHttpRequest
   *
   ***************************************************************************************************/
  function printUpdateHTML($get)
  {

    $archive_ext = ".ar";

    $results = $this->getResultsArray($this->cfg["SERVER_RESULTS_DIR"], $this->cfg["SERVER_ARCHIVE_DIR"],
                                      $archive_ext, $get["offset"], $get["length"], $get["filter_type"], $get["filter_value"]);

    $keys = array_keys($results);
    rsort($keys);

    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<results>\n";
    for ($i=0; $i<count($keys); $i++) {
      $k = $keys[$i];
      $r = $results[$k];
      $rkeys = array_keys($r);
      $xml .= "<result>\n";
      $xml .= "<UTC_START>".$k."</UTC_START>\n";
      for ($j=0; $j<count($rkeys); $j++) {
        $rk = $rkeys[$j]; 
        if (($rk != "FRES_AR") && ($rk != "TRES_AR")) {
          $xml .= "<".$rk.">".$r[$rk]."</".$rk.">\n";
        }
      }
      $xml .= "</result>\n";
    }
    $xml .= "</results>\n";

    header('Content-type: text/xml');
    echo $xml;

  }

  function handleRequest()
  {

    if ($_GET["update"] == "true") {
      $this->printUpdateHTML($_GET);
    } else {
      $this->printHTML($_GET);
    }

  }

  function getResultsArray($results_dir, $archive_dir, $archive_ext, $offset=0, $length=0, $filter_type, $filter_value) 
  {

    $all_results = array();

    $observations = array();
    $dir = $results_dir;

    if (($filter_type == "") || ($filter_value == "")) {
      $observations = getSubDirs($results_dir, $offset, $length, 1);
    } else {

      # get a complete list
      if ($filter_type == "UTC_START") {
        $cmd = "find ".$results_dir."/*".$filter_value."* -maxdepth 1 -name 'obs.info' -printf '%h\n' | awk -F/ '{print \$NF}' | sort -r";
      } else {
        $cmd = "find ".$results_dir." -maxdepth 2 -type f -name obs.info | xargs grep ".$filter_type." | grep ".$filter_value." | awk -F/ '{print $(NF-1)}' | sort -r";
      }
      $last = exec($cmd, $all_obs, $rval);
      $observations = array_slice($all_obs, $offset, $length);
    }

    for ($i=0; $i<count($observations); $i++) {

      $o = $observations[$i];
      $dir = $results_dir."/".$o;

      /* read the obs.info file into an array */
      if (file_exists($dir."/obs.info")) {
        $arr = getConfigFile($dir."/obs.info");
        $all_results[$o]["SOURCE"] = $arr["SOURCE"];
        $all_results[$o]["CFREQ"] = sprintf("%5.2f",$arr["CFREQ"]);
        $all_results[$o]["BANDWIDTH"] = $arr["BANDWIDTH"];
        $all_results[$o]["PID"] = $arr["PID"];
        $all_results[$o]["PROC_FILE"] = $arr["PROC_FILE"];

        // these will only exist after this page has loaded and the values have been calculated once
        $all_results[$o]["INT"] = (array_key_exists("INT", $arr)) ? $arr["INT"] : "NA";
        $all_results[$o]["SNR"] = (array_key_exists("SNR", $arr)) ? $arr["SNR"] : "NA";
        //$all_results[$o]["OBS.START"] = (array_key_exists("OBS.START", $arr)) ? $arr["OBS.START"] : "NA";
        $all_results[$o]["FRES_AR"] = (array_key_exists("FRES_AR", $arr)) ? $arr["FRES_AR"] : "NA";
        $all_results[$o]["TRES_AR"] = (array_key_exists("TRES_AR", $arr)) ? $arr["TRES_AR"] : "NA";
        $all_results[$o]["IMG"] = (array_key_exists("IMG", $arr)) ? $arr["IMG"] : "NA";
      }

      # if this observation is finished check if any of the observations require updating of the obs.info
      if (file_exists($dir."/obs.finished")) {
        $finished = 1;
      } else {
        $finished = 0;
      }

      # check the obs.start filename
      /*
      if ($all_results[$o]["OBS.START"] == "NA") {

        $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f -name '*obs.start' | tail -n 1";
        $an_obs_start = exec($cmd);
        $all_results[$observations[$i]]["obs_start"] = $an_obs_start;
        if ($finished) {
          system("echo 'OBS.START           ".$an_obs_start."' >> ".$results_dir."/".$o."/obs.info");
          system("echo 'OBS.START           ".$an_obs_start."' >> ".$archive_dir."/".$o."/obs.info");
        }
      }
      */
      # try to find the name of the summed tres/fres archives
      if (($all_results[$o]["FRES_AR"] == "NA") || ($all_results[$o]["TRES_AR"] == "NA")) {

        $tres_archive = "NA";
        $fres_archive = "NA";

        $ars = array();
        $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f -name '*.tot'";
        $last = exec($cmd, $ars, $rval);
        for ($j=0; $j<count($ars); $j++) {
          if (strpos($ars[$j],"_t") !== FALSE)
            $tres_archive = $ars[$j];
          if (strpos($ars[$j],"_f") !== FALSE)
            $fres_archive = $ars[$j];
        }
        $all_results[$o]["FRES_AR"] = $fres_archive;
        $all_results[$o]["TRES_AR"] = $tres_archive;

        if ($finished) {
          if ($all_results[$o]["TRES_AR"] != "NA") {
            system("echo 'TRES_AR             ".$tres_archive."' >> ".$results_dir."/".$o."/obs.info");
            system("echo 'TRES_AR             ".$tres_archive."' >> ".$archive_dir."/".$o."/obs.info");
          }
          if ($all_results[$o]["FRES_AR"] != "NA") {
            system("echo 'FRES_AR             ".$fres_archive."' >> ".$results_dir."/".$o."/obs.info");
            system("echo 'FRES_AR             ".$fres_archive."' >> ".$archive_dir."/".$o."/obs.info");
          }
        }
      }

      # get the integration length 
      if (($all_results[$o]["INT"] == "NA") && ($all_results[$o]["TRES_AR"] != "NA")) {
        $int = instrument::getIntergrationLength($all_results[$o]["TRES_AR"]);
        $all_results[$o]["INT"] = $int;
        if ($finished) {
          system("echo 'INT                ".$int."' >> ".$results_dir."/".$o."/obs.info");
          system("echo 'INT                ".$int."' >> ".$archive_dir."/".$o."/obs.info");
        }
      }

      # get the SNR
      if (($all_results[$o]["SNR"] == "NA") && ($all_results[$o]["FRES_AR"] != "NA")) {
        $snr = instrument::getSNR($all_results[$o]["FRES_AR"]);
        $all_results[$o]["SNR"] = $snr;
        if ($finished) {
          system("echo 'SNR                ".$snr."' >> ".$results_dir."/".$o."/obs.info");
          system("echo 'SNR                ".$snr."' >> ".$archive_dir."/".$o."/obs.info");
        }
      }

      # get the phase vs flux image
      if ($all_results[$o]["IMG"] == "NA") {
        $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f -name 'phase_vs_flux*_240x180.png' -printf '%f\n' -o -name 'phase_vs_flux*_200x150.png' -printf '%f\n' | sort -n | tail -n 1";
        $img = exec($cmd, $output, $rval);
        if (($rval == 0) && ($img != "")) {
          $all_results[$o]["IMG"] = $img;
        } else {
          $all_results[$o]["IMG"] = "../../../images/blankimage.gif";
        }  
        if ($finished) {
          system("echo 'IMG                 ".$img."' >> ".$results_dir."/".$o."/obs.info");
          system("echo 'IMG                 ".$img."' >> ".$archive_dir."/".$o."/obs.info");
        }
      }

      if (file_exists($dir."/obs.txt")) {
        $all_results[$o]["annotation"] = file_get_contents($dir."/obs.txt");
      } else {
        $all_results[$o]["annotation"] = "";
      }

      if (file_exists($dir."/obs.processing")) {
        $all_results[$o]["processing"] = 1;
      } else {
        $all_results[$o]["processing"] = 0;
      }
    }

    return $all_results;
  }

}
$obj = new results();
$obj->handleRequest($_GET);
