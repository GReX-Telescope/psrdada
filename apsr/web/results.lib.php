<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class results extends apsr_webpage 
{

  var $filter_types = array("", "SOURCE", "CFREQ", "BANDWIDTH", "PID", "UTC_START", "PROC_FILE", "MODE");
  var $cfg = array();
  var $length;
  var $offset;
  var $inline_images;
  var $filter_type;
  var $filter_value;

  function results()
  {
    apsr_webpage::apsr_webpage();
    $inst = new apsr();
    $this->cfg = $inst->config;

    $this->length = (isset($_GET["length"])) ? $_GET["length"] : 20;
    $this->offset = (isset($_GET["offset"])) ? $_GET["offset"] : 0;
    $this->inline_images = (isset($_GET["inline_images"])) ? $_GET["inline_images"] : "false";
    $this->filter_type = (isset($_GET["filter_type"])) ? $_GET["filter_type"] : "";
    $this->filter_value = (isset($_GET["filter_value"])) ? $_GET["filter_value"] : "";
  
  }

  function printJavaScriptHead()
  {
?>
    <style type="text/css">
 
      table.resultstable td {
        text-align: left;
        padding-left: 5px;
        padding-right: 5px;
      }
      
      .hidden {
        display: none;
      }
      .processing {
        background-color: #FFFFFF;
      }

      .failed {
        background-color: #FFaaaa;
      }

      .finished {
        background-color: #cae2ff;
      }

      .transferred{
        background-color: #caffe2;
      }

      .deleted{
        background-color: #cccccc;
      }

      .unknown{
        background-color: #dddddd;
      }
    </style>

    <script type='text/javascript'>

      var offset = <?echo $this->offset?>;
  
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

      // If a page reload is required
      function changeLength() {

        var i = document.getElementById("displayLength").selectedIndex;
        var length = document.getElementById("displayLength").options[i].value;

        var show_inline;
        if (document.getElementById("inline_images").checked)
          show_inline = "true";
        else
          show_inline = "false";

        var offset = getOffset()

        var URL = "results.lib.php?single=true&offset="+offset+"&length="+length+"&inline_images="+show_inline;

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

        if (document.getElementById("inline_images").checked) {
          show_inline = true;
          document.getElementById("IMAGE_TR").innerHTML = "IMAGE";
        } else {
          show_inline = false;
          document.getElementById("IMAGE_TR").innerHTML = "";
        }

        for (i=0; i<length; i++) {
          img = document.getElementById("img_"+i);
          if (show_inline) 
          {
            img.style.display = "";
            img.className = "processing";
          }
          else
          {
            img.style.display = "none";
          }
        }
      }

      function set_retreiving_data()
      {
        var i = document.getElementById("displayLength").selectedIndex;
        var length = document.getElementById("displayLength").options[i].value;
        for (i=0; i<length; i++)
        {
          document.getElementById("row_"+i).className = "hidden";
        }
        document.getElementById("status").innerHTML = "Retreiving data...";
      }

      function results_update_request() 
      {
        var ru_http_requset;

        set_retreiving_data(); 

        // get the offset 
        var offset = getOffset();
        var length = getLength();

        var url = "results.lib.php?update=true&offset="+offset+"&length="+length;

        // check if inline images has been specified
        if (document.getElementById("inline_images").checked)
          show_inline = "true";
        else
          show_inline = "false";
        url = url + "&inline_images="+show_inline;
        
        // check if a filter has been specified
        var u, filter_type, filter_vaule;
        i = document.getElementById("filter_type").selectedIndex;
        filter_type = document.getElementById("filter_type").options[i].value;
        filter_value = document.getElementById("filter_value").value;
        if ((filter_value != "") && (filter_type != "")) {
          url = url + "&filter_type="+filter_type+"&filter_value="+filter_value;
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

                var url = "result.lib.php?single=true&utc_start="+utc_start;
                var link = document.getElementById("link_"+i);
                var img = value;
                link.href = url;
                link.onmouseover = new Function("Tip(\"<img src='results/"+utc_start+"/"+img+"' width=241 height=181>\")");
                link.onmouseout = new Function("UnTip()");
                link.innerHTML = this_result["SOURCE"];

                try {
                  document.getElementById("img_"+i).src = "results/"+utc_start+"/"+value;
                } catch (e) {
                  // do nothing
                }

              } else if (key == "state") {
                  document.getElementById("row_"+i).className = value;
  
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
          document.getElementById("status").innerHTML = "Matching Observations";
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

  function printHTML() 
  {
?>
    <table cellspacing=0 cellpadding=0 border=0 width="100%">
      <tr>
        <td width='380px' height='60px'><img src='/apsr/images/apsr_logo.png' width='380px' height='60px'></td>
        <td height="60px" align=left style='padding-left: 20px'>
          <span class="largetext">Recent Results</span>
        </td>
      </tr>
    </table>

    <table cellspacing=0 cellpadding="10px" border=0 width="100%">
      <tr>
        <td valign="top" width="150px">
<?
    $this->openBlockHeader("Search Parameters");
?>
    <table class="resultstable">
      <tr>
        <td>Filter</td>
        <td>
          <select name="filter_type" id="filter_type">
<?
          for ($i=0; $i<count($this->filter_types); $i++) {
            $t = $this->filter_types[$i];
            echoOption($t, $t, FALSE, $this->filter_type);
          }
?>
          </select>
        </td>
      </tr>
      <tr>
        <td>For</td>
        <td><input name="filter_value" id="filter_value" value="<?echo $this->filter_value?>" onChange="changeLength()"></td>
      </tr>

      <tr>
        <td>Images</td>
        <td>
          <input type=checkbox id="inline_images" name="inline_images" onChange="toggle_images()"<? if($this->inline_images == "true") echo " checked";?>>
        </td>
      </tr>

      <tr>
        <td>Show</td>
        <td>
          <select name="displayLength" id="displayLength" onChange='changeLength()'>
<?
        echoOption("20", "20", FALSE, $this->length);
        echoOption("50", "50", FALSE, $this->length);
        echoOption("100", "100", FALSE, $this->length);
?>
          </select>
        </td>
      </tr>
      <tr>
        <td colspan=2><a href="/apsr/old_results.lib.php?single=true">Archived APSR Results</a></td>
      </tr>

    </table>
<?
    $this->closeBlockHeader();
?>

    <br/>

<?
    $this->openBlockHeader("Legend");
?>

    <table>
      <tr class="processing">
        <td>Processing</td>
        <td>The observation is currently recording</td>
      </tr>
      <tr>
        <td class="failed">Failed</td>
        <td>A failure has ocurred</td>
      </tr>
      <tr>
        <td class="finished">Finished</td>
        <td>The obs has been completed</td>
      </tr>
      <tr>
        <td class="transferred">Transferred</td>
        <td>To remote storage disk</td>
      </tr>
      <tr>
        <td class="deleted">Deleted</td>
        <td>From local disk storage</td>
      </tr>
    </table>
<?
    $this->closeBlockHeader();

?>
        </td>
        <td valign="top" width="700px">
<?
    $this->openBlockHeader("<span id='status'>Retreiving Data</span>");

    $basedir = $this->cfg["SERVER_RESULTS_DIR"];
    $cmd = "";

    if (($this->filter_type == "") && ($this->filter_value == "")) {
      $cmd = "find ".$basedir." -maxdepth 2 -name 'obs.info' | wc -l";
    } else {
      if ($this->filter_type == "UTC_START") {
        $cmd = "find ".$basedir."/*".$this->filter_value."* -maxdepth 1 -name 'obs.info' | wc -l";
      } else {
        $cmd = "find ".$basedir." -maxdepth 2 -type f -name obs.info | xargs grep ".$this->filter_type." | grep ".$this->filter_value." | wc -l";
      }
    }
    $total_num_results = exec($cmd);

    $results = array();
    for ($i=0; $i<$this->length; $i++)
    {
      $result = array("UTC_START" => "NOT SET");
      $results[$i] = $result;
    } 
/*
    $results = $this->getResultsArray($this->cfg["SERVER_RESULTS_DIR"], $this->cfg["SERVER_ARCHIVE_DIR"],  
                                      $this->offset, $this->length, $this->filter_type, $this->filter_value);
*/

    ?>
    <div style="text-align: right; padding-bottom:10px;">
      <span style="padding-right: 10px">
        <a href="javascript:newest()">&#171; Newest</a>
      </span>
      <span style="padding-right: 10px">
        <a href="javascript:newer()">&#8249; Newer</a>
      </span>
      <span style="padding-right: 10px">
        Showing <span id="showing_from"><?echo $this->offset?></span> - <span id="showing_to"><?echo (min($this->length, $total_num_results))?></span> of <span id="showing_total"><?echo $total_num_results?></span>
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

    <table width="100%" class='resultstable'>
      <tr>
<?
      if ($this->inline_images == "true")
        echo "        <th id='IMAGE_TR' align=left>IMAGE</th>\n";
      else 
        echo "        <th id='IMAGE_TR' align=left></th>\n";
?>
        <th align=left>SOURCE</th>
        <th align=left>UTC START</th>
        <th align=left>MODE</th>
        <th align=left>CFREQ</th>
        <th align=left>BW</th>
        <th align=left>LENGTH</th>
        <th align=left>SNR</th>
        <th align=left>PID</th>
        <th align=left>PROC_FILE</th>
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
          
          $url = $result_url."?single=true&utc_start=".$k;
          $img = $r["IMG"];

          $mousein = "onmouseover=\"Tip('<img src=\'".$results_dir."/".$k."/".$img."\' width=241 height=181>')\"";
          $mouseout = "onmouseout=\"UnTip()\"";
  
          // If archives have been finalised and its not a brand new obs
          echo "  <tr id='row_".$i."' class='hidden'>\n";

          // IMAGE
          if ($this->inline_images == "true") 
            $style = "";
          else
            $style = "display: none;";
          echo "    <td><img style='".$style."' id='img_".$i."' src='' width=64 height=48></td>\n";
          
          // SOURCE 
          echo "    <td><a id='link_".$i."' href=''></a></td>\n";

          // UTC_START 
          echo "    <td><span id='UTC_START_".$i."'></span></td>\n";

          // MODE
          echo "    <td><span id='MODE_".$i."'></span></td>\n";

          // CFREQ
          echo "    <td><span id='CFREQ_".$i."'></span></td>\n";

          // BW
          echo "    <td><span id='BANDWIDTH_".$i."'></span></td>\n";

          // INTERGRATION LENGTH
          echo "    <td><span id='INT_".$i."'></span></td>\n";

          // SNR
          echo "    <td><span id='SNR_".$i."'></span></td>\n";

          // PID
          echo "    <td><span id='PID_".$i."'></span></td>\n";

          // PROC_FILE
          echo "    <td><span id='PROC_FILE_".$i."'></span></td>\n";

          // ANNOTATION
          echo "    <td class=\"trunc\"><div><span id='annotation_".$i."'></span></div></td>\n";

          echo "  </tr>\n";

        }
?>
    </table>
<?
    $this->closeBlockHeader();
?>
        </td>
      </tr>
    </table>

    <script type='text/javascript'>
      results_update_request();
    </script>
<?
  }

  /*************************************************************************************************** 
   *
   * Prints raw text to be parsed by the javascript XMLHttpRequest
   *
   ***************************************************************************************************/
  function printUpdateHTML($get)
  {

    $results = $this->getResultsArray($this->cfg["SERVER_RESULTS_DIR"], $this->cfg["SERVER_ARCHIVE_DIR"],
                                      $this->offset, $this->length, $this->filter_type, $this->filter_value);

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
        $xml .= "<".$rk.">".$r[$rk]."</".$rk.">\n";
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

  function getResultsArray($results_dir, $archive_dir, $offset=0, $length=0, $filter_type, $filter_value) 
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
      $adir = $archive_dir."/".$o;
      $update_config = false;

      /* read the obs.info file into an array */
      if (file_exists($dir."/obs.info")) {
        $arr = getConfigFile($dir."/obs.info");
        $all_results[$o]["SOURCE"] = $arr["SOURCE"];
        $all_results[$o]["CFREQ"] = sprintf("%5.0f",$arr["CFREQ"]);
        $all_results[$o]["BANDWIDTH"] = sprintf("%5.0f",$arr["BANDWIDTH"]);
        $all_results[$o]["PID"] = $arr["PID"];
        $all_results[$o]["PROC_FILE"] = $arr["PROC_FILE"];

        // these will only exist after this page has loaded and the values have been calculated once
        $all_results[$o]["INT"] = (array_key_exists("INT", $arr)) ? $arr["INT"] : "NA";
        $all_results[$o]["SNR"] = (array_key_exists("SNR", $arr)) ? $arr["SNR"] : "NA";
        $all_results[$o]["FRES_AR"] = (array_key_exists("FRES_AR", $arr)) ? $arr["FRES_AR"] : "NA";
        $all_results[$o]["TRES_AR"] = (array_key_exists("TRES_AR", $arr)) ? $arr["TRES_AR"] : "NA";
        $all_results[$o]["IMG"] = (array_key_exists("IMG", $arr)) ? $arr["IMG"] : "NA";
        $all_results[$o]["MODE"] = (array_key_exists("MODE", $arr)) ? $arr["MODE"] : "";
      }

      # if this observation is finished check if any of the observations require updating of the obs.info
      $finished = 0;
      if (file_exists($adir."/obs.finished")) {
        $all_results[$o]["state"] = "finished";
        $finished = 1;
      } else if (file_exists($dir."/obs.processing"))  {
        $all_results[$o]["state"] = "processing";
      } else if (file_exists($dir."/obs.failed")) {
        $all_results[$o]["state"] = "failed";
      } else if (file_exists($adir."/obs.transferred")) {
        $all_results[$o]["state"] = "transferred";
        $finished = 1;
      } else if (file_exists($adir."/obs.deleted")) {
        $all_results[$o]["state"] = "deleted";
        $finished = 1;
      } else {
        $all_results[$o]["state"] = "unknown";
      }


      # find the name of the summed tres/fres archives
      if (($all_results[$o]["FRES_AR"] == "NA") || ($all_results[$o]["TRES_AR"] == "NA")) 
      {
        $tres_archive = "NA";
        $fres_archive = "NA";

        $ars = array();
        $cmd = "find ".$dir." -maxdepth 1 -name '*.ar'";
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
            $arr["TRES_AR"] = $tres_archive;
            $update_config = true;
          }
          if ($all_results[$o]["FRES_AR"] != "NA") {
            $arr["FRES_AR"] = $fres_archive;
            $update_config = true;
          } 
        }
      }

      # get the most recent phase vs flux image
      if (($all_results[$o]["IMG"] == "NA") || ($all_results[$o]["IMG"] == "")) {
        $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f -name 'phase_vs_flux_*_240x180.png' -printf '%h/%f\n' | sort -n | tail -n 1 | awk -F/ '{print \$NF}'";
        $img = exec($cmd, $output, $rval);
        if (($rval == 0) && ($img != "")) {
          $all_results[$o]["IMG"] = $img;
        } else {
          $all_results[$o]["IMG"] = "../../../images/blankimage.gif";
        }
        if ($finished) {
          $arr["IMG"] = $img;
          $update_config = true;
        }
      }

      # get the integration length 
      if ((($all_results[$o]["INT"] == "NA") || ($all_results[$o]["INT"] <= 0)) && ($all_results[$o]["TRES_AR"] != "NA")) {
        $int = instrument::getIntergrationLength($all_results[$o]["TRES_AR"]);
        $all_results[$o]["INT"] = $int;
        if ($finished) {
          $arr["INT"] = $int;
          $update_config = true;
        }
      }

      # get the SNR
      if ((($all_results[$o]["SNR"] == "NA") || ($all_results[$o]["SNR"] <= 0)) && ($all_results[$o]["FRES_AR"] != "NA")) {
        $snr = instrument::getSNR($all_results[$o]["FRES_AR"]);
        $all_results[$o]["SNR"] = $snr;
        if ($finished) {
          $arr["SNR"] = $snr;
          $update_config = true;
        }
      }

      # get the mode
      if ($all_results[$o]["MODE"] == "")
      {
        $cmd = "grep MODE `find ".$dir." -mindepth 2 -maxdepth 2 -name 'obs.start' | head -n 1` | awk '{print \$2}'";
        $mode = exec($cmd, $output, $rval);
        if (($rval == 0) && ($mode != ""))
        {
          $all_results[$o]["MODE"] = $mode;
        }
        if ($finished)
        {
          $arr["MODE"] = $mode;
          $update_config = true;
        }
      }


      if (file_exists($dir."/obs.txt")) {
        $all_results[$o]["annotation"] = file_get_contents($dir."/obs.txt");
      } else {
        $all_results[$o]["annotation"] = "";
      }

      # update the config file with any new changes
      if ($update_config && (file_exists($dir."/obs.info")))
      {
        updateConfigFile($dir."/obs.info", $arr);
      }
    }

    return $all_results;
  }
}

handledirect("results");
