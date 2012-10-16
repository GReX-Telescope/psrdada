<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class old_results extends bpsr_webpage 
{

  var $filter_types = array("", "SOURCE", "CFREQ", "BANDWIDTH", "PID", "UTC_START", "PROC_FILE");
  var $cfg = array();
  var $length;
  var $offset;
  var $inline_images;
  var $filter_type;
  var $filter_value;

  function old_results()
  {
    bpsr_webpage::bpsr_webpage();
    $inst = new bpsr();
    $this->cfg = $inst->config;
    $this->title = "BPSR | Archived Results";

    $this->length = (isset($_GET["length"])) ? $_GET["length"] : 20;
    $this->offset = (isset($_GET["offset"])) ? $_GET["offset"] : 0;
    $this->inline_images = (isset($_GET["inline_images"])) ? $_GET["inline_images"] : "false";
    $this->filter_type = (isset($_GET["filter_type"])) ? $_GET["filter_type"] : "";
    $this->filter_value = (isset($_GET["filter_value"])) ? $_GET["filter_value"] : "";
    $this->filter_type = "";
    $this->filter_value = "";
  
  }

  function printJavaScriptHead()
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

        var URL = "old_results.lib.php?single=true&offset="+offset+"&length="+length+"&inline_images="+show_inline;

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

      function old_results_update_request() 
      {
        var ru_http_requset;

        // get the offset 
        var offset = getOffset();
        var length = getLength();

        var url = "old_results.lib.php?update=true&offset="+offset+"&length="+length;

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
          url = URL + "&filter_type="+filter_type+"&filter_value="+filter_value;
        }

        if (window.XMLHttpRequest)
          ru_http_request = new XMLHttpRequest()
        else
          ru_http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        ru_http_request.onreadystatechange = function() 
        {
          handle_old_results_update_request(ru_http_request)
        }

        ru_http_request.open("GET", url, true)
        ru_http_request.send(null)

      }

      function handle_old_results_update_request(xml_request) 
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

                var url = "result.lib.php?single=true&state=old&utc_start="+utc_start;
                var link = document.getElementById("link_"+i);
                var img = value.replace("112x84","400x300");
                link.href = url;
                link.onmouseover = new Function("Tip(\"<img src='old_results/"+utc_start+"/"+img+"' width=401 height=301>\")");
                link.onmouseout = new Function("UnTip()");
                link.innerHTML = this_result["SOURCE"];

                try {
                  document.getElementById("img_"+i).src = "old_results/"+utc_start+"/"+value;
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

  function printHTML() 
  {
?>

    <table cellpadding="10px" width="100%">

      <tr>
        <td width='210px' height='60px'><img src='/bpsr/images/bpsr_logo.png' width='200px' height='60px'></td>
        <td align=left><font size='+2'>Archived Results</font></td>
      </tr>

      <tr>
        <td valign="top" width="200px">
<?
    $this->openBlockHeader("Search Parameters");
?>
    <table>
<!--
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
-->
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
    </table>
  <?
    $this->closeBlockHeader();

    echo "</td><td>\n";

    $this->openBlockHeader("Matching Observations");

    $basedir = $this->cfg["SERVER_OLD_RESULTS_DIR"];
    $cmd = "";

    if (($this->filter_type == "") && ($this->filter_value == "")) {
      $cmd = "ls -1 ".$basedir." | wc -l";
    } else {
      if ($this->filter_type == "UTC_START") {
        $cmd = "find ".$basedir."/*".$this->filter_value."* -maxdepth 1 -name 'obs.info' | wc -l";
      } else {
        $cmd = "find ".$basedir." -maxdepth 2 -type f -name obs.info | xargs grep ".$this->filter_type." | grep ".$this->filter_value." | wc -l";
      }
    }
    $total_num_results = exec($cmd);


    $results = $this->getResultsArray($this->cfg["SERVER_OLD_RESULTS_DIR"], $this->cfg["SERVER_OLD_ARCHIVE_DIR"],  
                                      $this->offset, $this->length, $this->filter_type, $this->filter_value);

    ?>
    <div style="text-align: right; padding-bottom:10px;">
      <span style="padding-right: 10px">
        <a href="javascript:newest()">&#171; Newest</a>
      </span>
      <span style="padding-right: 10px">
        <a href="javascript:newer()">&#8249; Newer</a>
      </span>
      <span style="padding-right: 10px">
        Showing <span id="showing_from"><?echo $this->offset?></span> - <span id="showing_to"><?echo (min($this->length, $total_num_results))?></span> of <?echo $total_num_results?>
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
        old_results_update_request();    
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
        old_results_update_request();

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
        old_results_update_request();
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
        old_results_update_request();
      }

    </script>

    <table width="100%">
      <tr>
<?
      if ($this->inline_images == "true")
        echo "        <th id='IMAGE_TR' align=left>IMAGE</th>\n";
      else 
        echo "        <th id='IMAGE_TR' align=left></th>\n";
?>
        <th align=left>SOURCE</th>
        <th align=left>UTC START</th>
        <th align=left>CFREQ</th>
        <th align=left>BW</th>
        <th align=left>LENGTH</th>
        <th align=left>PID</th>
        <th align=left>PROC_FILE</th>
        <th class="trunc">Annotation</th>
      </tr>
<?

        $keys = array_keys($results);
        rsort($keys);
        $result_url = "result.lib.php";
        $results_dir = "old_results";

        for ($i=0; $i < count($keys); $i++) {

          $k = $keys[$i];
          $r = $results[$k];
          
          $url = $result_url."?single=true&state=old&utc_start=".$k;
          if (strpos($r["IMG"], "pvf") !== FALSE)
            $url .= "&type=pvf";
          else
            $url .= "&type=bp";

          // guess the larger image size
          $image = str_replace("112x84", "400x300", $r["IMG"]);
          if (!file_exists($basedir."/".$k."/".$mid_image))
             $image = $r["IMG"];

          $mousein = "onmouseover=\"Tip('<img src=\'".$results_dir."/".$k."/".$image."\' width=401 height=301>')\"";
          $mouseout = "onmouseout=\"UnTip()\"";
  
          // If archives have been finalised and its not a brand new obs
          echo "  <tr id='row_".$i."' class='".(($results[$keys[$i]]["processing"] === 1) ? "processing" : "finished")."'>\n";

          // IMAGE
          if ($this->inline_images == "true") 
            $style = "";
          else
            $style = "display: none;";
          echo "    <td class='processing'><img style='".$style."' id='img_".$i."' src=".$results_dir."/".$k."/".$r["IMG"]." width=64 height=48>\n";
          
          // SOURCE 
          echo "      <td ".$bg_style."><a id='link_".$i."' href='".$url."' ".$mousein." ".$mouseout.">".$r["SOURCE"]."</a></td>\n";

          // UTC_START 
          echo "    <td ".$bg_style."><span id='UTC_START_".$i."'>".$k."</span></td>\n";

          // CFREQ
          echo "    <td ".$bg_style."><span id='CFREQ_".$i."'>".$r["CFREQ"]."</span></td>\n";

          // BW
          echo "    <td ".$bg_style."><span id='BANDWIDTH_".$i."'>".$r["BANDWIDTH"]."</span></td>\n";

          // INTERGRATION LENGTH
          echo "    <td ".$bg_style."><span id='INT_".$i."'>".$r["INT"]."</span></td>\n";

          // PID
          echo "    <td ".$bg_style."><span id='PID_".$i."'>".$r["PID"]."</span></td>\n";

          // PROC_FILE
          echo "    <td ".$bg_style."><span id='PROC_FILE_".$i."'>".$r["PROC_FILE"]."</span></td>\n";

          // ANNOTATION
          echo "    <td ".$bg_style." class=\"trunc\"><div><span id='annotation_".$i."'>".$r["annotation"]."</span></div></td>\n";

          echo "  </tr>\n";

        }
?>
    </table>
<?
    $this->closeBlockHeader();

    echo "</td></tr></table>\n";
  }

  /*************************************************************************************************** 
   *
   * Prints raw text to be parsed by the javascript XMLHttpRequest
   *
   ***************************************************************************************************/

  function printUpdateHTML($get)
  {

    $results = $this->getResultsArray($this->cfg["SERVER_OLD_RESULTS_DIR"], $this->cfg["SERVER_OLD_ARCHIVE_DIR"],
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

  function getResultsArray($results_dir, $archive_dir, $offset=0, $length=0, $filter_type, $filter_value) 
  {

    $all_results = array();

    $observations = array();
    $dir = $results_dir;

    if (($filter_type == "") || ($filter_value == "")) {
      $observations = getSubDirs($results_dir, $offset, $length, 1);
    } else {

      echo "Filter?? should be disabled for old results<BR>\n";
      flush();
      exit(0);
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
        $all_results[$o]["IMG"] = (array_key_exists("IMG", $arr)) ? $arr["IMG"] : "NA";
      }

      # if this observation is finished check if any of the observations require updating of the obs.info
      if (file_exists($dir."/obs.finished")) {
        $finished = 1;
      } else {
        $finished = 0;
      }

      # get the image bp or pvf 
      if (($all_results[$o]["IMG"] == "NA") || ($all_results[$o]["IMG"] == "")) {
        $cmd = "find ".$dir." -mindepth 2 -maxdepth 2 -type f -name '*.bp_112x84.png' -printf '%h/%f\n' -o -name '*.pvf_112x84.png' -printf '%h/%f\n' | sort -n | head -n 1 | awk -F/ '{print $(NF-1)\"/\"\$NF}'";
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

      # get the integration length 
      if (($all_results[$o]["INT"] == "NA") || ($all_results[$o]["INT"] <= 0)) {

        if (($all_results[$o]["IMG"] != "NA") && ($all_results[$o]["IMG"] != "") &&
            (strpos($all_results[$o]["IMG"], "blank") === FALSE)) 
        {
          $int = $this->calcIntLength($o, $all_results[$o]["IMG"]);
          $all_results[$o]["INT"] = $int;
        } else {
          $all_results[$o]["INT"] = "NA";
        } 
        if ($finished) {
          system("echo 'INT                 ".$int."' >> ".$results_dir."/".$o."/obs.info");
          system("echo 'INT                 ".$int."' >> ".$archive_dir."/".$o."/obs.info");
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

  function calcIntLength($utc_start, $image) 
  {
    $image = substr($image, 3);
    $array = split("\.",$image);
    $image_utc = $array[0];

    $offset = 0;
    if (strpos($image, "pvf") !== FALSE) 
      $offset = (11*60*60);

    # add ten as the 10 second image file has a UTC referring to the first byte of the file 
    $length = (unixTimeFromGMTime($image_utc)+(10-$offset)) - unixTimeFromGMTime($utc_start);

    return $length;
  }


}

handledirect("old_results");
