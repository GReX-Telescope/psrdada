<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class results extends mopsr_webpage 
{

  var $filter_types = array("", "SOURCE", "FREQ", "BW", "UTC_START", "PROC_FILE");
  var $cfg = array();
  var $length;
  var $offset;
  var $inline_images;
  var $filter_type;
  var $filter_value;
  var $class = "new";
  var $results_dir;
  var $results_link;
  var $archive_dir;

  function results()
  {
    mopsr_webpage::mopsr_webpage();
    $inst = new mopsr();
    $this->cfg = $inst->config;

    $this->length = (isset($_GET["length"])) ? $_GET["length"] : 20;
    $this->offset = (isset($_GET["offset"])) ? $_GET["offset"] : 0;
    $this->inline_images = (isset($_GET["inline_images"])) ? $_GET["inline_images"] : "false";
    $this->filter_type = (isset($_GET["filter_type"])) ? $_GET["filter_type"] : "";
    $this->filter_value = (isset($_GET["filter_value"])) ? $_GET["filter_value"] : "";
    $this->results_dir = $this->cfg["SERVER_RESULTS_DIR"];
    $this->archive_dir = $this->cfg["SERVER_ARCHIVE_DIR"];
    $this->results_link = "/mopsr/results";
    $this->results_title = "Recent Results";
    $this->class = (isset($_GET["class"])) ? $_GET["class"] : "new";
    if ($this->class == "old")
    {
      $this->results_dir = $this->cfg["SERVER_OLD_RESULTS_DIR"];
      $this->archive_dir = $this->cfg["SERVER_OLD_ARCHIVE_DIR"];
      $this->results_link = "/mopsr/old_results";
      $this->results_title = "Archived Results";
    }
  }

  function printJavaScriptHead()
  {
?>
    <style type="text/css">
      .processing {
        background-color: #FFFFFF;
        padding-right: 10px;
      }

      .finished {
        background-color: #cae2ff;
      }

      .transferred {
        background-color: #caffe2;
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

        var filename = window.location.pathname;
        filename = filename.substring(filename.lastIndexOf('/')+1);
        var url = filename + "?single=true&offset="+offset+"&length="+length+"&inline_images="+show_inline;

        i = document.getElementById("filter_type").selectedIndex;
        var filter_type = document.getElementById("filter_type").options[i].value;

        var filter_value = encodeURIComponent(document.getElementById("filter_value").value);

        if ((filter_value != "") && (filter_type != "")) {
          url = url + "&filter_type="+filter_type+"&filter_value="+filter_value;
        }
        document.location = url;
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

      function results_update_request() 
      {
        var ru_http_requset;

        // get the offset 
        var offset = getOffset();
        var length = getLength();

        var filename = window.location.pathname;
        filename = filename.substring(filename.lastIndexOf('/')+1);
        var url = filename + "?update=true&offset="+offset+"&length="+length+"&class=<?echo $this->class?>";

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

            // row ID
            rid = "row_" + i
  
            result = results[i];
            this_result = new Array();
            
            this_result["ANNOTATION"] = "";
            this_result["SOURCE"] = "";

            for (j=0; j<result.childNodes.length; j++) 
            {
              // if the child node is an element
              if (result.childNodes[j].nodeType == 1) {
                key = result.childNodes[j].nodeName;
                if (result.childNodes[j].childNodes.length == 1) {
                  val = result.childNodes[j].childNodes[0].nodeValue;
                  if (key == "UTC_START") {
                    this_result[key] = val
                  }
                  else if (key == "SOURCE") {
                    name = result.childNodes[j].getAttribute("name")
                    if (this_result[key] == '')
                      this_result[key] = name
                    else
                      this_result[key] = this_result[key] + " " + name
                  }
                  else 
                  {
                    this_result[key] = val;
                  }
                }
              }
            }

            for ( key in this_result) {
              utc_start = this_result["UTC_START"]
              value = this_result[key];
              rid = "row_" + i

              if (key == "UTC_START") {

              } else if (key == "IMG") {
                var _class, _results_link;
                if (this_result["ANNOTATION"].startsWith("<i>Data in old_results.</i>")) {
                  _class = "old";
                  _results_link = "/mopsr/old_results";
                } else {
                  _class = "new";
                  _results_link = "/mopsr/results";
                }

                var url = "result.lib.php?single=true&utc_start="+utc_start+"&class="+_class;
                var link = document.getElementById("link_"+i);
                var img = value.replace("112x84","400x300");
                link.href = url;
                link.onmouseover = new Function("Tip(\"<img src='"+_results_link+"/"+utc_start+"/"+img+"' width=401 height=301>\")");
                link.onmouseout = new Function("UnTip()");
                link.innerHTML = this_result["UTC_START"];

                try {
                  document.getElementById("img_"+i).src = _results_link+"/"+utc_start+"/"+value;
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
                  alert ("failed key="+key+" value="+value);
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
        <td width='210px' height='60px'><img src='/mopsr/images/mopsr_logo.png' width='200px' height='60px'></td>
        <td align=left><font size='+2'><?echo $this->results_title?></font></td>
      </tr>

      <tr>
        <td valign="top" width="200px">
<?
    $this->openBlockHeader("Search Parameters");
?>
    <table>
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
        <td colspan=2><a href="/mopsr/<?php echo basename(__FILE__);?>?single=true">Recent MOPSR Results</a></td>
      </tr>
      <tr>
        <td colspan=2><a href="/mopsr/<?php echo basename(__FILE__);?>?single=true&class=old">Archived MOPSR Results</a></td>
      </tr>

    </table>
  <?
    $this->closeBlockHeader();

    echo "</td><td>\n";

    $this->openBlockHeader("Matching Observations");

    $cmd = "";
    $using_db = $this->filter_type === "SOURCE" && strpos($this->filter_value, "J") === 0;
    if (($this->filter_type == "") && ($this->filter_value == "")) {
      $cmd = "find ".$this->results_dir." -maxdepth 2 -name 'obs.info' | wc -l";
    } else {
      if ($this->filter_type == "UTC_START") {
        $cmd = "find ".$this->results_dir."/*".$this->filter_value."* -maxdepth 1 -name 'obs.info' | wc -l";
      } elseif ($using_db) {
        # Handle requests for pulsar observations using the asteria database
        include MYSQL_DB_CONFIG_FILE;

        $pdo = new PDO ('mysql:dbname='.MYSQL_DB.';host='.MYSQL_HOST, MYSQL_USER, MYSQL_PWD);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $q = 'SELECT count(*) FROM (Pulsars JOIN TB_Obs ON Pulsars.id = TB_Obs.psr_id AND Pulsars.name LIKE "'.$this->filter_value.'%")';
        try {
          $stmt = $pdo -> query($q);
        } catch (PDOException $ex) {
          print $ex->getMessage();
        }
        $row = $stmt->fetch();
        $total_num_results = $row[0];
      } else {
        $cmd = "find ".$this->results_dir." -maxdepth 2 -type f -name obs.info | xargs grep ".$this->filter_type." | grep ".$this->filter_value." | wc -l";
      }
    }
    if (!$using_db) {
      $total_num_results = exec($cmd);
      $results = $this->getResultsArray($this->results_dir,
                                        $this->offset, $this->length, 
                                        $this->filter_type, $this->filter_value);
    } else {
      $results = $this->getResultsArray_db($pdo, $this->results_dir,
                                           $this->offset, $this->length, 
                                           $this->filter_type, $this->filter_value);
      $pdo = null;
    }


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
      if ($this->inline_images == "true")
        echo "        <th id='IMAGE_TR' align=left>IMAGE</th>\n";
      else 
        echo "        <th id='IMAGE_TR' align=left></th>\n";
?>
        <th align=left>UTC START</th>
        <th align=left>SOURCEs</th>
        <th align=left>STATE</th>
        <th align=left>PID</th>
        <th align=left>LENGTH</th>
        <th class="trunc">ANNOTATION</th>
      </tr>
<?

        $keys = array_keys($results);
        rsort($keys);
        $result_url = "result.lib.php";

        for ($i=0; $i < count($keys); $i++) 
        {
          $k = $keys[$i];
          $r = $results[$k];

          if (strpos($r["ANNOTATION"], '<i>Data in old_results.</i>') === 0) {
            $_results_link = "/mopsr/old_results";
            $_class = "old";
          } else {
            $_results_link = "/mopsr/results";
            $_class = "new";
          }

          $url = $result_url."?single=true&utc_start=".$k."&class=".$_class;

          // guess the larger image size
          $image = $r["IMG"];

          #$mousein = "onmouseover=\"Tip('<img src=\'".$this->results_link."/".$k."/".$image."\' width=201 height=151>')\"";
          #$mouseout = "onmouseout=\"UnTip()\"";
  
          // If archives have been finalised and its not a brand new obs
          echo "  <tr id='row_".$i."' class='".(($results[$keys[$i]]["processing"] === 1) ? "processing" : "finished")."'>\n";

          // IMAGE
          if ($this->inline_images == "true") 
            $style = "";
          else
            $style = "display: none;";

          $bg_style = "class='processing'";

          echo "    <td ".$bg_style."><img style='".$style."' id='img_".$i."' src=".$_results_link."/".$k."/".$r["IMG"]." width=64 height=48>\n";
          
          // UTC_START 
          echo "    <td ".$bg_style."><span id='UTC_START_".$i."'><a id='link_".$i."' href='".$url."'>".$k."</a></span></td>\n";

          $sources = "";

          // Cast to array to handle cases when no SOURCES column was set (i.e., data in DB only)
          foreach ((array)$r["SOURCES"] as $s => $v)
          {
            $sources .= $s." ";
/*
            if ($v["TYPE"] == "CORR")
              $sources .= $this->tippedImage ($i, $url, $v["IMAGE"], "purple", $s);
            if ($v["TYPE"] == "TB")
              $sources .= $this->tippedImage ($i, $url, $v["IMAGE"], "blue", $s);
            if ($v["TYPE"] == "FB")
              $sources .= $this->tippedImage ($i, $url, $v["IMAGE"], "green", $s);
*/
          }
  

          // SOURCE 
          echo "    <td ".$bg_style."><span id='SOURCE_".$i."'>".$sources."</span></td>\n";

          // Observation State
          echo "    <td ".$bg_style."><span id='STATE_".$i."'>".$r["STATE"]."</span></td>\n";

          // PID 
          echo "    <td ".$bg_style."><span id='PID_".$i."'>".$r["PID"]."</span></td>\n";

          //echo "    <td ".$bg_style."><a id='link_".$i."' href='".$url."' ".$mousein." ".$mouseout.">".$r["SOURCE"]."</a></td>\n";

          // CONFIG
          //echo "    <td ".$bg_style."><span id='CONFIG_".$i."'>".$r["CONFIG"]."</span></td>\n";

          // FREQ
          //echo "    <td ".$bg_style."><span id='FREQ_".$i."'>".$r["FREQ"]."</span></td>\n";

          // BW
          //echo "    <td ".$bg_style."><span id='BW_".$i."'>".$r["BW"]."</span></td>\n";

          // INTERGRATION LENGTH
          echo "    <td ".$bg_style."><span id='INT_".$i."'>".$r["INT"]."</span></td>\n";

          // PROC_FILEs
          //if (array_key_exists("AQ_PROC_FILE", $r))
          //{
          //  echo "    <td ".$bg_style.">";
          //    echo "<span id='AQ_PROC_FILE_".$i."'>".$r["AQ_PROC_FILE"]."</span> ";
          //    echo "<span id='BF_PROC_FILE_".$i."'>".$r["BF_PROC_FILE"]."</span> ";
          //    echo "<span id='BP_PROC_FILE_".$i."'>".$r["BP_PROC_FILE"]."</span>";
          //  echo "</td>\n";
         // }
          //else
         // {
          //  echo "    <td ".$bg_style."><span id='PROC_FILE_".$i."'>".$r["PROC_FILE"]."</span></td>\n";
          //}

          // ANNOTATION
          echo "    <td ".$bg_style." class=\"trunc\"><div><span id='ANNOTATION_".$i."'>".$r["ANNOTATION"]."</span></div></td>\n";

          echo "  </tr>\n";

        }
?>
    </table>
<?
    $this->closeBlockHeader();

    echo "</td></tr></table>\n";
  }

  function tippedimage($i, $url, $image, $color, $text)
  {
    $mousein = "onmouseover=\"Tip('<img src=\'".$this->results_link."/".$image."\' width=201 height=151>')\"";
    $mouseout = "onmouseout=\"UnTip()\"";
    $link = "<a id='link_".$i."' href='".$url."' ".$mousein." ".$mouseout."><font color='".$color."'>".$text."</font></a>&nbsp;&nbsp;";
    return $link;
  }

  /*************************************************************************************************** 
   *
   * Prints raw text to be parsed by the javascript XMLHttpRequest
   *
   ***************************************************************************************************/
  function printUpdateHTML($get)
  {
    $using_db = $this->filter_type =="SOURCE" && strpos($this->filter_value, "J") === 0;
    if (!$using_db) {
      $results = $this->getResultsArray($this->results_dir,
                                        $this->offset, $this->length, 
                                        $this->filter_type, $this->filter_value);
    } else {
      $pdo = new PDO ('sqlite:/home/dada/linux_64/web/mopsr/asteria.db');
      $results = $this->getResultsArray_db($pdo, $this->results_dir,
                                        $this->offset, $this->length, 
                                        $this->filter_type, $this->filter_value);
      $pdo = null;
    }
    $keys = array_keys($results);
    rsort($keys);

    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<results>\n";
    foreach ($results as $utc => $array)
    {
      $xml .= "<result>\n";
      $xml .= "<UTC_START>".$utc."</UTC_START>\n";
      foreach ($array as $k => $v)
      {
        if ($k == "SOURCE")
        {
          // ignore
        }
        else if ($k == "SOURCES")
        {
          foreach ($v as $source => $vv)
          {
            $xml .= "<SOURCE name='".$source."'>\n";
            if (is_array($vv))
            {
              foreach ($vv as $kkk => $vvv)
              {
                if ($kkk == "IMAGE")
                {
                  $xml .= $vvv;
                }
              }
            }
            $xml .= "</SOURCE>";
          }
        }
        else
        {
          $xml .= "<".$k.">".htmlspecialchars($v)."</".$k.">\n";
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

  function getResultsArray($results_dir, $offset=0, $length=0, $filter_type, $filter_value) 
  {
    $all_results = array();

    $observations = array();
    $dir = $results_dir;

    if (($filter_type == "") || ($filter_value == "")) 
    {
      $observations = getSubDirs ($results_dir, $offset, $length, 1);
    } 
    else 
    {
      # get a complete list
      if ($filter_type == "UTC_START") {
        $cmd = "find ".$results_dir."/*".$filter_value."* -maxdepth 1 ".
               "-name 'obs.info' -printf '%h\n' | awk -F/ '{print \$NF}' | sort -r";
      } else {
        $cmd = "find ".$results_dir." -maxdepth 2 -type f -name obs.info ".
               "| xargs grep ".$filter_type." | grep ".$filter_value." ".
               "| awk -F/ '{print $(NF-1)}' | sort -r";
      }
      $last = exec($cmd, $all_obs, $rval);
      $observations = array_slice($all_obs, $offset, $length);
    }

    for ($i=0; $i<count($observations); $i++)
    {
      $o = $observations[$i];
      $dir = $results_dir."/".$o;

      // read the obs.info file into an array 
      if (file_exists($dir."/obs.info")) 
      {
        $arr = getConfigFile($dir."/obs.info");
        $all = array();

        $all["STATE"] = "unknown";
        if (file_exists($dir."/obs.processing"))
          $all["STATE"] = "processing";
        else if (file_exists($dir."/obs.finished"))
          $all["STATE"] = "finished";
        else if (file_exists($dir."/obs.transferred"))
          $all["STATE"] = "transferred";
        else if (file_exists($dir."/obs.completed"))
          $all["STATE"] = "completed";
        else if (file_exists($dir."/obs.failed"))
          $all["STATE"] = "failed";
        else
           $all["STATE"] = "unknown";

        $all["SOURCE"] = $arr["SOURCE"];

        $all["SOURCES"] = array();
        if ($arr["FB_ENABLED"] == "true")
        {
          $all["SOURCES"]["FB"] = array();
          $all["SOURCES"]["FB"]["TYPE"] = "FB";
          $all["SOURCES"]["FB"]["IMAGE"] = $this->getFBImage($dir, $o, $arr["FB_IMG"]);
          if ((($all["STATE"] == "finished") || ($all["STATE"] == "transferred")) && ($arr["FB_IMG"] == ""))
            $this->updateImage ($dir."/obs.info", "FB_IMG", $all["SOURCES"]["FB"]["IMAGE"]);
        }

        if ($arr["MB_ENABLED"] == "true")
        {
          $all["SOURCES"]["MB"] = array();
          $all["SOURCES"]["MB"] = array();
          $all["SOURCES"]["MB"]["IMAGE"] = "../../../images/blankimage.gif";
        }

        if ($arr["CORR_ENABLED"] == "true")
        {
          $source = $arr["SOURCE"];
          $all["SOURCES"][$source] = array();
          $all["SOURCES"][$source]["TYPE"] = "CORR";
          $all["SOURCES"][$source]["IMAGE"] = $this->getCorrImage($dir, $o, $arr["CORR_IMG"]);
          if ((($all["STATE"] == "finished") || ($all["STATE"] == "transferred")) && ($arr["CORR_IMG"] == ""))
            $this->updateImage ($dir."/obs.info", "CORR_IMG", $all["SOURCES"][$source]["IMAGE"]);
        }

        for ($j=0; $j<4; $j++)
        {
          $tbe_key = "TB".$j."_ENABLED";
          if ((array_key_exists ($tbe_key, $arr)) && ($arr[$tbe_key] == "true"))
          {
            $source = $arr["TB".$j."_SOURCE"];
            $all["SOURCES"][$source] = array();
            $all["SOURCES"][$source]["TYPE"] = "TB";
            $all["SOURCES"][$source]["IMAGE"] = $this->getTBImage($dir, $o, $source, $arr["TB".$j."_IMG"]);
            if ((($all["STATE"] == "finished") || ($all["STATE"] == "transferred")) && ($arr["TB".$j."_IMG"] == ""))
              $this->updateImage ($dir."/obs.info", "TB".$j."_IMG", $all["SOURCES"][$source]["IMAGE"]);
          }
        }

        # use the primary PID
        $all["PID"] = $arr["PID"];

        $all["IMG"] = "NA";
        # find an image of the observation, if not existing
        if (($arr["IMG"] == "NA") || ($arr["IMG"] == ""))
        {
          # preferentially find a pulsar profile plot
          $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
            "-name '*.fl.120x90.png' -printf '%f\n' ".
            "| sort -n | head -n 1";
          $img = exec ($cmd, $output, $rval);

          if (($rval == 0) && ($img != "")) {
            $all["IMG"] = $img;
          } else {
            $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
              "-name '*.*.ad.160x120.png' -printf '%f\n' ".
              "-o -name '*.FB.00.*png' -printf '%f\n' ".
              "| sort -n | head -n 1";
            $img = exec ($cmd, $output, $rval);

            if (($rval == 0) && ($img != "")) {
              $all["IMG"] = $img;
            } else {
              $all["IMG"] = "../../../images/blankimage.gif";
            }
          }
        }
        else
        {
          $all["IMG"] = $arr["IMG"];
        }
      
        # if the integration length does not yet exist
        $int = 0;
        if (($arr["INT"] == "NA") || ($arr["INT"] <= 0))
        {
          if ($arr["CORR_ENABLED"] == "true")
            $int = $this->calcIntLengthCorr($o, $arr["SOURCE"]);
          else if ($arr["FB_ENABLED"] == "true")
            $int = $this->calcIntLengthFB($o, "FB");
          else if ($arr["TB0_ENABLED"] == "true")
            $int = $this->calcIntLengthTB($o, $arr["TB0_SOURCE"]);
          else
            $int = "0";
          $all["INT"] = $int;
        }
        else
          $all["INT"] = $arr["INT"];

        # if the observation is 
        if (($all["STATE"] == "finished") || ($all["STATE"] == "transferred"))
        {
          if (($arr["INT"] == "NA") || ($arr["INT"] <= 0) && ($all["INT"] > 0))
          {
            system("perl -ni -e 'print unless /^INT/' ".$results_dir."/".$o."/obs.info");
            system("echo 'INT              ".$int."' >> ".$results_dir."/".$o."/obs.info");
          }
        }
      }

      if (file_exists($dir."/obs.txt")) {
        $all["ANNOTATION"] = file_get_contents($dir."/obs.txt");
      } else {
        $all["ANNOTATION"] = "";
      }

      $all_results[$o] = $all;
    }

    return $all_results;
  }

  function getResultsArray_db($pdo, $results_dir, $offset=0, $length=0, $filter_type, $filter_value) 
  {
    if ($filter_type != "SOURCE") {
      throw new Exception('getResultsArray_db only works for SOURCE filter');
    } elseif (strpos($filter_value, 'J') !== 0) {
      throw new Exception('getResultsArray_db only works for pulsar SOURCE (starting with J)');
    }

    $all_results = array();

    $observations = array();
    $dir = $results_dir;
    
    $q = 'SELECT utc FROM (Pulsars JOIN UTCs JOIN TB_Obs ON Pulsars.id = TB_Obs.psr_id AND UTCs.id =
      TB_Obs.utc_id) WHERE name LIKE "'.$filter_value.'%" ORDER BY utc DESC LIMIT '.$length.' OFFSET '.$offset;
    try {
      $stmt = $pdo -> query ($q);
    } catch (PDOException $ex) {
      print $ex->getMessage();
    }

    $observations = $stmt->fetchall(PDO::FETCH_COLUMN, 0);

    $_results_dir = $this->cfg["SERVER_RESULTS_DIR"];
    $_old_results_dir = $this->cfg["SERVER_OLD_RESULTS_DIR"];
    $_archive_dir = $this->cfg["SERVER_ARCHIVE_DIR"];
    $_old_archive_dir = $this->cfg["SERVER_OLD_ARCHIVE_DIR"];
    for ($i=0; $i<count($observations); $i++)
    {
      $o = $observations[$i]; // This sets o to a UTC
      $dir = $_results_dir."/".$o;
      if (file_exists($_results_dir."/".$o)) {
        $dir = $_results_dir."/".$o;
        $this->results_dir  = $_results_dir;
        $this->archive_dir  = $_archive_dir;
        $is_new_old_db = "new";
      } elseif (file_exists($_old_results_dir."/".$o)) {
        $dir = $_old_results_dir."/".$o;
        $this->results_dir  = $_old_results_dir;
        $this->archive_dir  = $_old_archive_dir;
        $is_new_old_db = "old";
      } else {
        $is_new_old_db = "DB";
        $dir = "";
      }

      // read the obs.info file into an array 
      if ($dir != "")
      {
        # this branch is for observations which are still present on the backend drives
        $arr = getConfigFile($dir."/obs.info");
        $all = array();

        $all["STATE"] = "unknown";
        if (file_exists($dir."/obs.processing"))
          $all["STATE"] = "processing";
        else if (file_exists($dir."/obs.finished"))
          $all["STATE"] = "finished";
        else if (file_exists($dir."/obs.transferred"))
          $all["STATE"] = "transferred";
        else if (file_exists($dir."/obs.completed"))
          $all["STATE"] = "completed";
        else if (file_exists($dir."/obs.failed"))
          $all["STATE"] = "failed";
        else
           $all["STATE"] = "unknown";

        $all["SOURCE"] = $arr["SOURCE"];

        $all["SOURCES"] = array();
        if ($arr["FB_ENABLED"] == "true")
        {
          $all["SOURCES"]["FB"] = array();
          $all["SOURCES"]["FB"]["TYPE"] = "FB";
          $all["SOURCES"]["FB"]["IMAGE"] = $this->getFBImage($dir, $o, $arr["FB_IMG"]);
          if ((($all["STATE"] == "finished") || ($all["STATE"] == "transferred")) && ($arr["FB_IMG"] == ""))
            $this->updateImage ($dir."/obs.info", "FB_IMG", $all["SOURCES"]["FB"]["IMAGE"]);
        }

        if ($arr["MB_ENABLED"] == "true")
        {
          $all["SOURCES"]["MB"] = array();
          $all["SOURCES"]["MB"] = array();
          $all["SOURCES"]["MB"]["IMAGE"] = "../../../images/blankimage.gif";
        }

        if ($arr["CORR_ENABLED"] == "true")
        {
          $source = $arr["SOURCE"];
          $all["SOURCES"][$source] = array();
          $all["SOURCES"][$source]["TYPE"] = "CORR";
          $all["SOURCES"][$source]["IMAGE"] = $this->getCorrImage($dir, $o, $arr["CORR_IMG"]);
          if ((($all["STATE"] == "finished") || ($all["STATE"] == "transferred")) && ($arr["CORR_IMG"] == ""))
            $this->updateImage ($dir."/obs.info", "CORR_IMG", $all["SOURCES"][$source]["IMAGE"]);
        }

        for ($j=0; $j<4; $j++)
        {
          $tbe_key = "TB".$j."_ENABLED";
          if ((array_key_exists ($tbe_key, $arr)) && ($arr[$tbe_key] == "true"))
          {
            $source = $arr["TB".$j."_SOURCE"];
            $all["SOURCES"][$source] = array();
            $all["SOURCES"][$source]["TYPE"] = "TB";
            $all["SOURCES"][$source]["IMAGE"] = $this->getTBImage($dir, $o, $source, $arr["TB".$j."_IMG"]);
            if ((($all["STATE"] == "finished") || ($all["STATE"] == "transferred")) && ($arr["TB".$j."_IMG"] == ""))
              $this->updateImage ($dir."/obs.info", "TB".$j."_IMG", $all["SOURCES"][$source]["IMAGE"]);
          }
        }

        # use the primary PID
        $all["PID"] = $arr["PID"];

        $all["IMG"] = "NA";
        # find an image of the observation, if not existing
        if (($arr["IMG"] == "NA") || ($arr["IMG"] == ""))
        {
          if ($filter_type === "SOURCE" && strpos($this->filter_value, "J") === 0) {
            $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
              "-name '*".$filter_value."*.fl.120x90.png' -printf '%f\n' ".
              "| sort -n | head -n 1";
          } else {
            # preferentially find a pulsar profile plot
            $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
              "-name '*.fl.120x90.png' -printf '%f\n' ".
              "| sort -n | head -n 1";
          }
          $img = exec ($cmd, $output, $rval);

          if (($rval == 0) && ($img != "")) {
            $all["IMG"] = $img;
          } else {
            $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
              "-name '*.*.ad.160x120.png' -printf '%f\n' ".
              "-o -name '*.FB.00.*png' -printf '%f\n' ".
              "| sort -n | head -n 1";
            $img = exec ($cmd, $output, $rval);

            if (($rval == 0) && ($img != "")) {
              $all["IMG"] = $img;
            } else {
              $all["IMG"] = "../../../images/blankimage.gif";
            }
          }
        }
        else
        {
          $all["IMG"] = $arr["IMG"];
        }
      
        # if the integration length does not yet exist
        $int = 0;
        if (($arr["INT"] == "NA") || ($arr["INT"] <= 0))
        {
          if ($arr["CORR_ENABLED"] == "true")
            $int = $this->calcIntLengthCorr($o, $arr["SOURCE"]);
          else if ($arr["FB_ENABLED"] == "true")
            $int = $this->calcIntLengthFB($o, "FB");
          else if ($arr["TB0_ENABLED"] == "true")
            $int = $this->calcIntLengthTB($o, $arr["TB0_SOURCE"]);
          else
            $int = "0";
          $all["INT"] = $int;
        }
        else
          $all["INT"] = $arr["INT"];

        # if the observation is 
        if (($all["STATE"] == "finished") || ($all["STATE"] == "transferred"))
        {
          if (($arr["INT"] == "NA") || ($arr["INT"] <= 0) && ($all["INT"] > 0))
          {
            system("perl -ni -e 'print unless /^INT/' ".$dir."/obs.info");
            system("echo 'INT              ".$int."' >> ".$dir."/obs.info");
          }
        }
        if (file_exists($dir."/obs.txt")) {
          $all["ANNOTATION"] = file_get_contents($dir."/obs.txt");
        } else {
          $all["ANNOTATION"] = "";
        }
        if ($is_new_old_db == "old") {
          $all["ANNOTATION"] = "<i>Data in old_results.</i> ".$all["ANNOTATION"];
        }
      } else {
        // this branch is for observations with entries only in the DB
        // TODO needs to be actually implemeneted
        $all = array();
        if ($is_new_old_db == "DB") {
          $all["ANNOTATION"] = "<i>Data in database only, check Timing dir or on gstar.<i> ".$all["ANNOTATION"];
        } else {
          print "This should never happen";
        }
      }
      $all_results[$o] = $all;
    }
    // Restore page config to its original state:
    if ($this->class == "new") {
      $this->results_dir = $_results_dir;
      $this->archive_dir= $_archive_dir;
    } else {
      $this->results_dir = $_old_results_dir;
      $this->archive_dir= $_ol_archive_dir;
    } // Do I actually need to do this?

    return $all_results;
  }

  function calcIntLengthCorr($utc_start, $source)
  {
    $cc_file = $this->results_dir."/".$utc_start."/".$source."/cc.sum"; 
    if (file_exists ($cc_file))
    {
      $cmd = "find ".$this->archive_dir."/".$utc_start."/".$source." -name '*.ac' | sort -n | tail -n 1";
      $output = array();
      $ac = exec($cmd, $output, $rval);

      $parts = explode("_", $ac);
      $time_to = $parts[count($parts)-1];

      #$cmd = "grep BYTES_PER_SECOND ".$this->results_dir."/".$utc_start."/".$source."/obs.header | awk '{print $2}'";
      #$output = array();
      #$Bps = exec($cmd, $output, $rval);

      $length = $time_to;
      return sprintf ("%5.0f", $length);
    }
    return 0;
  }

  function calcIntLengthFB ($utc_start, $source)
  {
    $ac_file = $this->results_dir."/".$utc_start."/".$source."/all_candidates.dat";
    if (file_exists($ac_file))
    {
      $cmd = "tail -n 1000 ".$ac_file." | awk '{print $3}' | sort -n | tail -n 1";
      $length = exec($cmd, $output, $rval);
      return sprintf("%5.0f", $length); 
    }
    return 0;
  }

  function calcIntLengthTB($utc_start, $source) 
  {
    $dir = $this->results_dir."/".$utc_start."/".$source." ".
           $this->archive_dir."/".$utc_start."/".$source;
    $length = 0;

    # try to find a TB/*_f.tot file
    $cmd = "find ".$this->results_dir."/".$utc_start."/".$source." -mindepth 1 -maxdepth 1 -type f -name '*_f.tot' | sort -n | tail -n 1";
    $tot = exec($cmd, $output, $rval);
    if ($tot != "")
    {
      $cmd = $this->cfg["SCRIPTS_DIR"]."/psredit -Q -c length ".$tot;
      $output = array();
      $result = exec($cmd, $output, $rval);
      list ($file, $length) = split(" ", $result);
      if ($length != "" && $rval == 0)
        return sprintf ("%5.0f",$length);
    }
    
    # try to find a 2*.ar file
    $cmd = "find ".$dir." -mindepth 2 -maxdepth 2 -type f -name '2*.ar' -printf '%f\n' | sort -n | tail -n 1";
    $ar = exec($cmd, $output, $rval);
    if ($ar != "")
    {
      $array = split("\.",$ar);
      $ar_time_str = $array[0];

      # if image is pvf, then it is a local time, convert to unix time
      $ar_time_unix = unixTimeFromGMTime($ar_time_str);
      
      # add ten as the 10 second image file has a UTC referring to the first byte of the file 
      $length = $ar_time_unix - unixTimeFromGMTime($utc_start);
    }

    return $length;
  }

  function getFBImage($dir, $o, $existing)
  {
    if (($existing == "NA") || ($existing == ""))
    {
      $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
             " -name '*.FB.00.*png' -printf '%f\n' ".
              "| sort -n | head -n 1";
      $img = exec ($cmd, $output, $rval);
      if (($rval == 0) && ($img != ""))
      {
        return $o."/".$img;
      }
      return "../../../images/blankimage.gif";
    }
    else
      return $existing;
  }

  function getCorrImage ($dir, $o, $existing)
  {
    if (($existing == "NA") || ($existing == ""))
    {
      $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
             " -name '*.*.ad.160x120.png' -printf '%f\n' ".
              "| sort -n | head -n 1";
      $img = exec ($cmd, $output, $rval);
      if (($rval == 0) && ($img != ""))
      {
        return $o."/".$img;
      }
      return "../../../images/blankimage.gif";
    }
    else
      return $existing;
  }

  function getTBImage ($dir, $o, $s, $existing)
  {
    if (($existing == "NA") || ($existing == ""))
    {
      $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f".
             " -name '*.".$s.".fl.120x90.png' -printf '%f\n'".
             " | sort -n | head -n 1";
      $img = exec ($cmd, $output, $rval);
      if (($rval == 0) && ($img != ""))
      {
        return $o."/".$img;
      }
      return "../../../images/blankimage.gif";
    }
    else
      return $existing;
  }

  function updateImage($file, $key, $value)
  {
    system("perl -ni -e 'print unless /^".$key."/' ".$file);
    system("echo '".$key."            ".$value."' >> ".$file);
  }

}
handledirect("results");
