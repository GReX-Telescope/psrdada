<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class transient_viewer extends bpsr_webpage 
{
  var $results_dir;

  var $utc_start;

  var $default_snr_cut;

  var $default_filter_cut;

  var $inst;

  var $max_candidates;

  var $update_required;

  function transient_viewer()
  {
    bpsr_webpage::bpsr_webpage();
    $this->title = "BPSR Transient Pipeline";
    $this->callback_freq = 16000;
    $this->inst = new bpsr();

    $this->default_snr_cut = 6.5;
    $this->default_filter_cut = 11;
    $this->default_dm_cut = 1.5;
    $this->max_candidates = 20;

    $this->utc_start = isset($_GET["utc_start"]) ? $_GET["utc_start"] : "";

    $this->results_dir = $this->inst->config["SERVER_RESULTS_DIR"];
    if (isset($_GET["results"]) && ($_GET["results"] == "old"))
      $this->results_dir = $this->inst->config["SERVER_OLD_RESULTS_DIR"];

    if ($this->utc_start != "")
      $this->update_required = (file_exists($this->results_dir."/".$this->utc_start."/obs.processing"));
    else
      $this->update_required = true;
  }

  function javaScriptCallback()
  {
    if ($this->update_required)
      return "transient_viewer_request();";
    else
      return "";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      var counter = 0;
      var utc_start = "<?echo $this->utc_start;?>";
      var default_snr_cut = <?echo $this->default_snr_cut;?>;
      var default_filter_cut = <? echo $this->default_filter_cut;?>;
      var default_dm_cut = <? echo $this->default_dm_cut;?>;
      var default_beam_mask = 8191

      function calculate_beam_mask()
      {
        var i;
        var mask = 8191
        var beam = "";
        var beams = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"];
        for (i=0; i<beams.length; i++)
        {
          beam = beams[i];
          if (!(document.getElementById("beam"+beam).checked))
          {
            mask -= Math.pow(2,i);
          }
        }
        return mask;
      }

      function update_beam_visibility(beam)
      {
        transient_viewer_request();
      }

      function update_snr()
      {
        transient_viewer_request();
      }

      function update_filter()
      {
        transient_viewer_request();
      }

      function update_dm()
      {
        transient_viewer_request();
      }

      function start_pipeline()
      {
        url = "transient_viewer.lib.php?action=start";
        tw_action_request(url);
      }

      function stop_pipeline()
      {
        url = "transient_viewer.lib.php?action=stop";
        tw_action_request(url);
      }
        
      function tw_action_request()
      {
        var tw_action_http_request;
        if (window.XMLHttpRequest)
          tw_action_http_request = new XMLHttpRequest();
        else
          tw_action_http_request = new ActiveXObject("Microsoft.XMLHTTP");
  
        tw_action_http_request.onreadystatechange = function() 
        {
          handle_tw_action_request(tw_action_http_request);
        }

        tw_action_http_request.open("GET", url, true);
        tw_action_http_request.send(null);
      }

      function handle_tw_action_request(tw_action_http_request)
      {
        if (tw_action_http_request.readyState == 4)
        {
          var response = String(tw_action_http_request.responseText);
          alert(response);
        }
      }

      function handle_transient_viewer_request(tw_xml_request) 
      {
        if (tw_xml_request.readyState == 4)
        {
          var xmlDoc = tw_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;

            var cands = xmlObj.getElementsByTagName("transient_candidates");

            var i = 0;
            var j = 0;

            var pipeline_state = xmlObj.getElementsByTagName("pipeline_state")[0].childNodes[0].nodeValue;
            document.getElementById("pipeline_state").innerHTML = pipeline_state;

            var beam_infos = xmlObj.getElementsByTagName("beam_info");
            for (i=0; i<beam_infos.length; i++)
            {
              var beam = beam_infos[i];
              var beam_name = beam.getAttribute("beam");
              var ra_element = document.getElementById("beam"+beam_name+"_ra");
              var dec_element = document.getElementById("beam"+beam_name+"_dec");
              ra_element.innerHTML = beam.getAttribute("raj");
              dec_element.innerHTML = beam.getAttribute("decj");

              // check if any children
              var psr_element = document.getElementById("beam"+beam_name+"_psr");
              var dm_element = document.getElementById("beam"+beam_name+"_dm");
              var s1400_element = document.getElementById("beam"+beam_name+"_s1400");
              var psr_list = "";
              var dm_list = "";
              var s1400_list = "";

              try 
              {
                children = beam.childNodes;
                if (children.length > 0)
                {
                  for (j=0; j<children.length; j++)
                  {
                    if (children[j].nodeType == 1)
                    { 
                      psr_list += children[j].getAttribute("name");
                      dm_list += children[j].getAttribute("dm");
                      s1400_list += children[j].getAttribute("S1400");

                      if (j < children.length-1)
                      {
                        psr_list += "<br/>";
                        dm_list += "<br/>";
                        s1400_list += "<br/>";
                      }
                    }
                  }
                }
              }
              catch (e)
              {
                //psr_list += ""
              }
              psr_element.innerHTML = psr_list;
              dm_element.innerHTML = dm_list;
              s1400_element.innerHTML = s1400_list;
            }

            var cand_list = xmlObj.getElementsByTagName("cand_list");
            for (i=0; i<cand_list.length; i++)
            {
              for (j=0; j<cand_list[i].childNodes.length; j++)
              {
                var cand = cand_list[i].childNodes[j];
                var snr = cand.getAttribute("snr");
                var samp_idx = cand.getAttribute("samp_idx");
                var filter = cand.getAttribute("filter");
                var width = Math.pow(2,filter) * 0.064;
                var time = parseFloat(cand.getAttribute("time"));
                var dm = parseFloat(cand.getAttribute("dm"));
                var beam = cand.getAttribute("prim_beam");
                if (beam < 10)
                  beam = "0"+ beam;

                document.getElementById("cand_"+j+"_snr").innerHTML = snr;
                document.getElementById("cand_"+j+"_time").innerHTML = time.toFixed(2);
                document.getElementById("cand_"+j+"_dm").innerHTML = dm.toFixed(2);
                document.getElementById("cand_"+j+"_width").innerHTML = width;
                document.getElementById("cand_"+j+"_beam").innerHTML = beam;

                if (utc_start != "") 
                { 
                  var link1 = "<a href='/bpsr/candidate_viewer.lib.php?single=true&utc_start="+utc_start+
                              "&beam="+beam+"&sample="+samp_idx+"&filter="+filter+"&dm="+dm+"&snr="+snr+
                              "' target='cand_popup'>cand</a>";

                  var link2 = "<a href='/bpsr/candidate_viewer.lib.php?single=true&utc_start="+utc_start+
                              "&beam="+beam+"&sample="+samp_idx+"&filter="+filter+"&dm="+dm+"&snr="+snr+"&proc_type=dspsr"+
                              "' target='cand_popup'>dspsr</a>";

                  document.getElementById("cand_"+j+"_link").innerHTML = link1 + "&nbsp;&nbsp;" + link2
                }
                else
                {
                  var link = "<a href='/bpsr/live_candidate_viewer.lib.php?single=true&utc_start="+utc_start+
                              "&beam="+beam+"&sample="+samp_idx+"&filter="+filter+"&dm="+dm+"&snr="+snr+"&proc_type=dspsr"+
                              "' target='cand_popup'>dspsr</a>";
                  document.getElementById("cand_"+j+"_link").innerHTML = link
                }
              }
              for (j=cand_list[i].childNodes.length; j<<?echo $this->max_candidates?>; j++)
              {
                document.getElementById("cand_"+j+"_snr").innerHTML = "";
                document.getElementById("cand_"+j+"_time").innerHTML = "";
                document.getElementById("cand_"+j+"_dm").innerHTML = "";
                document.getElementById("cand_"+j+"_width").innerHTML = "";
                document.getElementById("cand_"+j+"_beam").innerHTML = "";
                document.getElementById("cand_"+j+"_link").innerHTML = "";
              }
            }

            for (i=0; i<cands.length; i++)
            {
              var cand = cands[i];
              var img_element = document.getElementById("candidate");
              for (j=0; j<cand.childNodes.length; j++)
              {
                img = cand.childNodes[j];
                if (img.nodeType == 1)
                { 
                  if (img.getAttribute("type") == "cands")
                  {
                    var new_image = new Image();
                    var webplot_url = http_server + url_prefix + img.childNodes[0].nodeValue;
                    new_image.id = "candidate";
                    new_image.src = webplot_url;
                    new_image.onload = function() {
                      var img = document.getElementById("candidate");
                      img.parentNode.insertBefore(new_image, img);
                      img.parentNode.removeChild(img);
                    }
                  }
                }
              }
            }

          }
        }
      }
                  
      function transient_viewer_request() 
      {
        var snr_cut    = document.getElementById('snr_cut').value;
        var filter_cut = document.getElementById('filter_cut').value;
        var dm_cut     = document.getElementById('dm_cut').value;
        var beam_mask  = calculate_beam_mask();

        var url  = "transient_viewer.lib.php?update=true";
        var url_args = "&beam_infos=true&cand_list=true";

        url_args += "&snr_cut="+snr_cut
        url_args += "&beam_mask="+beam_mask
        url_args += "&filter_cut="+filter_cut
        url_args += "&dm_cut="+dm_cut

        if (utc_start != "")
          url_args = url_args + "&utc_start="+utc_start;

 <?
        if (isset($_GET["results"]) && ($_GET["results"] == "old"))
        {
?>
        url_args += "&results=old";
<?
        }
?>

        var default_cands = ( (snr_cut == default_snr_cut) && 
                             (beam_mask == default_beam_mask) && 
                             (filter_cut == default_filter_cut) &&
                             (dm_cut == default_dm_cut) &&
                             (utc_start == "") );

        if (default_cands)
        {
          url_args += "&default_cands=true";
        }
        else
        {
          // update the image magically
          var img = document.getElementById("candidate");
          if (img.complete)
          {
            var new_image = new Image();
            var webplot_url = "transient_webplot.php?count="+counter+url_args;
            new_image.id = "candidate";
            new_image.src = webplot_url;
            new_image.onload = function() {
              var img = document.getElementById("candidate");
              img.parentNode.insertBefore(new_image, img);
              img.parentNode.removeChild(img);
            }
            counter += 1;
          }
        }

        url += url_args;

        if (window.XMLHttpRequest)
          tw_xml_request = new XMLHttpRequest();
        else
          tw_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        tw_xml_request.onreadystatechange = function() {
          handle_transient_viewer_request(tw_xml_request)
        };
        tw_xml_request.open("GET", url, true);
        tw_xml_request.send(null);
      }

      function popUp(URL, type) 
      {
        var to = "toolbar=1";
        var sc = "scrollbars=1";
        var l  = "location=1";
        var st = "statusbar=1";
        var mb = "menubar=1";
        var re = "resizeable=1";

        options = to+","+sc+","+l+","+st+","+mb+","+re
        eval("page" + type + " = window.open(URL, '" + type + "', '"+options+",width=1024,height=768');");
      }

    </script>
<?
  }

  function printSideBarHTML()
  {
    $this->openBlockHeader("Pipeline Control");
?>
    <table>
      <tr>
        <td><b>state:</b></td>
        <td><span id="pipeline_state">unknown</span></td>
      </tr>
      <tr>
        <td colspan=2>
          <div class="btns">
<!--
            <a href="javascript:start_pipeline()" class="btn" ><span>Start</span></a>
            <a href="javascript:stop_pipeline()" class="btn"><span>Stop</span></a>
-->
            <a href="javascript:popUp('transient_results.lib.php?single=true','history')" class="btn" ><span>History</span></a>
          </div>
        </td>
      </tr>
    </table>

<?
    $this->closeBlockHeader();

    $this->openBlockHeader("Data Cuts");
?>
    <table>
      <tr>
        <td><b>SNR</b></td>
        <td><input type='text' name='snr_cut' id='snr_cut' size='3' value='<? echo $this->default_snr_cut;?>' onChange='update_snr()'></input></td>
      </tr>
      <tr>
        <td><b>Filter</b></td>
        <td><input type='text' name='filter_cut' id='filter_cut' size='2' value='<? echo $this->default_filter_cut;?>' onChange='update_filter()'></input></td>
     </tr>
      <tr>
        <td><b>DM</b></td>
        <td><input type='text' name='dm_cut' id='dm_cut' size='2' value='<? echo $this->default_dm_cut;?>' onChange='update_dm()'></input></td>
     </tr>
    </table>
<?
    $this->closeBlockHeader();

    $this->openBlockHeader("Beam Information");
?>

    <table border=0 cellspacing=0 cellpadding='2px' width='300px'>
      <tr>
        <td><b>Beam</b></td><td><b>RA</b></td><td><b>DEC</b></td><td><b>Source</b></td><td><b>DM</b></td><td><b>S1400</b></td>
      </tr>
<?

    for ($i=1; $i<=13; $i++)
    {
      $beam = sprintf("%02d", $i);
      echo "    <tr>\n";
      echo "      <td>".$beam."&nbsp;<input type='checkbox' id='beam".$beam."' name='beam".$beam."' onChange='update_beam_visibility(\"".$beam."\")' checked></input></td>\n";
      echo "      <td style='text-align: center; font-size: 8pt;' id='beam".$beam."_ra'></td>\n";
      echo "      <td style='text-align: center; font-size: 8pt;' id='beam".$beam."_dec'></td>\n";
      echo "      <td style='text-align: center; font-size: 8pt;' id='beam".$beam."_psr'></td>\n";
      echo "      <td style='text-align: center; font-size: 8pt;' id='beam".$beam."_dm'></td>\n";
      echo "      <td style='text-align: center; font-size: 8pt;' id='beam".$beam."_s1400'></td>\n";
      echo "    </tr>\n";
    }
?>
    </table>
<?
    $this->closeBlockHeader();

    $this->openBlockHeader("Top ".$this->max_candidates." Candidates");
?>
    <table border=0 cellspacing=0 cellpadding='2px' width='300px'>
      <tr>
        <td><b>SNR</b></td><td><b>Time</b></td><td><b>DM</b></td><td><b>Width</b></td><td><b>Beam</b></td><td><b>Link</b></td>
      </tr>
<?
    for ($i=0; $i<$this->max_candidates; $i++)
    {
      echo  "      <tr>";
      echo  "<td id='cand_".$i."_snr'></td>";
      echo  "<td id='cand_".$i."_time'></td>";
      echo  "<td id='cand_".$i."_dm'></td>";
      echo  "<td id='cand_".$i."_width'></td>";
      echo  "<td id='cand_".$i."_beam'></td>";
      echo  "<td id='cand_".$i."_link'></td>";
      echo "</tr>\n";
    }
?> 
    </table>
<?
    $this->closeBlockHeader();
  }


  /* HTML for this page */
  function printHTML() 
  {
     $this->openBlockHeader("Heimdall Transient Event Pipeline");
?>
   <center>
     <img src="/images/blankimage.gif" border=0 width='1024px' height='768px'; id="candidate" TITLE="Current Candidate" alt="alt"></br>
     <font size="-1">image updates every <?echo ($this->callback_freq / 1000)." seconds";?></font>
    </center>
<?
    if (!$this->update_required)
    {
?>
    <script type="text/javascript">
      transient_viewer_request();
    </script>
<?  } 

    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {
    $host = $this->inst->config["SERVER_HOST"];
    $port = $this->inst->config["SERVER_WEB_MONITOR_PORT"];

    $utc_start = (isset($get["utc_start"]) && ($get["utc_start"] != "")) ? $get["utc_start"] : "";
    $default = (isset($get["default_cands"]) && $get["default_cands"] == "true");
 
    # get the the UTC_START of the current observation   
    if (($utc_start == ""))
    {
      $cmd = "find ".$this->results_dir." -mindepth 2 -maxdepth 2 -type f -name 'all_candidates.dat' -printf '%h\n' | sort | tail -n 1 | awk -F/ '{print \$NF}'";
      $output = array();
      $utc_start = exec($cmd, $output, $rval);
    }


    $url = "http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"];

    $xml = "<plot_update>";
    $xml .= "<http_server>http://".$_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]."</http_server>"; 
    $xml .= "<url_prefix>/bpsr/results/</url_prefix>";

    // if we are viewing the default generated image from server_bpsr_results_manager.pl
    if ($default)
    {
      list ($socket, $result) = openSocket($host, $port);
      $data = "";
      $response = "initial";
      if ($result == "ok") 
      {
        $xml .= "<images>";
        $bytes_written = socketWrite($socket, "img_info\r\n");
        $max = 100;
        while ($socket && ($result == "ok") && ($response != "") && ($max > 0))
        {
          list ($result, $response) = socketRead($socket);
          if ($result == "ok") 
          {
            $data .= $response;
          }
          $max--;
        }
        if (($result == "ok") && ($socket))
          socket_close($socket);
        $socket = 0;
        $xml .= $data;
        $xml .="</images>";
      }
    }

    if (isset($get["beam_infos"]) && ($get["beam_infos"] == "true"))
    {
      $data = "";
      $xml .= "<beam_infos>";
      if ($default)
      {
        # now reopen socket for beam info update
        $response = "initial";
        list ($socket, $result) = openSocket($host, $port);
        if ($result == "ok")
        {
          $bytes_written = socketWrite($socket, "beam_info\r\n");
          $max = 100;
          while ($socket && ($result == "ok") && ($response != "") && ($max > 0))
          {
            list ($result, $response) = socketRead($socket);
            if ($result == "ok")
            {
              $data .= $response;
            }
            $max--;
          }
          if (($result == "ok") && ($socket))
            socket_close($socket);
          $socket = 0;
        }
      }
      else
      {
        $xml_file = $this->results_dir."/".$utc_start."/beaminfo.xml";
        if (file_exists($xml_file))
        {
          $data = file_get_contents($xml_file);
        }
      }
      $xml .= $data;
      $xml .= "</beam_infos>";
    }

    if (isset($get["cand_list"]) && ($get["cand_list"] == "true"))
    {
      $data = "";
      $xml .= "<cand_list>";
      #if ($default)
      if (0)
      {
        $response = "initial";
        list ($socket, $result) = openSocket($host, $port);
        if ($result == "ok")
        {
          $bytes_written = socketWrite($socket, "cand_list\r\n");
          $max = 100;
          while ($socket && ($result == "ok") && ($response != "") && ($max > 0))
          {
            list ($result, $response) = socketRead($socket);
            if ($result == "ok")
            {
              $data .= $response;
            }
            $max--;
          }
          if (($result == "ok") && ($socket))
            socket_close($socket);
          $socket = 0;
        }
      }
      else
      {
        $cmd = $this->inst->config["SCRIPTS_DIR"]."/trans_gen_overview.py -cand_list_xml";
        $cmd .= " -cands_file ".$this->results_dir."/".$utc_start."/all_candidates.dat";
        if (isset($get["filter_cut"]) && ($get["filter_cut"] != ""))
          $cmd .= " -filter_cut ".$get["filter_cut"];
        if (isset($get["beam_mask"]) && ($get["beam_mask"] != ""))
          $cmd .= " -beam_mask ".$get["beam_mask"];
        if (isset($get["snr_cut"]) && ($get["snr_cut"] != ""))
          $cmd .= " -snr_cut ".$get["snr_cut"];
        if (isset($get["dm_cut"]) && ($get["dm_cut"] != ""))
          $cmd .= " -dm_cut ".$get["dm_cut"];

        $cands_list = array();
        $last_line = exec($cmd, $cands_list, $rval);
        for ($i=0; (($i<$this->max_candidates) && ($i < count($cands_list))); $i++)
          $data .= $cands_list[$i];
      }
      $xml .= $data;
      $xml .= "</cand_list>";
    }

    $xml .= "<pipeline_state>";
    if (file_exists("/home/dada/linux_64/share/run_heimdall"))
      $xml .= "active";
    else
      $xml .= "off";
    $xml .= "</pipeline_state>";

    $xml .= "</plot_update>";

    header('Content-type: text/xml');
    echo $xml;
  }

  function printActionHTML($get)
  {
    $action = $get["action"];
    $response = "";

    if ($action == "stop")
    {
      if (file_exists("/home/dada/linux_64/share/run_heimdall"))
      {
        if (unlink("/home/dada/linux_64/share/run_heimdall"))
          $response = "Pipeline will not run on any future observations";
        else
          $response = "Failed to unlink file: ~dada/linux_64/share/run_heimdall";
      }
      else
        $response = "Pipeline was already disabled";
    }
    else if ($action == "start")
    {
      if (file_exists("/home/dada/linux_64/share/run_heimdall"))
        $response = "Pipeline was already active";
      else
      {
        $cmd = "touch /home/dada/linux_64/share/run_heimdall";
        $array = array();
        $lastline = exec($cmd, $array, $rval);
        if ($rval == 0)
          $response = "Pipeline will run on all future observations";
        else
          $response = "Error touching file: ~dada/linux_64/share/run_heimdall: ".$response;
      }
    }
    echo $response;
  }
}

handleDirect("transient_viewer");

