<?PHP

include("caspsr_webpage.lib.php");
include("definitions_i.php");
include("functions_i.php");
include($instrument.".lib.php");

class control extends caspsr_webpage 
{

  var $inst = "";
  var $pwc_dbs = array();
  var $pwc_dbs_str = "";
  var $demux_dbs = array();
  var $demux_dbs_str = "";

  function control()
  {
    caspsr_webpage::caspsr_webpage();
    $this->inst = new caspsr();

    $config = $this->inst->config;
    $data_blocks = split(" ",$config["DATA_BLOCKS"]);
    for ($i=0; $i<count($data_blocks); $i++)
    {
      $key = strtolower($data_blocks[$i]);
      $nbufs = $config[$data_blocks[$i]."_BLOCK_NBUFS"];
      $bufsz = $config[$data_blocks[$i]."_BLOCK_BUFSZ"];
      array_push($this->pwc_dbs, array("key" => $key, "nbufs" => $nbufs, "bufsz" => $bufsz));
      if ($i == 0)
        $this->pwcs_dbs_str = "\"buffer_".$key."\"";
      else
        $this->pwcs_dbs_str .= ", \"buffer_".$key."\"";
    }

    $data_blocks = split(" ",$config["DEMUX_DATA_BLOCKS"]);
    for ($i=0; $i<count($data_blocks); $i++)
    {
      $key = strtolower($data_blocks[$i]);
      $nbufs = $config["DEMUX_".strtoupper($key)."_NBUFS"];
      $bufsz = $config["DEMUX_".strtoupper($key)."_BUFSZ"];
      array_push($this->demux_dbs, array("key" => $key, "nbufs" => $nbufs, "bufsz" => $bufsz));
      if ($i == 0)
        $this->demux_dbs_str = "\"buffer_".$key."\"";
      else
        $this->demux_dbs_str .= ", \"buffer_".$key."\"";
    }
  }

  function javaScriptCallback()
  {
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>

      var srv_hosts = new Array("srv0");

      var pwc_hosts = new Array(<?
      echo "'".$this->inst->config["PWC_0"]."'";
      for ($i=1; $i<$this->inst->config["NUM_PWC"]; $i++) {
        echo ",'".$this->inst->config["PWC_".$i]."'";
      }?>);

      var pwc_hosts_str = "";
      for (i=0; i<pwc_hosts.length; i++) {
        pwc_hosts_str += "&host_"+i+"="+pwc_hosts[i];
      }

      var demux_hosts = new Array(<?
      $demuxers    = getConfigMachines($this->inst->config, "DEMUX");
      $demuxers = array_unique($demuxers);
      asort($demuxers);
      echo "'".$demuxers[0]."'";
      for ($i=1; $i<count($demuxers); $i++) {
        echo ",'".$demuxers[$i]."'";
      }?>);

      var demux_hosts_str = "";
      for (i=0; i<demux_hosts.length; i++) {
        demux_hosts_str += "&host_"+i+"="+demux_hosts[i];
      }

      var stage2_wait = 20;
      var stage3_wait = 20;
      var stage4_wait = 20;


      function poll_server()
      {
        daemon_info_request();
        setTimeout('poll_server()', 2000);
      }

      function set_daemons_to_grey(host)
      {
        var imgs = document.getElementsByTagName("img");
        var i=0;
        for (i=0; i<imgs.length; i++) {
          if (imgs[i].id.indexOf(host) != -1) {
            imgs[i].src="/images/grey_light.png";
          }
        }
      }

      function handle_daemon_action_request(http_request)
      {
        if ((http_request.readyState == 4) || (http_request.readyState == 3)) {
          var response = String(http_request.responseText);
          var lines = response.split("\n");

          var area = "";
          var area_container;
          var output_id;
          var output_container;
          var span;
          var tmp;
          var i = 0;

          if (lines.length == 1) {
            alert("only 1 line received: "+lines[0]);

          } else if (lines.length >= 2) {

            area = lines[0];
            output_id = lines[1];
            area_container = document.getElementById(area+"_output");

            output_container = document.getElementById(output_id);
            try {
              tmp = output_container.innerHTML;
            } catch(e) {
              area_container.innerHTML += "<div id='"+output_id+"'></div>"; 
              output_container = document.getElementById(output_id);
            }

            output_container.innerHTML = "";
            for (i=2; i<lines.length; i++) {
              if (lines[i] != "")
                output_container.innerHTML += lines[i]+"<br>";
            } 
            if (http_request.readyState == 3) 
               output_container.style.color = "#777777";
            else
               output_container.style.color = "#000000";

            var controls_container = document.getElementById(area+"_controls");
            var can_keep_clearing = true;
            while (can_keep_clearing && (area_container.offsetHeight > (controls_container.offsetHeight+5))) {
              var children = area_container.childNodes;
              if (children.length > 1) {
                area_container.removeChild(children[0]);
              } else {
                can_keep_clearing = false;
              }
            }
            //alert ("controls_height = "+document.getElementById(area+"_controls").offsetHeight+", output_height="+area_container.offsetHeight);

          } else {
            alert("lines.length = "+lines.length);
          }
        }
      }
    
      function toggleDaemon(host, daemon, args) {
        var img = document.getElementById("img_"+host+"_"+daemon);
        var src = new String(img.src);
        var action = "";
        var url = "";

        if (src.indexOf("green_light.png",0) != -1)
          action = "stop";
        else if (src.indexOf("red_light.png",0) != -1)
          action = "start";
        else if (src.indexOf("yellow_light.png",0) != -1)
          action = "stop";
        else
          action = "ignore";

        if (action != "ignore") {
          url = "control.lib.php?action="+action+"&nhosts=1&host_0="+host+"&daemon="+daemon;
          if (args != "")
            url += "&args="+args

          var da_http_request;
          if (window.XMLHttpRequest)
            da_http_request = new XMLHttpRequest()
          else
            da_http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
          da_http_request.onreadystatechange = function() 
          {
            handle_daemon_action_request(da_http_request)
          }
          da_http_request.open("GET", url, true)
          da_http_request.send(null)
        } 
      }

      function toggleDaemons(action, daemon, host_string)
      {
        var hosts = host_string.split(" ");
        var i = 0;
        var url = "control.lib.php?action="+action+"&daemon="+daemon+"&nhosts="+hosts.length;
        for (i=0; i<hosts.length; i++) {
          url += "&host_"+i+"="+hosts[i];
        }

        var da_http_request;
        if (window.XMLHttpRequest)
          da_http_request = new XMLHttpRequest()
        else
          da_http_request = new ActiveXObject("Microsoft.XMLHTTP");
  
        da_http_request.onreadystatechange = function() 
        {
          handle_daemon_action_request(da_http_request)
        }

        da_http_request.open("GET", url, true)
        da_http_request.send(null)
      }

      function daemon_action_request(url) 
      {
        var da_http_request;
        if (window.XMLHttpRequest)
          da_http_request = new XMLHttpRequest();
        else
          da_http_request = new ActiveXObject("Microsoft.XMLHTTP");
  
        da_http_request.onreadystatechange = function() 
        {
          handle_daemon_action_request(da_http_request);
        }

        da_http_request.open("GET", url, true);
        da_http_request.send(null);
      }

      // machines sure the machines[m] and daemons[m] light all matches the 
      // string in c 
      function checkMachinesAndDaemons(m, d, c) {
        var i=0;
        var j=0;
        var ready = true;
        for (i=0; i<m.length; i++) {
          for (j=0; j<d.length; j++) {
            element = document.getElementById("img_"+m[i]+"_"+d[j]);
            try {
              if (element.src.indexOf(c) == -1) {
                ready = false;
              }
            } catch (e) {
              alert("checkMachinesAndDameons: m="+m+", d="+d+", c="+c+" did not exist");
            }
          }
        }
        return ready;
      }
      

      function startCaspsr() 
      {

        // start the server's master control script
        url = "control.lib.php?action=start&daemon=caspsr_master_control&nhosts=1&host_0=srv0"
        daemon_action_request(url);

        // start the demux's master control script
        url = "control.lib.php?action=start&daemon=caspsr_master_control&nhosts="+demux_hosts.length+demux_hosts_str;
        daemon_action_request(url);

        // start the pwc's master control script
        url = "control.lib.php?action=start&daemon=caspsr_master_control&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        stage2_wait = 20;
        startCaspsrStage2();
      }

      function startCaspsrStage2()
      {
        var machines = srv_hosts.concat(pwc_hosts, demux_hosts);
        var daemons = new Array("caspsr_master_control");
        var ready = checkMachinesAndDaemons(machines, daemons, "green_light.png");
        if ((!ready) && (stage2_wait > 0)) {
          stage2_wait--;
          setTimeout('startCaspsrStage2()', 1000);
          return 0;
        }
        stage2_wait = 0;
                
        // init the pwc's datablocks
<?
        for ($i=0; $i<count($this->pwc_dbs); $i++)
        {
          $key = $this->pwc_dbs[$i]["key"];
          $nbufs = $this->pwc_dbs[$i]["nbufs"];
          $bufsz =  $this->pwc_dbs[$i]["bufsz"];
          $args = $nbufs."_".$bufsz;
          echo "         url = \"control.lib.php?action=start&daemon=buffer_".$key."&args=".$args."&nhosts=\"+pwc_hosts.length+pwc_hosts_str\n";
          echo "         daemon_action_request(url);\n";
        }
?>

        // init the demux datablocks
        <?
        for ($i=0; $i<count($this->demux_dbs); $i++)
        {
          $key = $this->demux_dbs[$i]["key"];
          $nbufs = $this->demux_dbs[$i]["nbufs"];
          $bufsz =  $this->demux_dbs[$i]["bufsz"];
          $args = $nbufs."_".$bufsz;
          echo "         url = \"control.lib.php?action=start&daemon=buffer_".$key."&args=".$args."&nhosts=\"+demux_hosts.length+demux_hosts_str\n";
          echo "         daemon_action_request(url);\n";
        }
?>

        stage3_wait = 20;
        startCaspsrStage3();

      }

      function startCaspsrStage3()
      {

        var pwc_daemons = new Array(<?echo $this->pwcs_dbs_str?>); 
        var demux_daemons = new Array(<?echo $this->demux_dbs_str?>); 
     
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "green_light.png");
        var demux_ready = checkMachinesAndDaemons(demux_hosts, demux_daemons, "green_light.png");

        if ((!(pwc_ready && demux_ready)) && (stage3_wait > 0)) {
          stage3_wait--;
          setTimeout('startCaspsrStage3()', 1000);
          return 0;
        }
        stage3_wait = 0;

        // start the pwc's pwc
        url = "control.lib.php?action=start&daemon=pwcs&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        // start the server daemons 
        url = "control.lib.php?action=start&daemon=all&nhosts=1&host_0=srv0"
        daemon_action_request(url);

        stage4_wait = 20;
        startCaspsrStage4();
      }

      function startCaspsrStage4()
      {

        var pwc_daemons = new Array("pwcs");
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "green_light.png");

        var srv_daemons = new Array("caspsr_pwc_monitor", "caspsr_sys_monitor", "caspsr_demux_monitor");
        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "green_light.png");

        if ((!(pwc_ready && srv_ready)) && (stage4_wait > 0)) {
          stage4_wait--;
          setTimeout('startCaspsrStage4()', 1000);
          return 0;
        }
        stage4_wait = 0;

        // start the pwc's daemons next
        url = "control.lib.php?action=start&daemon=all&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        // start the demux daemons next
        url = "control.lib.php?action=start&daemon=all&nhosts="+demux_hosts.length+demux_hosts_str;
        daemon_action_request(url);

      }

      function hardStopCaspsr()
      {
        // stop server TCS interface
        url = "control.lib.php?script=caspsr_hard_reset.pl";
        popUp(url);
      }

      function stopCaspsr()
      {

        // stop server TCS interface
        url = "control.lib.php?action=stop&daemon=caspsr_tcs_interface&nhosts=1&host_0=srv0";
        daemon_action_request(url);

        // stop the demux daemons next
        url = "control.lib.php?action=stop&daemon=all&nhosts="+demux_hosts.length+demux_hosts_str;
        daemon_action_request(url);

        // stop the pwc's daemons next
        url = "control.lib.php?action=stop&daemon=all&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        stage2_wait = 20;
        stopCaspsrStage2();
      }


      function stopCaspsrStage2()
      {

        var srv_daemons = new Array("caspsr_tcs_interface");
        var demux_daemons = new Array("caspsr_demux_manager");
        var pwc_daemons = new Array("caspsr_processing_manager", "caspsr_archive_manager");

        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "red_light.png");
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");
        var demux_ready = checkMachinesAndDaemons(demux_hosts, demux_daemons, "red_light.png");

        if ((!(srv_ready && pwc_ready && demux_ready)) && (stage2_wait > 0)) {
          stage2_wait--;
          setTimeout('stopCaspsrStage2()', 1000);
          return 0;
        }
        stage2_wait = 0;

        // stop the pwc's pwc
        url = "control.lib.php?action=stop&daemon=pwcs&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

         // destroy the demux's datablocks
<?
        for ($i=0; $i<count($this->demux_dbs); $i++)
        {
          $key = $this->demux_dbs[$i]["key"];
          echo "         url = \"control.lib.php?action=stop&daemon=buffer_".$key."&nhosts=\"+demux_hosts.length+demux_hosts_str;\n";
          echo "         daemon_action_request(url);\n";
        }
?>

        stage3_wait = 20;
        stopCaspsrStage3();
      } 

      function stopCaspsrStage3()
      {

        var demux_daemons = new Array(<?echo $this->demux_dbs_str?>);
        var pwc_daemons = new Array("pwcs");

        var demux_ready = checkMachinesAndDaemons(demux_hosts, demux_daemons, "red_light.png");
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");

        if ((!(pwc_ready && demux_ready)) && (stage3_wait > 0)) {
          stage3_wait--;
          setTimeout('stopCaspsrStage3()', 1000);
          return 0;
        }
        stage3_wait = 0;

        // destroy the pwc's datablocks
<?
        for ($i=0; $i<count($this->pwc_dbs); $i++)
        {
          $key = $this->pwc_dbs[$i]["key"];
          echo "         url = \"control.lib.php?action=stop&daemon=buffer_".$key."&nhosts=\"+pwc_hosts.length+pwc_hosts_str;\n";
          echo "         daemon_action_request(url);\n";
        }
?>

        // stop the server daemons 
        url = "control.lib.php?action=stop&daemon=all&nhosts=1&host_0=srv0"
        daemon_action_request(url);

        // stop the demux's master control script
        url = "control.lib.php?action=stop&daemon=caspsr_master_control&nhosts="+demux_hosts.length+demux_hosts_str;
        daemon_action_request(url);

        stage4_wait = 20;
        stopCaspsrStage4();

      }
    
      function stopCaspsrStage4()
      {

        var srv_daemons = new Array("caspsr_results_manager", "caspsr_pwc_monitor", "caspsr_sys_monitor", "caspsr_demux_monitor", "caspsr_web_monitor");
        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "red_light.png");

        var pwc_daemons = new Array(<?echo $this->pwcs_dbs_str?>); 
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");

        if ((!(srv_ready && pwc_ready)) && (stage4_wait > 0)) {
          stage4_wait--;
          setTimeout('stopCaspsrStage4()', 1000);
          return 0;
        }
        stage4_wait = 0;

        // stop the pwc's master control script
        url = "control.lib.php?action=stop&daemon=caspsr_master_control&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        // stop the server's master control script
        url = "control.lib.php?action=stop&daemon=caspsr_master_control&nhosts=1&host_0=srv0"
        daemon_action_request(url);

      }

      function handle_daemon_info_request(xml_request) 
      {
        if (xml_request.readyState == 4) {

          var xmlDoc=xml_request.responseXML;
          var xmlObj=xmlDoc.documentElement; 

          var i, j, k, result, key, value, span, this_result;

          var results = xmlObj.getElementsByTagName("daemon_info");

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

            host = this_result["host"];
            str = "";

            for ( key in this_result) {
              if (key == "host") {

              } else {
                
                value = this_result[key];
                str = str + " "+key+" -> "+value;

                var img = document.getElementById("img_"+host+"_"+key);
                if (img == null) {
                  alert("img_"+host+"_"+key+" did not exist");
                }
                if (value == "2")
                  img.src = "/images/green_light.png";
                else if (value == "1") 
                  img.src = "/images/yellow_light.png";
                else if (value == "0") {
                  if (key == "caspsr_master_control") {
                    set_daemons_to_grey(host);
                  }
                  img.src = "/images/red_light.png";
                } else
                  img.src = "/images/grey_light.png";
              
              } 
            }
          }
        }
      }

      function daemon_info_request() 
      {
        var di_http_request;
        var url = "control.lib.php?update=true&nhosts="+(srv_hosts.length+pwc_hosts.length+demux_hosts.length);
        var j = 0;

        for (i=0; i<srv_hosts.length; i++)
        {
          url += "&host_"+j+"="+srv_hosts[i];
          j++;
        } 
        for (i=0; i<pwc_hosts.length; i++)
        {
          url += "&host_"+j+"="+pwc_hosts[i];
          j++;
        } 
        for (i=0; i<demux_hosts.length; i++)
        {
          url += "&host_"+j+"="+demux_hosts[i];
          j++;
        } 

        if (window.XMLHttpRequest)
          di_http_request = new XMLHttpRequest()
        else
          di_http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        di_http_request.onreadystatechange = function() 
        {
          handle_daemon_info_request(di_http_request)
        }

        di_http_request.open("GET", url, true)
        di_http_request.send(null)

      }

      function popUp(URL) {

        var to = "toolbar=1";
        var sc = "scrollbar=1";
        var l  = "location=1";
        var st = "statusbar=1";
        var mb = "menubar=1";
        var re = "resizeable=1";

        var type = "hard_reset";
        options = to+","+sc+","+l+","+st+","+mb+","+re
        eval("page" + type + " = window.open(URL, '" + type + "', '"+options+",width=1024,height=768');");

      }
    
    </script>
<?
  }

  function printJavaScriptBody()
  {
?>
<?
  }

  function printSideBarHTML() 
  {
    $this->openBlockHeader("Instrument Controls");
?>
    <input type='button' value='Start' onClick="startCaspsr()">
    <input type='button' value='Stop' onClick="stopCaspsr()">
    <input type='button' value='Hard Stop' onClick="hardStopCaspsr()">
<?
    $this->closeBlockHeader();

    $this->openBlockHeader("Persistent Daemons");
    if (array_key_exists("SERVER_DAEMONS_PERSIST", $this->inst->config))
    {
      $server_daemons_persist = split(" ",$this->inst->config["SERVER_DAEMONS_PERSIST"]);
      $server_daemons_hash  = $this->inst->getServerLogInfo();
      $host = $this->inst->config["SERVER_HOST"];
      $host = substr($host, 0, strpos($host,"."));
?>
      <table width='100%'>
        <tr>
          <td>
            <table class='control' id="persist_controls">
<?
      for ($i=0; $i < count($server_daemons_persist); $i++)
      {
        $d = $server_daemons_persist[$i];
        $this->printServerDaemonControl($d, $server_daemons_hash[$d]["name"], $host);
      }
?>
            </table>
          </td>
        </tr>
        <tr>
          <td height='100px' id='persist_output' valign='top'></td>
        </tr>
      </table>
<?
    }

    $this->closeBlockHeader();



    $this->openBlockHeader("Usage");
?>
    <p>Instrument Controls [above] can be used to start/stop/restart all of the required CASPSR daemons.</p>
    <p>Click on the red/green lights to toggle the respective daemons on/off.</p>
    <p>Use the Start/Stop buttons to turn on/off on all the machines in that section.</p>
    <p>Messages will appear indicating activity, but it may take a few seconds for the daemon lights to turn on/off.</p>
<?
    $this->closeBlockHeader();


  }

  /*************************************************************************************************** 
   *
   * HTML for this page 
   *
   ***************************************************************************************************/
  function printHTML() 
  {
?>
<html>
<head>
  <title>CASPSR | Controls</title>
  <link rel='shortcut icon' href='/caspsr/images/caspsr_favicon.ico'/>
<?
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
    for ($i=0; $i<count($this->ejs); $i++)
      echo "   <script type='text/javascript' src='".$this->ejs[$i]."'></script>\n";
  
    $this->printJavaScriptHead();
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
        $this->printSideBarHTML();
?>
      </div><!-- sidebar1 -->
      <div class="content">
<?
        $this->printMainHTML();
?>
      </div> <!-- content -->
    </div> <!-- contentLayout -->
  </div> <!-- main -->
</body>
</html>
<?
  }

  function printMainHTML()
  {

    ###########################################################################
    #
    # Server Daemons
    # 
    $this->openBlockHeader("Server Daemons");

    $config = $this->inst->config;

    $server_daemons_hash  = $this->inst->getServerLogInfo();
    $server_daemons = split(" ", $config["SERVER_DAEMONS"]);
    $host = $config["SERVER_HOST"];
    $host = substr($host, 0, strpos($host,"."));
?>
    <table width='100%'>
      <tr>
        <td>
          <table id="srv_controls">
<?
    $this->printServerDaemonControl("caspsr_master_control", "Master Control", $host);
    for ($i=0; $i < count($server_daemons); $i++) {
      $d = $server_daemons[$i];
      $this->printServerDaemonControl($d, $server_daemons_hash[$d]["name"], $host);
    }
?>
          </table>
        </td>
        <td width='50%' id='srv_output'></td>
      </tr>

    </table>
<?
    $this->closeBlockHeader();


    ###########################################################################
    #
    # GPU Daemons
    #
    $this->openBlockHeader("GPU Daemons");

    $pwcs    = getConfigMachines($config, "PWC");
    $client_daemons = split(" ",$config["CLIENT_DAEMONS"]);
    $client_daemons_hash =  $this->inst->getClientLogInfo();
?>

    <table width='100%' border=0 cellpadding=0 cellspacing=0 marginwidth=0 marginheight=0>
      <tr>
        <td>
          <table id='gpu_controls'>
            <tr>
              <td></td>
<?
    for ($i=0; $i<count($pwcs); $i++) {
      echo "          <td style='text-align: center'><span title='".$pwcs[$i]."'>".$i."</span></td>\n";
    }
?>
              <td></td>
            </tr>
<?

    $this->printClientDaemonControl("caspsr_master_control", "Master Control", $pwcs, "daemon&name=".$d);

    for ($i=0; $i<count($this->pwc_dbs); $i++)
    {
      $key = $this->pwc_dbs[$i]["key"];
      $nbufs = $this->pwc_dbs[$i]["nbufs"];
      $bufsz =  $this->pwc_dbs[$i]["bufsz"];
      $this->printClientDBControl($key." DB", $pwcs, $key, $nbufs, $bufsz);
    }

    $this->printClientDaemonControl("pwcs", "PWC", $pwcs, "pwcs");
    # Print the client daemons
    for ($i=0; $i<count($client_daemons); $i++) {
      $d = $client_daemons[$i];
      $n = $client_daemons_hash[$d]["name"];
      $this->printClientDaemonControl($d, $n, $pwcs, "daemon&name=".$d);
    }
?>
          </table>
        </td>
        <td width='50%' id='gpu_output'></td>
      </tr>
    </table>
<?
    $this->closeBlockHeader();



    $this->openBlockHeader("Demuxer Daemons");

    $demuxer_daemons = split(" ",$this->inst->config["DEMUX_DAEMONS"]);
    $demuxer_daemons_hash = $this->inst->getDemuxLogInfo();
    $demuxers    = getConfigMachines($this->inst->config, "DEMUX");
    $demuxers = array_unique($demuxers);
    asort($demuxers);
?>
    <table width='100%'>
      <tr>
        <td id="demux_controls">
          <table id="demux_controls">
            <tr>
              <td></td>
<?
    for ($i=0; $i<count($demuxers); $i++) {
      echo "          <td style='text-align: center'><span title='".$demuxers[$i]."'>".$i."</span></td>\n";
    }
?>
              <td></td>
            </tr>
<?
    # Print the master control first
    $d = "caspsr_master_control";
    $this->printClientDaemonControl($d, "Master Control", $demuxers, "daemon&name=".$d);

    for ($i=0; $i<count($this->demux_dbs); $i++)
    {
      $key = $this->demux_dbs[$i]["key"];
      $nbufs = $this->demux_dbs[$i]["nbufs"];
      $bufsz =  $this->demux_dbs[$i]["bufsz"];
      $this->printClientDBControl($key." DB", $demuxers, $key, $nbufs, $bufsz);
    }

    # Print the client daemons
    for ($i=0; $i<count($demuxer_daemons); $i++) {
      $d = $demuxer_daemons[$i];
      $n = $demuxer_daemons_hash[$d]["name"];
      $this->printClientDaemonControl($d, $n, $demuxers, "daemon&name=".$d);
    }
?>
          </table>
        </td>
        <td width='50%' id='demux_output'></td>
      </tr>
    </table>
<?

    $this->closeBlockHeader();

  }

  #############################################################################
  #
  # print update information for the control page as XML
  #
  function printUpdateXML($get)
  {

    $host = "";
    $port = $this->inst->config["CLIENT_MASTER_PORT"];
    $cmd = "daemon_info_xml";

    $nhosts = $get["nhosts"];
    $hosts = array();
    for ($i=0; $i<$nhosts; $i++) {
      $hosts[$i] = $get["host_".$i];
    }

    $sockets = array();
    $results = array();
    $responses = array();


    # open the socket connections
    for ($i=0; $i<count($hosts); $i++) {
      $host = $hosts[$i];
      # echo "opening ".$host.":".$port."<br>\n";
      list ($sockets[$i], $results[$i]) = openSocket($host, $port);
    }

    # write the commands
    for ($i=0; $i<count($results); $i++) {
      $host = $hosts[$i];
      if ($results[$i] == "ok") {
        # echo "sending ".$cmd." to ".$host.":".$port."<br>\n";
        socketWrite($sockets[$i], $cmd."\r\n");
      } else {
        # echo "open socket failed on ".$host."<br>\n";
      }
    }

    # read the responses
    for ($i=0; $i<count($results); $i++) {
      $host = $hosts[$i];
      if ($results[$i] == "ok") {
        $read = socketRead($sockets[$i]);
        $responses[$i] = rtrim($read);
      }
    }

    # close the sockets
    for ($i=0; $i<count($results); $i++) {
      $host = $hosts[$i];
      # echo "closing ".$host.":".$port."<br>\n";
      if ($results[$i] == "ok") {
        socket_close($sockets[$i]);
      }
    }

    # produce the xml
    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<daemon_infos>\n";

    for ($i=0; $i<count($hosts); $i++) {
      if ($responses[$i] == "")
        $xml .= "<daemon_info><host>".$hosts[$i]."</host><caspsr_master_control>0</caspsr_master_control></daemon_info>\n";
      else
        $xml .= $responses[$i]."\n";
    }
    $xml .= "</daemon_infos>\n";

    header('Content-type: text/xml');
    echo $xml;

  }

  #############################################################################
  #
  # print from a _GET based action request
  #
  function printActionHTML($get)
  {

    # generate a unique ID for this output
    $unique_id = $this->generateHash();

    $nhosts = $get["nhosts"];
    for ($i=0; $i<$nhosts; $i++)
      $hosts[$i] = $get["host_".$i];
    $action = $get["action"];
    $daemon = $get["daemon"];
    $args = $get["args"];
    $area = "";

    if (($nhosts == "") || ($action == "") || ($daemon == "") || ($hosts[0] == "")) {
      echo "ERROR: malformed GET parameters\n";
      exit(0);
    }

    # determine which type of request this is (srv, gpu or demux)
    if ($hosts[0] == "srv0") 
      $area = "srv";
    if ($daemon == "caspsr_transfer_manager")
      $area = "persist";
    else {
      for ($i=0; $i<$this->inst->config["NUM_PWC"]; $i++)
        if ($this->inst->config["PWC_".$i] == $hosts[0])
          $area = "gpu";
      for ($i=0; $i<$this->inst->config["NUM_DEMUX"]; $i++)
        if ($this->inst->config["DEMUX_".$i] == $hosts[0])
          $area = "demux";
    }
    if ($area == "") {
      echo "ERROR: could not determine area\n";
      exit(0);
    }

    echo $area."\n";
    echo $unique_id."\n";
    flush();

    if (($daemon == "caspsr_master_control") && ($action == "start")) {
      $html = "Starting ".$daemon." on";
      for ($i=0; $i<$nhosts; $i++) {
        $html .= " ".$hosts[$i];
      }
      echo $html."\n";
      flush();
      for ($i=0; $i<$nhosts; $i++) {
        if ($area == "srv") {
          $cmd = "ssh -x -l dada ".$hosts[$i]." client_caspsr_master_control.pl";
        } else {
          $cmd = "ssh -x -l caspsr ".$hosts[$i]." client_caspsr_master_control.pl";
        }
        $output = array();
        $lastline = exec($cmd, $output, $rval);
      }
    } else {
      $sockets = array();
      $results = array();
      $responses = array();

      $port = $this->inst->config["CLIENT_MASTER_PORT"];

      $html = (($action == "start") ? "Starting" : "Stopping")." ";

      if ($daemon == "all") {
        $cmd = $action."_daemons";
        $html .= "all daemons on ";
      } else if (strpos($daemon, "buffer") !== FALSE) {
        $bits = split("_", $daemon);
        $key = $bits[1];
        if ($action == "stop")
          $cmd = "destroy_db ".$key;
        else 
          $cmd = "init_db ".$key." ".str_replace("_", " ", $args);
        $html .= $cmd." on";
      } else {
        $cmd = $action."_daemon ".$daemon;
        $html .= $daemon." on";
      }

      # open the socket connections
      for ($i=0; $i<count($hosts); $i++) {
        $host = $hosts[$i];
        $html .= " ".$host;
        $ntries = 5;
        $results[$i] = "not open";
        while (($results[$i] != "ok") && ($ntries > 0)) {
          list ($sockets[$i], $results[$i]) = openSocket($host, $port);
          if ($results[$i] != "ok") {
            sleep(2);
            $ntries--;
          } else {
            $ntries = 0;
          }
        }
      }
      echo $html."\n";
      flush();

      # write the commands
      for ($i=0; $i<count($results); $i++) {
        $host = $hosts[$i];
        if ($results[$i] == "ok") {
          socketWrite($sockets[$i], $cmd."\r\n");
        } else {
          echo "ERROR: failed to open socket to master control script for ".$host."\n";
        }
      }

      # read the responses
      for ($i=0; $i<count($results); $i++) {
        $host = $hosts[$i];
        if ($results[$i] == "ok") {

          $done = 32;
          # multiple lines may be returned, final line is always an "ok" or "fail"
          $responses[$i] = "";
          while ($done > 0) {
            $read = rtrim(socketRead($sockets[$i]));
            if (($read == "ok") || ($read == "fail")) {
              flush();
              $done = 0;
            } else {
              $done--;
            }
            if ($read != "")
              $responses[$i] .= $read."\n";
          }
        }
        $responses[$i] = rtrim($responses[$i]);
      }

      # close the sockets
      for ($i=0; $i<count($results); $i++) {
        $host = $hosts[$i];
        if ($results[$i] == "ok") {
          socket_close($sockets[$i]);
        }
      }

      # check the responses
      for ($i=0; $i<count($responses); $i++) {
        $bits = split("\n", $responses[$i]);
        if ($bits[count($bits)-1] != "ok") {
          for ($j=0; $j<count($bits)-1; $j++)
          echo $hosts[$i].": ".$bits[$j]."\n";
        }
      }
    }
  }

  #
  # Run the specified perl script printing the output
  # to the screen
  #
  function printScript($get)
  {
    $script_name = $get["script"];

?>
<html>
<head>
<?  
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
?>
</head>
  <div class='PageBackgroundSimpleGradient'>
  </div>
  <div class='Main'>
    <div class="contentLayout">
      <div class="sidebar1">
        <div style='text-align: center; vertical-align: middle;'>
          <img src="/caspsr/images/caspsr_logo_200x60.png" width=200 height=60>
        </div>
      </div><!-- sidebar1 -->
      <div class="content">
<?
    $this->openBlockHeader("Running ".$script_name);
    echo "<p>Script is now running in background, please wait...</p>\n";
    echo "<br>\n";
    echo "<br>\n";
    flush();
    $script = "source /home/dada/.bashrc; ".$script_name." 2>&1";
    echo "<pre>\n";
    system($script);
    echo "</pre>\n";
    echo "<p>It is now safe to close this window</p>\n";
    $this->closeBlockHeader();
?>  
      </div> <!-- content -->
    </div> <!-- contentLayout -->
  </div> <!-- main -->
</body>
</html>

<?
  }

  #
  # prints a status light with link, id and initially set to value
  #
  function statusLight($host, $daemon, $value, $args, $jsfunc="toggleDaemon") 
  {
    $id = $host."_".$daemon;
    $img_id = "img_".$id;
    $link_id = "link_".$id;
    $colour = "grey";
    if ($value == 0) $colour = "red";
    if ($value == 1) $colour = "yellow";
    if ($value == 2) $colour = "green";

    $img = "<img id='".$img_id."' src='/images/".$colour."_light.png' width='15px' height='15px'>";
    $link = "<a href='javascript:".$jsfunc."(\"".$host."\",\"".$daemon."\",\"".$args."\")'>".$img."</a>";

    return $link;
  }

  function printServerDaemonControl($daemon, $name, $host) 
  {
    echo "  <tr>\n";
    echo "    <td style='vertical-align: middle'>".$name."</td>\n";
    echo "    <td style='vertical-align: middle'>".$this->statusLight($host, $daemon, "-1", "")."</td>\n";
    echo "  </tr>\n";
  }

  function printClientDaemonControl($daemon, $name, $hosts, $cmd) 
  {
    $host_str = "";
    echo "  <tr>\n";
    echo "    <td style='vertical-align: middle'>".$name."</td>\n";
    for ($i=0; $i<count($hosts); $i++) {
      echo "    <td style='vertical-align: middle'>".$this->statusLight($hosts[$i], $daemon, -1, "")."</td>\n";
      $host_str .= $hosts[$i]." ";
    }
    $host_str = rtrim($host_str);
    if ($cmd != "" ) {
      echo "    <td style='text-align: center;'>\n";
      echo "      <input type='button' value='Start' onClick=\"toggleDaemons('start', '".$daemon."', '".$host_str."')\">\n";
      echo "      <input type='button' value='Stop' onClick=\"toggleDaemons('stop', '".$daemon."', '".$host_str."')\">\n";
      echo "    </td>\n";
    }
    echo "  </tr>\n";
  }

  #
  # Print the data block row
  #
  function printClientDBControl($name, $hosts, $key, $nbufs, $bufsz) 
  {

    $daemon = "buffer_".$key;
    $daemon_on = "buffer_".$key."&args=".$nbufs."_".$bufsz;
    $daemon_off = "buffer_".$key;

    $host_str = "";
    echo "  <tr>\n";
    echo "    <td style='vertical-align: middle'>".$name."</td>\n";
    for ($i=0; $i<count($hosts); $i++) {
      echo "    <td style='vertical-align: middle'>".$this->statusLight($hosts[$i], $daemon, -1, $nbufs."_".$bufsz)."</td>\n";
      $host_str .= $hosts[$i]." ";
    }
    $host_str = rtrim($host_str);
    echo "    <td style='text-align: center;'>\n";
    echo "      <input type='button' value='Init' onClick=\"toggleDaemons('start', '".$daemon_on."', '".$host_str."')\">\n";
    echo "      <input type='button' value='Dest' onClick=\"toggleDaemons('stop', '".$daemon_off."', '".$host_str."')\">\n";
    echo "    </td>\n";
    echo "  </tr>\n";

  }

  function generateHash()
  {
    $result = "";
    $charPool = '0123456789abcdefghijklmnopqrstuvwxyz';
    for($p = 0; $p<7; $p++)
    $result .= $charPool[mt_rand(0,strlen($charPool)-1)];
    return substr(sha1(md5(sha1($result))), 4, 16);
  }

  function handleRequest()
  {

    if ($_GET["update"] == "true") {
      $this->printUpdateXML($_GET);
    } else if (isset($_GET["action"])) {
      $this->printActionHTML($_GET);
    } else if (isset($_GET["script"])) {
      $this->printScript($_GET);
    } else {
      $this->printHTML($_GET);
    }
  }

}
$obj = new control();
$obj->handleRequest($_GET);
