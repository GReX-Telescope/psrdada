<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class control extends apsr_webpage 
{

  var $inst = "";
  var $pwc_dbs = array();
  var $pwc_dbs_str = "";

  function control()
  {
    apsr_webpage::apsr_webpage();
    $this->inst = new apsr();

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
  }

  function javaScriptCallback()
  {
  }

  function printJavaScriptHead()
  {
?>
    <style type='text/css'>
      table.control {

      }
      table.control td {
        padding-left: 2px;
        vertical-align: middle;
      }
    </style>

    <script type='text/javascript'>

      // by default, only poll/update every 20 seconds
      var poll_timeout;
      var poll_update = 20000;
      var poll_2sec_count = 0;

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

      var help_hosts = new Array(<?
      $helpers = getConfigMachines($this->inst->config, "DFB");
      $helpers = array_unique($helpers);
      asort($helpers);
      echo "'".$helpers[0]."'";
      for ($i=1; $i<count($helpers); $i++) {
        echo ",'".$helpers[$i]."'";
      }?>);

      var help_hosts_str = "";
      for (i=0; i<help_hosts.length; i++) {
        help_hosts_str += "&host_"+i+"="+help_hosts[i];
      }

      var stage2_wait = 20;
      var stage3_wait = 20;
      var stage4_wait = 20;
      var stage5_wait = 20;

      function poll_server()
      {
        document.getElementById("poll_update_secs").innerHTML= (poll_update / 1000);
        daemon_info_request();
        poll_timeout = setTimeout('poll_server()', poll_update);

        // revert poll_update to 20000 after 1 minute of 2 second polling
        if (poll_update == 1000)
        {
          poll_2sec_count ++;
          if (poll_2sec_count == 60)
          {
            poll_update = 20000;
            poll_2sec_count = 0;
          }
        }
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

          poll_update = 1000;
          clearTimeout(poll_timeout);
          poll_timeout = setTimeout('poll_server()', poll_update);

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

        poll_update = 1000;
        clearTimeout(poll_timeout);
        poll_timeout = setTimeout('poll_server()', poll_update);

      }

      function toggleDaemonPersist(host, daemon) {
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
          if (action == "start")
            url = "control.lib.php?action="+action+"&nhosts=1&host_0="+host+"&daemon="+daemon;
          else
            url = "control.lib.php?action="+action+"&nhosts=1&host_0="+host+"&daemon="+daemon;
          //alert(url);

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

          poll_update = 1000;
          clearTimeout(poll_timeout);
          poll_timeout = setTimeout('poll_server()', poll_update);
        } 
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
      
      function startApsr() 
      {

        poll_update = 1000;
        poll_2sec_count = 0;
        clearTimeout(poll_timeout);
        poll_timeout = setTimeout('poll_server()', poll_update);

        // start the server's master control script
        url = "control.lib.php?action=start&daemon=apsr_master_control&nhosts=1&host_0=srv0"
        daemon_action_request(url);

        // start the help's master control script
        url = "control.lib.php?action=start&daemon=apsr_master_control&nhosts="+help_hosts.length+help_hosts_str;
        daemon_action_request(url);

        // start the pwc's master control script
        url = "control.lib.php?action=start&daemon=apsr_master_control&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        stage2_wait = 20;
        startApsrStage2();
      }

      function startApsrStage2()
      {
        poll_2sec_count = 0;
        var machines = srv_hosts.concat(pwc_hosts, help_hosts);
        var daemons = new Array("apsr_master_control");
        var ready = checkMachinesAndDaemons(machines, daemons, "green_light.png");
        if ((!ready) && (stage2_wait > 0)) {
          stage2_wait--;
          setTimeout('startApsrStage2()', 1000);
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
          echo "         url = \"control.lib.php?action=start&daemon=buffer_".$key.
               "&args=".$args."&nhosts=\"+pwc_hosts.length+pwc_hosts_str\n";
          echo "         daemon_action_request(url);\n";
        }
?>

        stage3_wait = 20;
        startApsrStage3();

      }

      function startApsrStage3()
      {
        poll_2sec_count = 0;
        var pwc_daemons = new Array(<?echo $this->pwcs_dbs_str?>); 
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "green_light.png");

        if ((!pwc_ready) && (stage3_wait > 0)) {
          stage3_wait--;
          setTimeout('startApsrStage3()', 1000);
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
        startApsrStage4();
      }

      function startApsrStage4()
      {
        poll_2sec_count = 0;
        var pwc_daemons = new Array("pwcs");
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "green_light.png");

        var srv_daemons = new Array("apsr_pwc_monitor", "apsr_sys_monitor");
        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "green_light.png");

        if ((!(pwc_ready && srv_ready)) && (stage4_wait > 0)) {
          stage4_wait--;
          setTimeout('startApsrStage4()', 1000);
          return 0;
        }
        stage4_wait = 0;

        // start the pwc's daemons next
        url = "control.lib.php?action=start&daemon=all&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        stage5_wait = 20;
        startApsrStage5();

      }

      function startApsrStage5()
      {
        poll_2sec_count = 0;
        var pwc_daemons = new Array("apsr_observation_manager","apsr_archive_manager","apsr_disk_cleaner");
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "green_light.png");

        if ((!pwc_ready) && (stage5_wait > 0)) {
          stage5_wait--;
          setTimeout('startApsrStage5()', 1000);
          return 0;
        }
        stage5_wait = 0;

        // revert to 20 second polling
        poll_update = 20000;
      }

      function hardstopApsr()
      {
        // poll every 2 seconds during a stop
        poll_update = 1000;
        clearTimeout(poll_timeout);
        poll_2sec_count = 0;
        poll_timeout = setTimeout('poll_server()', poll_update);

        // stop server TCS interface
        url = "control.lib.php?script=apsr_hard_reset.pl";
        popUp(url);
      }

      function stopApsr()
      {
        // poll every 2 seconds during a stop
        poll_update = 1000;
        poll_2sec_count = 0;
        clearTimeout(poll_timeout);
        poll_timeout = setTimeout('poll_server()', poll_update);

        // stop server TCS interface
        url = "control.lib.php?action=stop&daemon=apsr_tcs_interface&nhosts=1&host_0=srv0";
        daemon_action_request(url);

        // stop the help daemons master control script 
        url = "control.lib.php?action=stop&daemon=apsr_master_control&nhosts="+help_hosts.length+help_hosts_str;
        daemon_action_request(url);

        // stop the pwc's daemons next
        url = "control.lib.php?action=stop&daemon=all&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        stage2_wait = 20;
        stopApsrStage2();
      }


      function stopApsrStage2()
      {
        poll_2sec_count = 0;

        var srv_daemons = new Array("apsr_tcs_interface");
        var help_daemons = new Array("apsr_master_control");
        var pwc_daemons = new Array("apsr_observation_manager","apsr_archive_manager","apsr_disk_cleaner");

        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "red_light.png");
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");
        var help_ready = checkMachinesAndDaemons(help_hosts, help_daemons, "red_light.png");

        if ((!(srv_ready && pwc_ready && help_ready)) && (stage2_wait > 0)) {
          stage2_wait--;
          setTimeout('stopApsrStage2()', 1000);
          return 0;
        }
        stage2_wait = 0;

        // stop the pwc's pwc
        url = "control.lib.php?action=stop&daemon=pwcs&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        stage3_wait = 20;
        stopApsrStage3();
      } 

      function stopApsrStage3()
      {
        poll_2sec_count = 0;

        var pwc_daemons = new Array("pwcs");
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");

        if ((!pwc_ready) && (stage3_wait > 0)) {
          stage3_wait--;
          setTimeout('stopApsrStage3()', 1000);
          return 0;
        }
        stage3_wait = 0;

        // destroy the pwc's datablocks
<?
        for ($i=0; $i<count($this->pwc_dbs); $i++)
        {
          $key = $this->pwc_dbs[$i]["key"];
          echo "         url = \"control.lib.php?action=stop&daemon=buffer_".$key.
               "&nhosts=\"+pwc_hosts.length+pwc_hosts_str;\n";
          echo "         daemon_action_request(url);\n";
        }
?>

        // stop the server daemons 
        url = "control.lib.php?action=stop&daemon=all&nhosts=1&host_0=srv0"
        daemon_action_request(url);

        stage4_wait = 20;
        stopApsrStage4();

      }
    
      function stopApsrStage4()
      {
        poll_2sec_count = 0;

        var srv_daemons = new Array("apsr_results_manager", "apsr_pwc_monitor", "apsr_sys_monitor", "apsr_web_monitor", "apsr_gain_manager");
        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "red_light.png");

        var pwc_daemons = new Array(<?echo $this->pwcs_dbs_str?>); 
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");

        if ((!(srv_ready && pwc_ready)) && (stage4_wait > 0)) {
          stage4_wait--;
          setTimeout('stopApsrStage4()', 1000);
          return 0;
        }
        stage4_wait = 0;

        // stop the pwc's master control script
        url = "control.lib.php?action=stop&daemon=apsr_master_control&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        // stop the server's master control script
        url = "control.lib.php?action=stop&daemon=apsr_master_control&nhosts=1&host_0=srv0"
        daemon_action_request(url);

        stage5_wait = 20;
        stopApsrStage5();
      }

      function stopApsrStage5()
      {
        poll_2sec_count = 0;
        var srv_daemons = new Array("apsr_master_control");
        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "red_light.png");

        var pwc_daemons = new Array("apsr_master_control");
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");

        if ((!(srv_ready && pwc_ready)) && (stage5_wait > 0)) {
          stage5_wait--;
          setTimeout('stopApsrStage5()', 1000);
          return 0;
        }
        stage5_wait = 0;

        // revert to 20 second polling
        poll_update = 20000;
      }

      function handle_daemon_info_request(xml_request) 
      {
        if (xml_request.readyState == 4) {

          var xmlDoc=xml_request.responseXML;
          if (xmlDoc != null)
          {
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

              try {
                var td = document.getElementById(host+"_message");
                td.innerHTML = "";
              } catch (e) {
                alert("host="+host+" host_message not defined");
              } 

              for ( key in this_result) {
                if (key == "host") {

                } else if (key == "message") {

                  value = this_result[key];
                  td.innerHTML = value;

                } else {
                  
                  value = this_result[key];
                  str = str + " "+key+" -> "+value;

                  var img = document.getElementById("img_"+host+"_"+key);
                  if (img == null) {
                    alert("img_"+host+"_"+key+" did not exist");
                  }
                  if (value == "2") {
                    img.src = "/images/green_light.png";
                  } else if (value == "1") {
                    img.src = "/images/yellow_light.png";
                  } else if (value == "0") {
                    if (key == "apsr_master_control") {
                      set_daemons_to_grey(host);
                    }
                    img.src = "/images/red_light.png";
                  } else {
                    img.src = "/images/grey_light.png";
                  } 
                } 
              }
            }
          }
        }
      }

      function daemon_info_request() 
      {
        var di_http_request;
        var url = "control.lib.php?update=true&nhosts="+(srv_hosts.length+pwc_hosts.length+help_hosts.length);
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
        for (i=0; i<help_hosts.length; i++)
        {
          url += "&host_"+j+"="+help_hosts[i];
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

?>
    <table cellpadding='5px'>
      <tr>
        <td>
<?
    $this->openBlockHeader("Instrument Controls");
?>
    <input type='button' value='Start' onClick="startApsr()">
    <input type='button' value='Stop' onClick="stopApsr()">
    <input type='button' value='Hard Kill' onClick="hardstopApsr()">
    <br/>
    <font size='-1'>[<span id="poll_update_secs"></span> sec updates]</font>
<?
    $this->closeBlockHeader();
?>
        </td>
      </tr>
      <tr>
        <td>
<?
    $this->openBlockHeader("Persistent Daemons");

    if (array_key_exists("SERVER_DAEMONS_PERSIST", $this->inst->config)) 
    {
      $server_daemons_persist = split(" ",$this->inst->config["SERVER_DAEMONS_PERSIST"]);
      $server_daemons_hash  = $this->inst->serverLogInfo();
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
        $this->printPersistentServerDaemonControl($d, $server_daemons_hash[$d]["name"], $host);
      }
?>
            </table>
          </td>
        </tr>
        <tr>
          <td height='50px' id='persist_output' valign='top'></td>
        </tr>
      </table>
<?
    }

    $this->closeBlockHeader();
?>
        </td>
      </tr>
      <tr>
        <td>
<?
    $this->openBlockHeader("Usage");
?>
    <p>Instrument Controls [above] can be used to start/stop/restart all of the required APSR daemons.</p>
    <p>Click on the red/green lights to toggle the respective daemons on/off.</p>
    <p>Use the Start/Stop buttons to turn on/off on all the machines in that section.</p>
    <p>Messages will appear indicating activity, but it may take a few seconds for the daemon lights to turn on/off.</p>
<?
    $this->closeBlockHeader();
?>
        </td>
      </tr>
    </table>
<?
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
  <title>APSR Controls</title>
  <link rel='shortcut icon' href='/apsr/images/apsr_favico.ico'/>
<?
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
    for ($i=0; $i<count($this->ejs); $i++)
      echo "   <script type='text/javascript' src='".$this->ejs[$i]."'></script>\n";
  
    $this->printJavaScriptHead();
?>
</head>


<body onload="poll_server()">
<div class="background"></div>
<div class="Main">
<?
  $this->printJavaScriptBody();
?>
  <table width='100%' cellpadding='0px'>
    <tr>
      <td style='vertical-align: top; width: 300px'>
<?
        $this->printSideBarHTML();
?>
      </td>
      <td style='vertical-align: top;'>
<?
        $this->printMainHTML();
?>
      </td>
    </tr>
  </table>
</div>
</body>
</html>
<?
  }

  function printMainHTML()
  {
?>
    <table width='100%' cellspacing='5px'>
      <tr>
        <td style='vertical-align: top;'>
<?

    ###########################################################################
    #
    # Server Daemons
    # 
    $this->openBlockHeader("Server Daemons");

    $config = $this->inst->config;

    $server_daemons_hash  = $this->inst->serverLogInfo();
    $server_daemons = split(" ", $config["SERVER_DAEMONS"]);
    $host = $config["SERVER_HOST"];
    $host = substr($host, 0, strpos($host,"."));
?>
    <table width='100%'>
      <tr>
        <td>
          <table class='control' id="srv_controls">
<?
    $this->printServerDaemonControl("apsr_master_control", "Master Control", $host);
    for ($i=0; $i < count($server_daemons); $i++) {
      $d = $server_daemons[$i];
      $this->printServerDaemonControl($d, $server_daemons_hash[$d]["name"], $host);
    }
    echo "<tr><td>Messages</td><td id='".$host."_message'></td></tr>\n";
?>
          </table>
        </td>
        <td width='300px' id='srv_output' valign='top'></td>
      </tr>

    </table>
<?
    $this->closeBlockHeader();
?>
    </td>
    </tr>
    <tr>
    <td style='vertical-align: top'>
<?

    ###########################################################################
    #
    # PWC Daemons
    #
    $this->openBlockHeader("Client Daemons");

    $pwcs    = getConfigMachines($config, "PWC");
    $client_daemons = split(" ",$config["CLIENT_DAEMONS"]);
    $client_daemons_hash =  $this->inst->clientLogInfo();
?>

    <table width='100%'>
      <tr>
        <td>
          <table class='control' id='pwc_controls'>
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

    $this->printClientDaemonControl("apsr_master_control", "Master&nbsp;Control", $pwcs, "daemon&name=".$d);

    for ($i=0; $i<count($this->pwc_dbs); $i++)
    {
      $key = $this->pwc_dbs[$i]["key"];
      $nbufs = $this->pwc_dbs[$i]["nbufs"];
      $bufsz =  $this->pwc_dbs[$i]["bufsz"];
      $this->printClientDBControl($key."&nbsp;DB", $pwcs, $key, $nbufs, $bufsz);
    }

    $this->printClientDaemonControl("pwcs", "PWC", $pwcs, "pwcs");
    # Print the client daemons
    for ($i=0; $i<count($client_daemons); $i++) {
      $d = $client_daemons[$i];
      $n = str_replace(" ", "&nbsp;", $client_daemons_hash[$d]["name"]);
      $this->printClientDaemonControl($d, $n, $pwcs, "daemon&name=".$d);
    }

    echo "<tr>\n";
    echo "  <td>Messages</td>\n";
    for ($i=0; $i<count($pwcs); $i++) {
      echo "<td id='".$pwcs[$i]."_message'></td>\n";
    }
    echo "</tr>\n";
?>
          </table>
        </td>
        <td width='300px' id='pwc_output' valign='top'></td>
      </tr>
    </table>
<?
    $this->closeBlockHeader();
?>
      </td>
    </tr>
    <tr>
      <td>
<?
    $this->openBlockHeader("Helper Machines");

    $helpers    = getConfigMachines($this->inst->config, "DFB");
    $helpers = array_unique($helpers);
    asort($helpers);
?>
    <table width='100%'>
      <tr>
        <td id="help_controls">
          <table class='control' id="help_controls">
            <tr>
              <td></td>
<?
    for ($i=0; $i<count($helpers); $i++) {
      echo "          <td style='text-align: center'><span title='".$helpers[$i]."'>".$i."</span></td>\n";
    }
?>
              <td></td>
            </tr>
<?
    # Print the master control first
    $d = "apsr_master_control";
    $this->printClientDaemonControl($d, "Master Control", $helpers, "daemon&name=".$d);

    echo "<tr>\n";
    echo " <td>Messages</td>";
    for  ($i=0; $i<count($helpers); $i++) {
      echo "  <td id='".$helpers[$i]."_message'></td>\n";
    }
    echo "</tr>\n";

?>
          </table>
        </td>
        <td width='300px' id='help_output' valign='top'></td>
      </tr>
    </table>
<?

    $this->closeBlockHeader();
?>
      </td>
    </tr>
  </table>
<?
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
      list ($sockets[$i], $results[$i]) = openSocket($host, $port, 4);
    }

    # write the commands
    for ($i=0; $i<count($results); $i++) {
      if (($results[$i] == "ok") && ($sockets[$i])) {
        socketWrite($sockets[$i], $cmd."\r\n");
      } else {
        $results[$i] = "fail";
        $responses[$i] = "";
      }
    }

    # read the responses
    for ($i=0; $i<count($results); $i++) {
      if (($results[$i] == "ok") && ($sockets[$i])) {
        $read = socketRead($sockets[$i]);
        $responses[$i] = rtrim($read);
      }
    }

    # close the sockets
    for ($i=0; $i<count($results); $i++) {
      if (($results[$i] == "ok") && ($sockets[$i])) {
        socket_close($sockets[$i]);
      }
    }

    # handle persistent daemons
    $persist_xml = "";
    if (array_key_exists("SERVER_DAEMONS_PERSIST", $this->inst->config))
    {
      $server_daemons_persist = explode(" ", $this->inst->config["SERVER_DAEMONS_PERSIST"]);
      list ($host, $domain)  = explode(".", $this->inst->config["SERVER_HOST"],2);
      $persist_xml = "<daemon_info><host>".$host."</host>";
      $running = array();
      for ($i=0; $i<count($server_daemons_persist); $i++)
      {
        $d = $server_daemons_persist[$i];

        # check if the script is running
        $cmd = "pgrep -f '^perl.*server_".$d.".pl'";
        $last = exec($cmd, $array, $rval);
        if ($rval == 0)
          $running[$d] = 1;
        else
          $running[$d] = 0;

        # check if the PID file exists
        if (file_exists($this->inst->config["SERVER_CONTROL_DIR"]."/".$d.".pid"))
          $running[$d]++;

        # update xml
        $persist_xml .= "<".$d.">".$running[$d]."</".$d.">";
      }
      $persist_xml .= "</daemon_info>\n";
    }

    # produce the xml
    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<daemon_infos>\n";

    for ($i=0; $i<count($hosts); $i++) {
      if ($results[$i] == "ok")
        $xml .= $responses[$i]."\n";
      else if ($responses[$i] == "")
        $xml .= "<daemon_info><host>".$hosts[$i]."</host><apsr_master_control>0</apsr_master_control><message></message></daemon_info>\n";
      else 
        $xml .= "<daemon_info><host>".$hosts[$i]."</host><apsr_master_control>0</apsr_master_control><message>".$responses[$i]."</message></daemon_info>\n";
    }
    $xml .= $persist_xml;
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

    # determine which type of request this is (srv, pwc or help)
    if ($hosts[0] == "srv0") 
    {  
      if (strpos($this->inst->config["SERVER_DAEMONS_PERSIST"], $daemon) !== FALSE)
        $area = "persist";
      else
        $area = "srv";
    }
    else {
      for ($i=0; $i<$this->inst->config["NUM_PWC"]; $i++)
        if ($this->inst->config["PWC_".$i] == $hosts[0])
          $area = "pwc";
      for ($i=0; $i<$this->inst->config["NUM_DFB"]; $i++)
        if ($this->inst->config["DFB_".$i] == $hosts[0])
          $area = "help";
    }
    if ($area == "") {
      echo "ERROR: could not determine area\n";
      exit(0);
    }

    echo $area."\n";
    echo $unique_id."\n";
    flush();

    # special case for starting/stopping persistent server daemons
    if ($area == "persist") {

      if ($action == "start") {
        echo "Starting ".$daemon." on srv0\n";
        flush();

        $cmd = "ssh -x -l dada srv0 'server_".$daemon.".pl";
        if ($args != "")
          $cmd .= " ".$args;
        $cmd .= "'";
        $output = array();
        $lastline = exec($cmd, $output, $rval);

      } else if ($action == "stop") {
        $quit_file = $this->inst->config["SERVER_CONTROL_DIR"]."/".$daemon.".quit";
        $pid_file = $this->inst->config["SERVER_CONTROL_DIR"]."/".$daemon.".pid";
        if (file_exists($pid_file))
        {
          echo "Stopping ".$daemon." on srv0\n";
          flush();

          $cmd = "touch ".$quit_file;
          $lastline = exec($cmd, $output, $rval);
          # wait for the PID file to be removed
          $max_wait = 10;
          while (file_exists($pid_file) && $max_wait > 0) {
            sleep(1);
            $max_wait--;
          }
          unlink($quit_file);
        }
        else
        {
          echo "No PID file [".$pid_file."] existed for ".$daemon." on srv0\n";
          flush();
        }
      } else {
        $html = "Unrecognized action [".$action."] for daemon [".$daemon."]\n";
        flush();
      }

    } else if (($daemon == "apsr_master_control") && ($action == "start")) {

      $html = "Starting master control on";
      if ($nhosts > 2)
      {
        $html .= " ".$hosts[0]." - ".$hosts[$nhosts-1];
      }
      else
      {
        for ($i=0; $i<$nhosts; $i++) {
          $html .= " ".$hosts[$i];
        }
      }
      echo $html."\n";
      flush();
      for ($i=0; $i<$nhosts; $i++) {
        if ($area == "srv") {
          $cmd = "ssh -x -l dada ".$hosts[$i]." client_apsr_master_control.pl";
        } else {
          $cmd = "ssh -x -l apsr ".$hosts[$i]." client_apsr_master_control.pl";
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
        $html .= str_replace("_", " ",$daemon)." on";
        if ($args != "")
          $cmd .= " ".$args;
      }

      # open the socket connections
      if (count($hosts) > 2)
        $html .= " ".$hosts[0]." - ".$hosts[count($hosts)-1];
      for ($i=0; $i<count($hosts); $i++) {
        $host = $hosts[$i];
        if (count($hosts) <= 2)
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
<body>
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

    $img = "<img border='0' id='".$img_id."' src='/images/".$colour."_light.png' width='15px' height='15px'>";
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

  function printPersistentServerDaemonControl($daemon, $name, $host) 
  {
    echo "  <tr>\n";
    echo "    <td>".$name."</td>\n";
    echo "    <td>".$this->statusLight($host, $daemon, "-1", "", "toggleDaemonPersist")."</td>\n";
    echo "  </tr>\n";
  }

  function printClientDaemonControl($daemon, $name, $hosts, $cmd) 
  {
    $host_str = "";
    echo "  <tr>\n";
    echo "    <td>".$name."</td>\n";
    for ($i=0; $i<count($hosts); $i++) {
      echo "    <td>".$this->statusLight($hosts[$i], $daemon, -1, "")."</td>\n";
      $host_str .= $hosts[$i]." ";
    }
    $host_str = rtrim($host_str);
    if ($cmd != "" ) {
      echo "    <td style='text-align: center;'>\n";
      echo "      <input type='button' value='Start' onClick=\"toggleDaemons('start', '".$daemon."', '".$host_str."')\">\n";
      echo "    </td>\n";
      echo "    <td>\n";
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
    echo "    <td>".$name."</td>\n";
    for ($i=0; $i<count($hosts); $i++) {
      echo "    <td>".$this->statusLight($hosts[$i], $daemon, -1, $nbufs."_".$bufsz)."</td>\n";
      $host_str .= $hosts[$i]." ";
    }
    $host_str = rtrim($host_str);
    echo "    <td style='text-align: center;'>\n";
    echo "      <input type='button' value='Init' onClick=\"toggleDaemons('start', '".$daemon_on."', '".$host_str."')\">\n";
    echo "    </td>\n";
    echo "    <td>\n";
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
