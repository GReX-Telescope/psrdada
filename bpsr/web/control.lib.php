<?PHP

error_reporting(E_ALL);
ini_set("display_errors", 1);

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class control extends bpsr_webpage 
{

  var $inst = "";
  var $pwc_dbs = array();
  var $pwc_dbs_str = "";

  var $pwc_list = array();
  var $host_list = array();

  var $server_host = "";

  function control()
  {
    bpsr_webpage::bpsr_webpage();
    $this->inst = new bpsr();
    $this->pwcs_dbs_str = "";

    $config = $this->inst->config;
    $data_block_ids = preg_split("/\s+/", $config["DATA_BLOCK_IDS"]);
    foreach ($data_block_ids as $dbid) 
    {
      array_push($this->pwc_dbs, $dbid);
      if ($this->pwcs_dbs_str == "") 
        $this->pwcs_dbs_str = "\"buffer_".$dbid."\"";
      else
        $this->pwcs_dbs_str .= ", \"buffer_".$dbid."\"";
    }

    $prev_host = "";
    for ($i=0; $i<$config["NUM_PWC"]; $i++) 
    {
      $host = $config["PWC_".$i];

      $exists = -1;
      for ($j=0; $j<count($this->host_list); $j++) {
        if (strpos($this->host_list[$j]["host"], $host) !== FALSE)
          $exists = $j;
      }

      if ($exists == -1) 
        array_push($this->host_list, array("host" => $host, "span" => 1));
      else
        $this->host_list[$exists]["span"]++;

      array_push($this->pwc_list, array("host" => $host, "span" => 1, "pwc" => $i));
      $prev_host = $host;
    }

    $this->server_host = $this->inst->config["SERVER_HOST"];
    if (strpos($this->server_host, ".") !== FALSE)
      list ($this->server_host, $domain)  = explode(".", $this->server_host);

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
        padding-left: 5px;
        vertical-align: middle;
      }
    </style>

    <script type='text/javascript'>

      // by default, only poll/update every 20 seconds
      var poll_timeout;
      var poll_update = 20000;
      var poll_2sec_count = 0;
      var srv_host = "<?echo $this->server_host?>";
      var srv_hosts = new Array(srv_host);

      var pwc_hosts = new Array(<?
      echo "'".$this->host_list[0]["host"]."'";
      for ($i=1; $i<count($this->host_list); $i++) {
        echo ",'".$this->host_list[$i]["host"]."'";
      }?>);


      var pwcs = new Array(<?
      echo "'".$this->inst->config["PWC_0"].":0'";
      for ($i=1; $i<$this->inst->config["NUM_PWC"]; $i++) {
        echo ",'".$this->inst->config["PWC_".$i].":".$i."'";
      }?>);

      var pwc_hosts_str = "";
      for (i=0; i<pwc_hosts.length; i++) {
        pwc_hosts_str += "&host_"+i+"="+pwc_hosts[i];
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

        // revert poll_update to 20000 after 1 minute of 1 second polling
        if (poll_update == 1000)
        {
          poll_2sec_count ++;
          if (poll_2sec_count == 30)
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
            //alert("only 1 line received: "+lines[0]);

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
          if (host.indexOf(":",0) != -1) {
            parts = host.split(":");
            host = parts[0];
            pwc = parts[1];
            url = "control.lib.php?action="+action+"&nhosts=1&host_0="+host+"&pwc="+pwc+"&daemon="+daemon;
          } else {
            url = "control.lib.php?action="+action+"&nhosts=1&host_0="+host+"&daemon="+daemon;
          }

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

      function toggleDaemonPID(host, daemon) {
        var img = document.getElementById("img_"+host+"_"+daemon);
        var src = new String(img.src);
        var i = document.getElementById(daemon+"_pid").selectedIndex;
        var pid = document.getElementById(daemon+"_pid").options[i].value;
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

        if ((pid == "") && (action == "start"))
        {
          alert("You must select a PID to Start the daemon with");
          action = "ignore";
        } 
        if (action != "ignore") {
          if (action == "start")
            url = "control.lib.php?action="+action+"&nhosts=1&host_0="+host+"&daemon="+daemon+"&args="+pid;
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
      function checkMachinesAndDaemons(m, d, c) 
      {
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

      function checkMachinesPWCsAndDaemons(m, d, c) 
      {
        //var i=0;
        var j=0;
        var ready = true;
        //for (i=0; i<m.length; i++) {
          for (j=0; j<pwcs.length; j++) {
            for (k=0; k<d.length; k++) {
              element = document.getElementById("img_"+pwcs[j]+"_"+d[k]);
              try {
                if (element.src.indexOf(c) == -1) {
                  ready = false;
                }
              } catch (e) {
                alert("checkMachinesPWCsAndDameons: [img_"+pwcs[j]+"_"+d[k]+"] c="+c+" did not exist");
              }
            }
          }
        //}
        return ready;
      }

      

      function startBpsr() 
      {

        poll_update = 1000;
        poll_2sec_count = 0;
        clearTimeout(poll_timeout);
        poll_timeout = setTimeout('poll_server()', poll_update);

        // start the server's master control script
        url = "control.lib.php?action=start&daemon=bpsr_master_control&nhosts=1&host_0="+srv_host
        daemon_action_request(url);

        // start the pwc's master control script
        url = "control.lib.php?action=start&daemon=bpsr_master_control&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        stage2_wait = 20;
        startBpsrStage2();
      }

      function startBpsrStage2()
      {
        poll_2sec_count = 0;
        var machines = srv_hosts.concat(pwc_hosts);
        var daemons = new Array("bpsr_master_control");
        var ready = checkMachinesAndDaemons(machines, daemons, "green_light.png");
        if ((!ready) && (stage2_wait > 0)) {
          stage2_wait--;
          setTimeout('startBpsrStage2()', 1000);
          return 0;
        }
        stage2_wait = 0;
                
        // init the pwc's datablocks
<?
        for ($i=0; $i<count($this->pwc_dbs); $i++)
        {
          $dbid = $this->pwc_dbs[$i];
          echo "         url = \"control.lib.php?action=start&daemon=buffer_".$dbid."&nhosts=\"+pwc_hosts.length+pwc_hosts_str\n";
          echo "         daemon_action_request(url);\n";
        }
?>

        stage3_wait = 20;
        startBpsrStage3();

      }

      function startBpsrStage3()
      {
        poll_2sec_count = 0;
        var pwc_daemons = new Array(<?echo $this->pwcs_dbs_str?>); 
        var pwc_ready = checkMachinesPWCsAndDaemons(pwc_hosts, pwc_daemons, "green_light.png");

        if ((!pwc_ready) && (stage3_wait > 0)) {
          stage3_wait--;
          setTimeout('startBpsrStage3()', 1000);
          return 0;
        }
        stage3_wait = 0;

        // start the pwc's pwc
        url = "control.lib.php?action=start&daemon=bpsr_pwc&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        // start the server daemons 
        url = "control.lib.php?action=start&daemon=all&nhosts=1&host_0="+srv_host;
        daemon_action_request(url);

        stage4_wait = 20;
        startBpsrStage4();
      }

      function startBpsrStage4()
      {
        poll_2sec_count = 0;
        var pwc_daemons = new Array("bpsr_pwc");
        var pwc_ready = checkMachinesPWCsAndDaemons(pwc_hosts, pwc_daemons, "green_light.png");

        var srv_daemons = new Array("bpsr_pwc_monitor", "bpsr_src_monitor", "bpsr_sys_monitor", "bpsr_tcs_interface", "bpsr_results_manager", "bpsr_roach_manager", "bpsr_web_monitor");
        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "green_light.png");

        if ((!(pwc_ready && srv_ready)) && (stage4_wait > 0)) {
          stage4_wait--;
          setTimeout('startBpsrStage4()', 1000);
          return 0;
        }
        stage4_wait = 0;

        // start the pwc's daemons next
        url = "control.lib.php?action=start&daemon=all&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        stage5_wait = 20;
        startBpsrStage5();

      }

      function startBpsrStage5()
      {
        poll_2sec_count = 0;
        //var pwc_daemons = new Array("bpsr_observation_manager","bpsr_results_monitor","bpsr_disk_cleaner","bpsr_transfer_raid");
        var pwc_daemons = new Array("bpsr_proc", "bpsr_events", "bpsr_auxiliary", "bpsr_heimdall", "bpsr_dumper", "bpsr_results_monitor","bpsr_disk_cleaner","bpsr_transfer_raid");
        var pwc_ready = checkMachinesPWCsAndDaemons(pwc_hosts, pwc_daemons, "green_light.png");

        if ((!pwc_ready) && (stage5_wait > 0)) {
          stage5_wait--;
          setTimeout('startBpsrStage5()', 1000);
          return 0;
        }
        stage5_wait = 0;

        // revert to 20 second polling
        poll_update = 20000;
      }

      function hardstopBpsr()
      {
        // poll every 2 seconds during a stop
        poll_update = 1000;
        clearTimeout(poll_timeout);
        poll_2sec_count = 0;
        poll_timeout = setTimeout('poll_server()', poll_update);

        // stop server TCS interface
        url = "control.lib.php?script=bpsr_hard_reset.pl";
        popUp(url);
      }

      function stopBpsr()
      {
        // poll every 2 seconds during a stop
        poll_update = 1000;
        poll_2sec_count = 0;
        clearTimeout(poll_timeout);
        poll_timeout = setTimeout('poll_server()', poll_update);

        // stop server TCS interface
        url = "control.lib.php?action=stop&daemon=bpsr_tcs_interface&nhosts=1&host_0="+srv_host;
        daemon_action_request(url);

        // stop the pwc's daemons next
        url = "control.lib.php?action=stop&daemon=all&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        stage2_wait = 20;
        stopBpsrStage2();
      }


      function stopBpsrStage2()
      {
        poll_2sec_count = 0;

        var srv_daemons = new Array("bpsr_tcs_interface");
        //var pwc_daemons = new Array("bpsr_observation_manager","bpsr_results_monitor","bpsr_disk_cleaner","bpsr_transfer_raid");
        var pwc_daemons = new Array("bpsr_proc", "bpsr_events", "bpsr_auxiliary", "bpsr_heimdall", "bpsr_dumper", "bpsr_results_monitor","bpsr_disk_cleaner","bpsr_transfer_raid");

        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "red_light.png");
        var pwc_ready = checkMachinesPWCsAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");

        if ((!(srv_ready && pwc_ready)) && (stage2_wait > 0)) {
          stage2_wait--;
          setTimeout('stopBpsrStage2()', 1000);
          return 0;
        }
        stage2_wait = 0;

        // stop the pwc's pwc
        url = "control.lib.php?action=stop&daemon=bpsr_pwc&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        stage3_wait = 20;
        stopBpsrStage3();
      } 

      function stopBpsrStage3()
      {
        poll_2sec_count = 0;

        var pwc_daemons = new Array("bpsr_pwc");
        var pwc_ready = checkMachinesPWCsAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");

        if ((!pwc_ready) && (stage3_wait > 0)) {
          stage3_wait--;
          setTimeout('stopBpsrStage3()', 1000);
          return 0;
        }
        stage3_wait = 0;

        // destroy the pwc's datablocks
<?
        for ($i=0; $i<count($this->pwc_dbs); $i++)
        {
          $dbid = $this->pwc_dbs[$i];
          echo "         url = \"control.lib.php?action=stop&daemon=buffer_".$dbid."&nhosts=\"+pwc_hosts.length+pwc_hosts_str;\n";
          echo "         daemon_action_request(url);\n";
        }
?>

        // stop the server daemons 
        url = "control.lib.php?action=stop&daemon=all&nhosts=1&host_0="+srv_host;
        daemon_action_request(url);

        stage4_wait = 20;
        stopBpsrStage4();

      }
    
      function stopBpsrStage4()
      {
        poll_2sec_count = 0;

        var srv_daemons = new Array("bpsr_results_manager", "bpsr_pwc_monitor", "bpsr_sys_monitor", "bpsr_src_monitor", "bpsr_web_monitor", "bpsr_roach_manager");
        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "red_light.png");

        var pwc_daemons = new Array(<?echo $this->pwcs_dbs_str?>); 
        var pwc_ready = checkMachinesPWCsAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");

        if ((!(srv_ready && pwc_ready)) && (stage4_wait > 0)) {
          stage4_wait--;
          setTimeout('stopBpsrStage4()', 1000);
          return 0;
        }
        stage4_wait = 0;

        // stop the pwc's master control script
        url = "control.lib.php?action=stop&daemon=bpsr_master_control&nhosts="+pwc_hosts.length+pwc_hosts_str;
        daemon_action_request(url);

        // stop the server's master control script
        url = "control.lib.php?action=stop&daemon=bpsr_master_control&nhosts=1&host_0="+srv_host
        daemon_action_request(url);

        stage5_wait = 20;
        stopBpsrStage5();
      }

      function stopBpsrStage5()
      {
        poll_2sec_count = 0;
        var srv_daemons = new Array("bpsr_master_control");
        var srv_ready = checkMachinesAndDaemons(srv_hosts, srv_daemons, "red_light.png");

        var pwc_daemons = new Array("bpsr_master_control");
        var pwc_ready = checkMachinesAndDaemons(pwc_hosts, pwc_daemons, "red_light.png");

        if ((!(srv_ready && pwc_ready)) && (stage5_wait > 0)) {
          stage5_wait--;
          setTimeout('stopBpsrStage5()', 1000);
          return 0;
        }
        stage5_wait = 0;

        // revert to 20 second polling
        poll_update = 20000;
      }

      // parse an XML entity returning an array of key/value paries
      function parseXMLValues(xmlObj, values, pids)
      {
        //alert("parseXMLValues("+xmlObj.nodeName+") childNodes.length="+xmlObj.childNodes.length);
        var j = 0;
        for (j=0; j<xmlObj.childNodes.length; j++) 
        {
          if (xmlObj.childNodes[j].nodeType == 1)
          {
            // special case for PWCs
            key = xmlObj.childNodes[j].nodeName;
            if (key == "pwc") 
            {
              pwc_id = xmlObj.childNodes[j].getAttribute("id");
              values[pwc_id] = new Array();
              parseXMLValues(xmlObj.childNodes[j], values[pwc_id], pids)
            }
            else
            {
              val = "";
              if (xmlObj.childNodes[j].childNodes.length == 1)
              {
                val = xmlObj.childNodes[j].childNodes[0].nodeValue; 
              }
              values[key] = val;
              if (xmlObj.childNodes[j].getAttribute("pid") != null)
              {
                pids[key] = xmlObj.childNodes[j].getAttribute("pid");
              }
            }
          }
        }
      }

      function setDaemon(host, pwc, key, value)
      {
        var img_id;

        if (pwc >= 0)
          img_id = "img_"+host+":"+pwc+"_"+key;
        else
          img_id = "img_"+host+"_"+key;

        var img  = document.getElementById(img_id);

        if (img == null) {
          alert(img_id + "  did not exist");
        }
        else
        {
          if (value == "2") {
            img.src = "/images/green_light.png";
          } else if (value == "1") {
            img.src = "/images/yellow_light.png";
          } else if (value == "0") {
            if (key == "bpsr_master_control") {
              set_daemons_to_grey(host);
            }
            img.src = "/images/red_light.png";
          } else {
            img.src = "/images/grey_light.png";
          }
        }
      }

      function handle_daemon_info_request(xml_request) 
      {
        if (xml_request.readyState == 4) {

          var xmlDoc=xml_request.responseXML;
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement; 

            var i, j, k, result, key, value, span, this_result, pids;

            var results = xmlObj.getElementsByTagName("daemon_info");

            // for each result returned in the XML DOC
            for (i=0; i<results.length; i++) {

              result = results[i];

              this_result = new Array();
              pids = new Array();

              parseXMLValues(result, this_result, pids);

              host = this_result["host"];

              for (key in this_result) 
              {
                //alert("key="+key)
                if (key == "host") 
                {

                } 
                else if (this_result[key] instanceof Array) 
                {
                  pwc = this_result[key];
                  pwc_id = key;
                  for (key in pwc)
                  {
                    value = pwc[key];
                    setDaemon(host, pwc_id, key, value)
                  }
                } 
                else 
                {
                  value = this_result[key];
                  setDaemon(host, -1, key, value);
                } 
              }

              for ( key in pids ) {
                try {
                  var select = document.getElementById(key+"_pid");

                  // disable changing of this select
                  if ((this_result[key] == "2") || (this_result[key] == "1")) {
                    select.disabled = true;
                  } else {
                    select.disabled = false;
                  }
                  for (j = 0; j < select.length; j++) {
                    if (select[j].value == pids[key]) {
                      if (pids[key] != "") 
                        select.selectedIndex = j;
                      else
                        if (select.disabled == true)
                          select.selectedIndex = j;
                    }
                  }
                } catch(e) {
                  alert("ERROR="+e);
                }
              }
            }
          }
        }
      }

      function daemon_info_request() 
      {
        var di_http_request;
        var url = "control.lib.php?update=true&nhosts="+(srv_hosts.length+pwc_hosts.length);
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
    Updating every <span id="poll_update_secs"></span> seconds<br>
    <input type='button' value='Start' onClick="startBpsr()">
    <input type='button' value='Stop' onClick="stopBpsr()">
    <input type='button' value='Hard Stop' onClick="hardstopBpsr()">
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
      $server_daemons_persist = explode(" ",$this->inst->config["SERVER_DAEMONS_PERSIST"]);
      $server_daemons_hash  = $this->inst->serverLogInfo();
      $host = $this->server_host;
      $pids = array("P858");
?>
      <table width='100%'>
        <tr>
          <td>
            <table class='control' id="persist_controls">
<?
      for ($i=0; $i < count($server_daemons_persist); $i++) 
      {
        $d = $server_daemons_persist[$i];
        if ($d == "bpsr_raid_pipeline")
          $this->printPersistentServerDaemonControl($d, $server_daemons_hash[$d]["name"], $host, "NA");
        else
          $this->printPersistentServerDaemonControl($d, $server_daemons_hash[$d]["name"], $host, $pids);
      }
?>
            </table>
          </td>
        </tr>
        <tr>
          <td height='60px' id='persist_output' valign='top'></td>
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
    <p>Instrument Controls can be used to start/stop/ the all required BPSR daemons</p>
    <p>Click on individual lights to toggle the respective daemons on/off. Start/Stop buttons control daemons on all machines</p>
    <p>Messages are printed indicating activity, but may take a few seconds for the daemon lights to respond</p>
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
  <title>BPSR Controls</title>
  <link rel='shortcut icon' href='/bpsr/images/bpsr_favico.ico'/>
<?
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
    for ($i=0; $i<count($this->ejs); $i++)
      echo "   <script type='text/javascript' src='".$this->ejs[$i]."'></script>\n";
  
    $this->printJavaScriptHead();
?>
</head>


<body onload="poll_server()">
<?
  $this->printJavaScriptBody();
?>
  <table width='100%' cellpadding='0px'>
    <tr>
      <td style='vertical-align: top; width: 220px'>
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
    $server_daemons = explode(" ", $config["SERVER_DAEMONS"]);
    $host = $this->server_host;
?>
    <table width='100%'>
      <tr>
        <td>
          <table class='control' id="srv_controls">
<?
    $this->printServerDaemonControl("bpsr_master_control", "Master Control", $host);
    for ($i=0; $i < count($server_daemons); $i++) {
      $d = $server_daemons[$i];
      $this->printServerDaemonControl($d, $server_daemons_hash[$d]["name"], $host);
    }
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

    $client_daemons = explode(" ",$config["CLIENT_DAEMONS"]);
    $client_daemons_hash =  $this->inst->clientLogInfo();
?>

    <table width='100%'>
      <tr>
        <td>
          <table class='control' id='pwc_controls'>
            <tr>
              <td>Host</td>
<?
    for ($i=0; $i<count($this->host_list); $i++) 
    {
      $host = $this->host_list[$i]["host"];
      $span = $this->host_list[$i]["span"];
      echo "          <td colspan='".$span."'>".$host."</td>\n";
    }
?>
            </tr>
            <tr>
              <td>PWC</td>
<?
    for ($i=0; $i<count($this->pwc_list); $i++) 
    {
      $host = $this->pwc_list[$i]["host"];
      $pwc  = $this->pwc_list[$i]["pwc"];
      echo "          <td style='text-align: center'><span title='".$host."_".$pwc."'>".$pwc."</span></td>\n";
    }
?>
              <td></td>
            </tr>
<?

    $this->printClientDaemonControl("bpsr_master_control", "Master&nbsp;Control", $this->host_list, "daemon&name=".$d);

    # Data Blocks
    for ($i=0; $i<count($this->pwc_dbs); $i++)
    {
      $id = $this->pwc_dbs[$i];
      $this->printClientDBControl("DB&nbsp;".$id, $this->pwc_list, $id);
    }

    # Primary Write Client
    #$this->printClientDaemonControl("bpsr_pwc", "PWC", $this->pwc_list, "pwcs");

    # Print the client daemons
    for ($i=0; $i<count($client_daemons); $i++) {
      $d = $client_daemons[$i];
      $n = str_replace(" ", "&nbsp;", $client_daemons_hash[$d]["name"]);
      $this->printClientDaemonControl($d, $n, $this->pwc_list, "daemon&name=".$d);
    }
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
    </table>
<?
  }

  #############################################################################
  #
  # print update information for the control page as XML
  #
  function printUpdateXML($get)
  {
    $port = $this->inst->config["CLIENT_MASTER_PORT"];
    $cmd = "daemon_info_xml";
    $nhosts = 0;

    if (array_key_exists("nhosts", $get))
      $nhosts = $get["nhosts"];
    $hosts = array();
    for ($i=0; $i<$nhosts; $i++) {
      $hosts[$i] = $get["host_".$i];
    }

    $host = "";
    $sockets = array();
    $results = array();
    $responses = array();

    # open the socket connections
    for ($i=0; $i<count($hosts); $i++) {
      $host = $hosts[$i];
      list ($sockets[$i], $results[$i]) = openSocket($host, $port);
    }

    # write the commands
    for ($i=0; $i<count($results); $i++) {
      if ($results[$i] == "ok") {
        socketWrite($sockets[$i], $cmd."\r\n");
      } else {
        $results[$i] = "fail";
        $responses[$i] = "";
      }
    }

    # read the responses
    for ($i=0; $i<count($results); $i++) {
      if (($results[$i] == "ok") && ($sockets[$i])) {
        list ($result, $response) = socketRead($sockets[$i]);
        if ($result == "ok")
          $responses[$i] = $response;
        else
          $responses[$i] = "";
      }
    }

    # close the sockets
    for ($i=0; $i<count($results); $i++) {
      if ($results[$i] == "ok") {
        socket_close($sockets[$i]);
      }
    }

    # handle persistent daemons
    $persist_xml = "";
    if (array_key_exists("SERVER_DAEMONS_PERSIST", $this->inst->config))
    {
      $server_daemons_persist = explode(" ", $this->inst->config["SERVER_DAEMONS_PERSIST"]);
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

        # get the PID for this persistent daemon
        $dpid = "NA";
        if (array_key_exists("SERVER_".$d."_PID_PORT", $this->inst->config))
        {
          $port = $this->inst->config["SERVER_".$d."_PID_PORT"];
          list ($sock, $result) = openSocket($this->inst->config["SERVER_HOST"], $port);
          if ($result == "ok")
          {
            socketWrite($sock, "get_pid\r\n");
            list ($result, $response) = socketRead($sock);
            if ($result == "ok")
              $dpid = $response;
            else
            socket_close($sock);
          }
          else
            $dpid = "";
        }

        # update xml
        if ($dpid != "NA")
          $persist_xml .= "<".$d." pid='".$dpid."'>".$running[$d]."</".$d.">";
        else
          $persist_xml .= "<".$d.">".$running[$d]."</".$d.">";
      }
      #$persist_xml .= "</daemon_info>\n";
    }

    # produce the xml
    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<daemon_infos>\n";

    for ($i=0; $i<count($hosts); $i++) {
      $xml .= "<daemon_info>";

      # if no response was available
      if ((!array_key_exists($i, $responses)) || ($responses[$i] == ""))
        $xml .= "<host>".$hosts[$i]."</host><bpsr_master_control>0</bpsr_master_control>";
      else 
        $xml .= $responses[$i];

      if ($hosts[$i] == $this->server_host)
        $xml .= $persist_xml;

      $xml .="</daemon_info>\n";
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
    $pwc = "";
    if (array_key_exists("pwc", $get))
      $pwc = $get["pwc"];
    $args = "";
    if (array_key_exists("args", $get))
      $args = $get["args"];
    $area = "";

    if (($nhosts == "") || ($action == "") || ($daemon == "") || ($hosts[0] == "")) {
      echo "ERROR: malformed GET parameters\n";
      exit(0);
    }

    # determine which type of request this is (srv or pwc)
    if ($hosts[0] == $this->server_host)
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
        echo "Starting ".$daemon." on ".$this->server_host."\n";
        flush();

        $cmd = "ssh -x -l dada ".$this->server_host." 'server_".$daemon.".pl";
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
          echo "Stopping ".$daemon." on ".$this->server_host."\n";
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
          echo "No PID file [".$pid_file."] existed for ".$daemon." on ".$this->server_host."\n";
          flush();
        }
      } else {
        $html = "Unrecognized action [".$action."] for daemon [".$daemon."]\n";
        flush();
      }

    } else if (($daemon == "bpsr_master_control") && ($action == "start")) {
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
          $cmd = "ssh -x -l dada ".$hosts[$i]." client_bpsr_master_control.pl";
        } else {
          $cmd = "ssh -x -l bpsr ".$hosts[$i]." client_bpsr_master_control.pl";
        }

        $output = array();
        $lastline = exec($cmd, $output, $rval);
      }
    } else {
      $sockets = array();
      $results = array();
      $responses = array();

      $port = $this->inst->config["CLIENT_MASTER_PORT"];

      $html = "";

      # all daemons started or stopped
      if ($daemon == "all") {
        $cmd = "cmd=".$action."_daemons";
        $html .= (($action == "start") ? "Starting" : "Stopping")." ";
        $html .= "all daemons on ";
      } 
      # the command is in related to datablocks
      else if (strpos($daemon, "buffer") === 0)
      {
        list ($junk, $db_id) = explode("_", $daemon);
        if ($action == "stop")
        {
          $cmd = "cmd=destroy_db&args=".$db_id;
          $html .= "Destroying DB ".$db_id." on";
        }
        else 
        {
          $cmd = "cmd=init_db&args=".$db_id;
          $html .= "Creating DB ".$db_id." on";
        }
      }
      # the command is related to a specified daemon
      else
      {
        $html .= (($action == "start") ? "Starting" : "Stopping")." ";
        $cmd = "cmd=".$action."_daemon&args=".$daemon;
        $html .= str_replace("_", " ",$daemon)." on";
        if ($args != "")
          $cmd .= " ".$args;
      }

      # if we are only running this command on a single PWC of a host
      if ($pwc != "")
      {
        $cmd .= "&pwc=".$pwc;
      }

      # open the socket connections
      if (count($hosts) > 2)
        $html .= " ".$hosts[0]." - ".$hosts[count($hosts)-1];
      for ($i=0; $i<count($hosts); $i++) {
        $host = $hosts[$i];
        if (count($hosts) <= 2)
        {
          $html .= " ".$host;
          if ($pwc != "")
            $html .= ":".$pwc;
        }
        
        $ntries = 5;
        $results[$i] = "not open";
        while (($results[$i] != "ok") && ($ntries > 0)) {
          #echo "[".gettimeofday(true)."] ".$ntries.": openSocket($host,$port)<BR>\n";
          #flush();
          list ($sockets[$i], $results[$i]) = openSocket($host, $port);
          if ($results[$i] != "ok") {
            #echo "[".gettimeofday(true)."] ".$ntries.": openSocket($host,$port) failed - sleep(2)<BR>\n";
            #flush();
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
          #echo "<BR>[".gettimeofday(true)."] ".$ntries.": socketWrite($i, $cmd)<BR>\n";
          #flush();
          socketWrite($sockets[$i], $cmd."\r\n");
          #echo "[".gettimeofday(true)."] ".$ntries.": socketWrite($i, $cmd) done<BR>\n";
          #flush();
        } else {
          echo "ERROR: failed to open socket to master control script for ".$host."\n";
        }
      }

      # read the responses
      for ($i=0; $i<count($results); $i++) 
      {
        $host = $hosts[$i];
        if ($results[$i] == "ok") {

          $done = 32;
          # multiple lines may be returned, final line is always an "ok" or "fail"
          $responses[$i] = "";
          while ($done > 0) {
            #echo "[".gettimeofday(true)."] ".$ntries.": socketRead($i)<BR>\n";
            #flush();
            list ($result, $response) = socketRead($sockets[$i]);
            #echo "[".gettimeofday(true)."] ".$ntries.": socketRead($i) done<BR>\n";
            #flush();
            if (($response == "ok") || ($response == "fail")) {
              #echo "[".gettimeofday(true)."] ".$ntries.": socketRead($i) read='".$read."'<BR>\n";
              #flush();
              $done = 0;
            } else {
              #echo "[".gettimeofday(true)."] ".$ntries.": socketRead($i) read='".$read."'<BR>\n";
              #flush();
              $done--;
            }
            if ($response != "")
              $responses[$i] .= $response."\n";
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
        $bits = explode("\n", $responses[$i]);
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
    putenv("DADA_ROOT=".DADA_ROOT);
    $script = $this->inst->config["SCRIPTS_DIR"]."/".$script_name." 2>&1";
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

  function printPersistentServerDaemonControl($daemon, $name, $host, $pids) 
  {
    echo "  <tr>\n";
    echo "    <td>".$name."</td>\n";
    if (is_array($pids))
      echo "    <td>".$this->statusLight($host, $daemon, "-1", "", "toggleDaemonPID")."</td>\n";
    else
      echo "    <td>".$this->statusLight($host, $daemon, "-1", "", "toggleDaemonPersist")."</td>\n";
    echo "    <td>\n";
    if (is_array($pids))
    {
      echo "      <select id='".$daemon."_pid'>\n";
      echo "        <option value=''>--</option>\n";
      for ($i=0; $i<count($pids); $i++)
      {
        echo "        <option value='".$pids[$i]."'>".$pids[$i]."</option>\n";
      } 
      echo "      </select>\n";
    }
    else
      echo $pids;
    echo "    </td>\n";
    echo "  </tr>\n";
  }

  function printClientDaemonControl($daemon, $name, $hosts, $cmd) 
  {
    $host_str = "";
    echo "  <tr>\n";
    echo "    <td>".$name."</td>\n";
    for ($i=0; $i<count($hosts); $i++) {
      $host = $hosts[$i]["host"];
      $span = $hosts[$i]["span"];
      $pwc = "";
      if (array_key_exists("pwc", $hosts[$i])) 
        $pwc = ":".$hosts[$i]["pwc"];
      
      echo "    <td colspan='".$span."' style='text-align: center;'>".$this->statusLight($host.$pwc, $daemon, -1, "")."</td>\n";
      if (strpos($host_str, $host) === FALSE)
        $host_str .= $host." ";
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
  function printClientDBControl($name, $hosts, $id) 
  {

    $daemon = "buffer_".$id;
    $daemon_on = "buffer_".$id;
    $daemon_off = "buffer_".$id;

    $host_str = "";
    echo "  <tr>\n";
    echo "    <td>".$name."</td>\n";
    for ($i=0; $i<count($hosts); $i++) {
      $host = $hosts[$i]["host"];
      $span = $hosts[$i]["span"]; 
      $pwc = "";
      if (array_key_exists("pwc", $hosts[$i])) 
        $pwc = ":".$hosts[$i]["pwc"];
      echo "    <td span='".$span."' style='text-align: center;'>".$this->statusLight($host.$pwc, $daemon, -1, "")."</td>\n";
      if (strpos($host_str, $host) === FALSE)
        $host_str .= $host." ";
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

    if (array_key_exists("update", $_GET) && ($_GET["update"] == "true")) {
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
