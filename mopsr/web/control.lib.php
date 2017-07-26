<?PHP

error_reporting(E_ALL);
ini_set("display_errors", 1);

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class control extends mopsr_webpage 
{
  var $inst = "";
  var $bf_cfg = 0;
  var $bp_cfg = 0;
  var $bs_cfg = 0;

  var $server_host = "";
  var $server_daemons = array();

  var $aq_daemons = array();
  var $aq_list = array();
  var $aq_dbs = array();
  var $aq_dbs_str = "";
  var $aq_host_list = array();

  var $bf_daemons = array();
  var $bf_list = array();
  var $bf_dbs = array();
  var $bf_dbs_str = "";
  var $bf_host_list = array();

  var $bp_daemons = array();
  var $bp_list = array();
  var $bp_dbs = array();
  var $bp_dbs_str = "";
  var $bp_host_list = array();

  var $bs_daemons = array();
  var $bs_list = array();
  var $bs_dbs = array();
  var $bs_dbs_str = "";
  var $bs_host_list = array();

  function control()
  {
    mopsr_webpage::mopsr_webpage();
    $this->inst = new mopsr();

    # get the Beam Former configuration
    $this->bf_cfg = $this->inst->configFileToHash(BF_FILE);

    # get the Beam Processor configuration
    $this->bp_cfg = $this->inst->configFileToHash(BP_FILE);

    # get the Beam Smirf configuration
    $this->bs_cfg = $this->inst->configFileToHash(BS_FILE);

    $config = $this->inst->config;

    # get a list of the server daemons 
    $this->server_daemons = split(" ", $config["SERVER_DAEMONS"]);

    $this->aq_dbs_str = "";

    $data_block_ids = preg_split("/\s+/", $config["DATA_BLOCK_IDS"]);
    foreach ($data_block_ids as $dbid) 
    {
      array_push($this->aq_dbs, $dbid);
      if ($this->aq_dbs_str == "") 
        $this->aq_dbs_str = "\"buffer_".$dbid."\"";
      else
        $this->aq_dbs_str .= ", \"buffer_".$dbid."\"";
    }

    for ($i=0; $i<$config["NUM_PWC"]; $i++) 
    {
      $host = $config["PWC_".$i];

      $exists = -1;
      for ($j=0; $j<count($this->aq_host_list); $j++) {
        if (strpos($this->aq_host_list[$j]["host"], $host) !== FALSE)
          $exists = $j;
      }
      if ($exists == -1)
        array_push($this->aq_host_list, array("host" => $host, "span" => 1));
      else
        $this->aq_host_list[$exists]["span"]++;

      array_push($this->aq_list, array("host" => $host, "span" => 1, "pwc" => $i));
    }

    # get the list of aq client daemons
    $ds = split(" ", $config["CLIENT_DAEMONS"]);
    foreach ($ds as $d)
    {
      if (($d != "mopsr_pwc")  && ($d != "mopsr_results_monitor"))
      {
        array_push ($this->aq_daemons, $d);
      }
    }

    # beam former configuration
    $config = $this->bf_cfg;
    if ($config["NUM_BF"] > 0)
    {
      $data_block_ids = preg_split("/\s+/", $config["DATA_BLOCK_IDS"]);
      foreach ($data_block_ids as $dbid)
      {
        array_push($this->bf_dbs, $dbid);
        if ($this->bf_dbs_str == "")
          $this->bf_dbs_str = "\"buffer_".$dbid."\"";
        else
          $this->bf_dbs_str .= ", \"buffer_".$dbid."\"";
      }

      for ($i=0; $i<$config["NUM_BF"]; $i++)
      {
        $host = $config["BF_".$i];

        $exists = -1;
        for ($j=0; $j<count($this->bf_host_list); $j++) {
          if (strpos($this->bf_host_list[$j]["host"], $host) !== FALSE)
            $exists = $j;
        }
        if ($exists == -1)
          array_push($this->bf_host_list, array("host" => $host, "span" => 1));
        else
          $this->bf_host_list[$exists]["span"]++;

        array_push($this->bf_list, array("host" => $host, "span" => 1, "bf" => $i));
      }

      # get the list of aq client daemons
      $this->bf_daemons = split(" ", $config["CLIENT_DAEMONS"]);
    }

    # beam processor configuration
    $config = $this->bp_cfg;
    if ($config["NUM_BP"] > 0)
    {
      $data_block_ids = preg_split("/\s+/", $config["DATA_BLOCK_IDS"]);
      foreach ($data_block_ids as $dbid)
      {
        array_push($this->bp_dbs, $dbid);
        if ($this->bp_dbs_str == "")
          $this->bp_dbs_str = "\"buffer_".$dbid."\"";
        else
          $this->bp_dbs_str .= ", \"buffer_".$dbid."\"";
      }

      for ($i=0; $i<$config["NUM_BP"]; $i++)
      {
        $host = $config["BP_".$i];

        $exists = -1;
        for ($j=0; $j<count($this->bp_host_list); $j++) {
          if (strpos($this->bp_host_list[$j]["host"], $host) !== FALSE)
            $exists = $j;
        }
        if ($exists == -1)
          array_push($this->bp_host_list, array("host" => $host, "span" => 1));
        else
          $this->bp_host_list[$exists]["span"]++;

        array_push($this->bp_list, array("host" => $host, "span" => 1, "bp" => $i));
      }

      # get the list of aq client daemons
      $this->bp_daemons = split(" ", $config["CLIENT_DAEMONS"]);
    }

    # beam smirf configuration
    $config = $this->bs_cfg;
    if ($config["NUM_BS"] > 0)
    {
      $data_block_ids = preg_split("/\s+/", $config["DATA_BLOCK_IDS"]);
      foreach ($data_block_ids as $dbid)
      {
        array_push($this->bs_dbs, $dbid);
        if ($this->bs_dbs_str == "")
          $this->bs_dbs_str = "\"buffer_".$dbid."\"";
        else
          $this->bs_dbs_str .= ", \"buffer_".$dbid."\"";
      }

      for ($i=0; $i<$config["NUM_BS"]; $i++)
      {
        $host = $config["BS_".$i];

        $exists = -1;
        for ($j=0; $j<count($this->bs_host_list); $j++) {
          if (strpos($this->bs_host_list[$j]["host"], $host) !== FALSE)
            $exists = $j;
        }
        if ($exists == -1)
          array_push($this->bs_host_list, array("host" => $host, "span" => 1));
        else
          $this->bs_host_list[$exists]["span"]++;

        array_push($this->bs_list, array("host" => $host, "span" => 1, "bp" => $i));
      }

      # get the list of aq client daemons
      $this->bs_daemons = split(" ", $config["CLIENT_DAEMONS"]);
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

      var aq_hosts = new Array(<?
      for ($i=0; $i<count($this->aq_host_list); $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->aq_host_list[$i]["host"]."'";
      }?>);

      var aqs = new Array(<?
      for ($i=0; $i<$this->inst->config["NUM_PWC"]; $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->inst->config["PWC_".$i].":".$i."'";
      }?>);

      var aq_hosts_str = "";
      for (i=0; i<aq_hosts.length; i++) {
        aq_hosts_str += "&host_"+i+"="+aq_hosts[i];
      }

      var aq_daemons_custom = new Array(<?;
      for ($i=0; $i<count($this->aq_daemons); $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->aq_daemons[$i]."'";
      }?>);

      var bf_hosts = new Array(<?
      for ($i=0; $i<count($this->bf_host_list); $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->bf_host_list[$i]["host"]."'";
      }?>);

      var bfs = new Array(<?
      for ($i=0; $i<$this->bf_cfg["NUM_BF"]; $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->bf_cfg["BF_".$i].":".$i."'";
      }?>);

      var bf_hosts_str = "";
      for (i=0; i<bf_hosts.length; i++) {
        bf_hosts_str += "&host_"+i+"="+bf_hosts[i];
      }

      var bf_daemons_custom = new Array(<?;
      for ($i=0; $i<count($this->bf_daemons); $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->bf_daemons[$i]."'";
      }?>);

      var bp_hosts = new Array(<?
      for ($i=0; $i<count($this->bp_host_list); $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->bp_host_list[$i]["host"]."'";
      }?>);

      var bps = new Array(<?
      for ($i=0; $i<$this->bp_cfg["NUM_BP"]; $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->bp_cfg["BP_".$i].":".$i."'";
      }?>);

      var bp_hosts_str = "";
      for (i=0; i<bp_hosts.length; i++) {
        bp_hosts_str += "&host_"+i+"="+bp_hosts[i];
      }

      var bp_daemons_custom = new Array(<?;
      for ($i=0; $i<count($this->bp_daemons); $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->bp_daemons[$i]."'";
      }?>);

      var bs_hosts = new Array(<?
      for ($i=0; $i<count($this->bs_host_list); $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->bs_host_list[$i]["host"]."'";
      }?>);

      var bss = new Array(<?
      for ($i=0; $i<$this->bs_cfg["NUM_BS"]; $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->bs_cfg["BS_".$i].":".$i."'";
      }?>);

      var bs_hosts_str = "";
      for (i=0; i<bs_hosts.length; i++) {
        bs_hosts_str += "&host_"+i+"="+bs_hosts[i];
      }

      var bs_daemons_custom = new Array(<?;
      for ($i=0; $i<count($this->bs_daemons); $i++) {
        if ($i > 0) echo ",";
        echo "'".$this->bs_daemons[$i]."'";
      }?>);

      var srv_daemons_custom = new Array(<?;
      $first = 1;
      for ($i=0; $i<count($this->server_daemons); $i++)
        if ($this->server_daemons[$i] != "mopsr_tmc_interface")
        {
          if ($first)
          {
            echo "'".$this->server_daemons[$i]."'";
            $first = 0;
          }
          else
            echo ",'".$this->server_daemons[$i]."'";
        }?>);


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
    
      function toggleDaemon (area, host, daemon, args) {
        var id = "img_"+area+"_"+host+"_"+daemon;
        var img = document.getElementById(id);
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
            url = "control.lib.php?area="+area+"&action="+action+"&nhosts=1&host_0="+host+"&pwc="+pwc+"&daemon="+daemon;
          } else {
            url = "control.lib.php?area="+area+"&action="+action+"&nhosts=1&host_0="+host+"&daemon="+daemon;
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

      function toggleDaemons(action, daemon, host_string, area)
      {
        var hosts = host_string.split(" ");
        var i = 0;
        var url = "control.lib.php?area="+area+"&action="+action+"&daemon="+daemon+"&nhosts="+hosts.length;
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

      function toggleDaemonPersist (host, daemon) {
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
          url = "control.lib.php?area=persist&action="+action+"&nhosts=1&host_0="+host+"&daemon="+daemon;
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
      function checkMachinesAndDaemons(a, m, d, c) 
      {
        var i=0;
        var j=0;
        var ready = true;
        for (i=0; i<m.length; i++) {
          for (j=0; j<d.length; j++) {
            element = document.getElementById("img_"+a+"_"+m[i]+"_"+d[j]);
            try {
              if (element.src.indexOf(c) == -1) {
                ready = false;
              }
            } catch (e) {
              alert("checkMachinesAndDameons: a="+a+" m="+m[i]+", d="+d[j]+", c="+c+" did not exist");
            }
          }
        }
        return ready;
      }

      function checkMachinesPWCsAndDaemons(a, m, d, c) 
      {
        if ( Object.prototype.toString.call( c ) === '[object Array]' ) {
          cs = c;
        } else {
          cs = new Array(c);
        }

        var j=0;
        var ready = true;
        var pwcs;
        
        if (a == "aq")
          pwcs = aqs;
        else if (a == "bf")
          pwcs = bfs;
        else if (a == "bp")
          pwcs = bps;
        else if (a == "bs")
          pwcs = bss;
        else
        {
          alert ("checkMachinesPWCsAndDaemons: unexpected area: "+a);
          return
        }

        for (j=0; j<pwcs.length; j++) {
          for (k=0; k<d.length; k++) {
            element = document.getElementById("img_"+a+"_"+pwcs[j]+"_"+d[k]);
            var any_match = false;
            for (i=0; i<cs.length; i++) {
              try {
                if (element.src.indexOf(cs[i]) != -1) {
                  any_match = true;
                }
              } catch (e) {
                alert("checkMachinesPWCsAndDameons: [img_"+a+"_"+pwcs[j]+"_"+d[k]+"] c="+cs[i]+" did not exist");
              }
            }
            if (!any_match)
              ready = false;
          }
        }
        return ready;
      }


      function startMopsr() 
      {
        poll_update = 1000;
        poll_2sec_count = 0;
        clearTimeout(poll_timeout);
        poll_timeout = setTimeout('poll_server()', poll_update);

        // start the server's master control script
        url = "control.lib.php?area=srv&action=start&daemon=mopsr_master_control&nhosts=1&host_0="+srv_host
        daemon_action_request(url);

        // start the AQ's master control script
        url = "control.lib.php?area=aq&action=start&daemon=mopsr_master_control&nhosts="+aq_hosts.length+aq_hosts_str;
        daemon_action_request(url);

        // start the BF's master control script
        if (bf_hosts.length > 0)
        {
          url = "control.lib.php?area=bf&action=start&daemon=mopsr_bf_master_control&nhosts="+bf_hosts.length+bf_hosts_str;
          daemon_action_request(url);
        }

        // start the BP's master control script
        if (bp_hosts.length > 0)
        {
          url = "control.lib.php?area=bp&action=start&daemon=mopsr_bp_master_control&nhosts="+bp_hosts.length+bp_hosts_str;
          daemon_action_request(url);
        }

        // start the BS's master control script
        if (bs_hosts.length > 0)
        {
          url = "control.lib.php?area=bp&action=start&daemon=mopsr_bs_master_control&nhosts="+bs_hosts.length+bs_hosts_str;
          daemon_action_request(url);
        }

        stage2_wait = 20;
        startMopsrStage2();
      }

      function startMopsrStage2()
      {
        poll_2sec_count = 0;

        var daemons = new Array("mopsr_master_control");

        var srv_ready = checkMachinesAndDaemons("srv", srv_hosts, daemons, "green_light.png");
        var aq_ready  = checkMachinesAndDaemons("aq", aq_hosts, daemons, "green_light.png");

        var bf_daemons;
        var bf_ready = true;      
        if (bf_hosts.length > 0)
        {
          bf_daemons = new Array("mopsr_bf_master_control");
          bf_ready  = checkMachinesAndDaemons("bf", bf_hosts, bf_daemons, "green_light.png");
        }

        var bp_daemons;
        var bp_ready = true;
        if (bp_hosts.length > 0)
        {
          var bp_daemons = new Array("mopsr_bp_master_control");
          var bp_ready  = checkMachinesAndDaemons("bp", bp_hosts, bp_daemons, "green_light.png");
        }

        var bs_daemons;
        var bs_ready = true;
        if (bs_hosts.length > 0)
        {
          var bs_daemons = new Array("mopsr_bs_master_control");
          var bs_ready  = checkMachinesAndDaemons("bs", bs_hosts, bs_daemons, "green_light.png");
        }

        if (((!srv_ready) || (!aq_ready) || (!bf_ready) || (!bp_ready) || (!bs_ready)) && (stage2_wait > 0))
        {
          stage2_wait--;
          setTimeout('startMopsrStage2()', 1000);
          return 0;
        }
        stage2_wait = 0;
                
        // init the AQ, BF, BP, BS datablocks
<?
        for ($i=0; $i<count($this->aq_dbs); $i++)
        {
          $dbid = $this->aq_dbs[$i];
          echo "         url = \"control.lib.php?area=aq&action=start&daemon=buffer_".$dbid."&nhosts=\"+aq_hosts.length+aq_hosts_str\n";
          echo "         daemon_action_request(url);\n";
        }

        for ($i=0; $i<count($this->bf_dbs); $i++)
        {
          $dbid = $this->bf_dbs[$i];
          echo "         url = \"control.lib.php?area=bf&action=start&daemon=buffer_".$dbid."&nhosts=\"+bf_hosts.length+bf_hosts_str\n";
          echo "         daemon_action_request(url);\n";
        }

        for ($i=0; $i<count($this->bp_dbs); $i++)
        {
          $dbid = $this->bp_dbs[$i];
          echo "         url = \"control.lib.php?area=bp&action=start&daemon=buffer_".$dbid."&nhosts=\"+bp_hosts.length+bp_hosts_str\n";
          echo "         daemon_action_request(url);\n";
        }

        for ($i=0; $i<count($this->bs_dbs); $i++)
        {
          $dbid = $this->bs_dbs[$i];
          echo "         url = \"control.lib.php?area=bs&action=start&daemon=buffer_".$dbid."&nhosts=\"+bs_hosts.length+bs_hosts_str\n";
          echo "         daemon_action_request(url);\n";
        }
?>

        for (var i=0; i<srv_daemons_custom.length; i++)
        {
          url = "control.lib.php?area=srv&action=start&daemon=" + srv_daemons_custom[i] + "&nhosts=1&host_0="+srv_host;
          daemon_action_request(url);
        }

        stage3_wait = 60;
        startMopsrStage3();

      }

      function startMopsrStage3()
      {
        poll_2sec_count = 0;

        var aq_daemons = new Array(<?echo $this->aq_dbs_str?>); 
        var aq_ready = checkMachinesPWCsAndDaemons("aq", aq_hosts, aq_daemons, new Array("green_light.png", "grey_light.png"));

        var bf_daemons;
        var bf_ready = true;
        if (bf_hosts.length > 0)
        {
          bf_daemons = new Array(<?echo $this->bf_dbs_str?>); 
          bf_ready = checkMachinesPWCsAndDaemons("bf", bf_hosts, bf_daemons, new Array("green_light.png", "grey_light.png"));
        }

        var bp_daemons;
        var bp_ready = true;
        if (bp_hosts.length > 0)
        {
          bp_daemons = new Array(<?echo $this->bp_dbs_str?>); 
          bp_ready = checkMachinesPWCsAndDaemons("bp", bp_hosts, bp_daemons, new Array("green_light.png", "grey_light.png"));
        }

        var bs_daemons;
        var bs_ready = true;
        if (bs_hosts.length > 0)
        {
          bs_daemons = new Array(<?echo $this->bs_dbs_str?>); 
          bs_ready = checkMachinesPWCsAndDaemons("bs", bs_hosts, bs_daemons, new Array("green_light.png", "grey_light.png"));
        }


        if (((!aq_ready) || (!bf_ready) || (!bp_ready) || (!bs_ready)) && (stage3_wait > 0))
        {
          stage3_wait--;
          setTimeout('startMopsrStage3()', 1000);
          return 0;
        }
        stage3_wait = 0;

        // start the AQ's PWC daemon
        url = "control.lib.php?area=aq&action=start&daemon=mopsr_pwc&nhosts="+aq_hosts.length+aq_hosts_str;
        daemon_action_request(url);
        url = "control.lib.php?area=aq&action=start&daemon=mopsr_results_monitor&nhosts="+aq_hosts.length+aq_hosts_str;
        daemon_action_request(url);

        // start all the BF's daemons
        if (bf_hosts.length > 0)
        {
          url = "control.lib.php?area=bf&action=start&daemon=all&nhosts="+bf_hosts.length+bf_hosts_str;
          daemon_action_request(url);
        }

        // start all the BP's daemons
        if (bp_hosts.length > 0)
        {
          url = "control.lib.php?area=bp&action=start&daemon=all&nhosts="+bp_hosts.length+bp_hosts_str;
          daemon_action_request(url);
        }

        // start all the BS's daemons
        if (bs_hosts.length > 0)
        {
          url = "control.lib.php?area=bs&action=start&daemon=all&nhosts="+bs_hosts.length+bs_hosts_str;
          daemon_action_request(url);
        }


        stage4_wait = 20;
        startMopsrStage4();
      }

      function startMopsrStage4()
      {
        poll_2sec_count = 0;

        var aq_daemons = new Array("mopsr_pwc", "mopsr_results_monitor");
        var aq_ready = checkMachinesPWCsAndDaemons("aq", aq_hosts, aq_daemons, "green_light.png");

        var bf_daemons;
        var bf_ready = true;
        if (bf_hosts.length > 0)
        {
          bf_daemons = bf_daemons_custom;
          bf_ready = checkMachinesPWCsAndDaemons("bf", bf_hosts, bf_daemons, "green_light.png");
        }

        var bp_daemons;
        var bp_ready = true;
        if (bp_hosts.length > 0)
        {
          bp_daemons = bp_daemons_custom;
          bp_ready = checkMachinesPWCsAndDaemons("bp", bp_hosts, bp_daemons, "green_light.png");
        }

        var bs_daemons;
        var bs_ready = true;
        if (bs_hosts.length > 0)
        {
          bs_daemons = bs_daemons_custom;
          bs_ready = checkMachinesPWCsAndDaemons("bs", bs_hosts, bs_daemons, "green_light.png");
        }

        if (((!aq_ready) || (!bf_ready) || (!bp_ready) || (!bs_ready)) && (stage4_wait > 0))
        {
          stage4_wait--;
          setTimeout('startMopsrStage4()', 1000);
          return 0;
        }
        stage4_wait = 0;

        // start the server TMC interface only
        url = "control.lib.php?area=srv&action=start&daemon=mopsr_tmc_interface&nhosts=1&host_0="+srv_host;
        daemon_action_request(url);

        // start the pwc's other daemons next
        url = "control.lib.php?area=aq&action=start&daemon=all&nhosts="+aq_hosts.length+aq_hosts_str;
        daemon_action_request(url);

        stage5_wait = 20;
        startMopsrStage5();
      }


      function startMopsrStage5()
      {
        poll_2sec_count = 0;

        var srv_daemons = new Array("mopsr_tmc_interface");
        var srv_ready = checkMachinesAndDaemons("srv", srv_hosts, srv_daemons, "green_light.png");

        // only for active
        var aq_daemons = aq_daemons_custom;
        var aq_lights = new Array("green_light.png");
        var aq_ready = checkMachinesPWCsAndDaemons("aq", aq_hosts, aq_daemons, aq_lights);

        if (((!srv_ready) || (!aq_ready)) && (stage5_wait > 0))
        {
          stage5_wait--;
          setTimeout('startMopsrStage5()', 1000);
          return 0;
        }
        stage5_wait = 0;

        // revert to 20 second polling
        poll_update = 20000;
      }

      function hardstopMopsr()
      {
        // poll every 2 seconds during a stop
        poll_update = 1000;
        clearTimeout(poll_timeout);
        poll_2sec_count = 0;
        poll_timeout = setTimeout('poll_server()', poll_update);

        // stop server TMC interface
        url = "control.lib.php?script=mopsr_hard_reset.csh";
        popUp(url);
      }

      function stopMopsr()
      {
        // poll every 2 seconds during a stop
        poll_update = 1000;
        poll_2sec_count = 0;
        clearTimeout(poll_timeout);
        poll_timeout = setTimeout('poll_server()', poll_update);

        // stop server TMC interface
        url = "control.lib.php?area=srv&action=stop&daemon=mopsr_tmc_interface&nhosts=1&host_0="+srv_host;
        daemon_action_request(url);

        // stop the AQ daemons
        url = "control.lib.php?area=aq&action=stop&daemon=all&nhosts="+aq_hosts.length+aq_hosts_str;
        daemon_action_request(url);

        // stop the BF daemons
        if (bf_hosts.length > 0)
        {
          url = "control.lib.php?area=bf&action=stop&daemon=all&nhosts="+bf_hosts.length+bf_hosts_str;
          daemon_action_request(url);
        }

        // stop the BP daemons
        if (bp_hosts.length > 0)
        {
          url = "control.lib.php?area=bp&action=stop&daemon=all&nhosts="+bp_hosts.length+bp_hosts_str;
          daemon_action_request(url);
        }

        // stop the BS daemons
        if (bs_hosts.length > 0)
        {
          url = "control.lib.php?area=bs&action=stop&daemon=all&nhosts="+bs_hosts.length+bs_hosts_str;
          daemon_action_request(url);
        }

        stage2_wait = 20;
        stopMopsrStage2();
      }


      function stopMopsrStage2()
      {
        poll_2sec_count = 0;

        var srv_daemons = new Array("mopsr_tmc_interface");

        var aq_daemons = aq_daemons_custom.concat(Array("mopsr_results_monitor", "mopsr_pwc"));
        var aq_lights = new Array("red_light.png", "grey_light.png");
        var aq_ready = checkMachinesPWCsAndDaemons("aq", aq_hosts, aq_daemons, aq_lights);

        var bf_daemons;
        var bf_ready = true;
        if (bf_hosts.length > 0)
        {
          bf_daemons = bf_daemons_custom;
          bf_ready = checkMachinesPWCsAndDaemons("bf", bf_hosts, bf_daemons, "red_light.png");
        }

        var bp_daemons;
        var bp_ready = true;
        if (bp_hosts.length > 0)
        {
          bp_daemons = bp_daemons_custom;
          bp_ready = checkMachinesPWCsAndDaemons("bp", bp_hosts, bp_daemons, "red_light.png");
        }

        var bs_daemons;
        var bs_ready = true;
        if (bs_hosts.length > 0)
        {
          bs_daemons = bs_daemons_custom;
          bs_ready = checkMachinesPWCsAndDaemons("bs", bs_hosts, bs_daemons, "red_light.png");
        }

        var srv_ready = checkMachinesAndDaemons("srv", srv_hosts, srv_daemons, "red_light.png");

        if ((!(srv_ready && aq_ready && bf_ready && bp_ready & bs_ready)) && (stage2_wait > 0))
        {
          stage2_wait--;
          setTimeout('stopMopsrStage2()', 1000);
          return 0;
        }
        stage2_wait = 0;

<?
        // destroy AQ datablocks
        for ($i=0; $i<count($this->aq_dbs); $i++)
        {
          $dbid = $this->aq_dbs[$i];
          echo "         url = \"control.lib.php?area=aq&action=stop&daemon=buffer_".$dbid."&nhosts=\"+aq_hosts.length+aq_hosts_str;\n";
          echo "         daemon_action_request(url);\n";
        }

        // destroy the BF datablocks
        for ($i=0; $i<count($this->bf_dbs); $i++)
        {
          $dbid = $this->bf_dbs[$i];
          echo "         url = \"control.lib.php?area=bf&action=stop&daemon=buffer_".$dbid."&nhosts=\"+bf_hosts.length+bf_hosts_str;\n";
          echo "         daemon_action_request(url);\n";
        }

        // destroy the BP datablocks
        for ($i=0; $i<count($this->bp_dbs); $i++)
        {
          $dbid = $this->bp_dbs[$i];
          echo "         url = \"control.lib.php?area=bp&action=stop&daemon=buffer_".$dbid."&nhosts=\"+bp_hosts.length+bp_hosts_str;\n";
          echo "         daemon_action_request(url);\n";
        }

        // destroy the BS datablocks
        for ($i=0; $i<count($this->bs_dbs); $i++)
        {
          $dbid = $this->bs_dbs[$i];
          echo "         url = \"control.lib.php?area=bs&action=stop&daemon=buffer_".$dbid."&nhosts=\"+bs_hosts.length+bs_hosts_str;\n";
          echo "         daemon_action_request(url);\n";
        }
?>

        // stop the server daemons 
        url = "control.lib.php?area=srv&action=stop&daemon=all&nhosts=1&host_0="+srv_host;
        daemon_action_request(url);

        stage3_wait = 20;
        stopMopsrStage3();

      }
    
      function stopMopsrStage3()
      {
        poll_2sec_count = 0;

        var srv_ready = checkMachinesAndDaemons("srv", srv_hosts, srv_daemons_custom, "red_light.png");

        var aq_daemons = new Array(<?echo $this->aq_dbs_str?>); 
        var aq_lights = new Array("red_light.png", "grey_light.png");
        var aq_ready = checkMachinesPWCsAndDaemons("aq", aq_hosts, aq_daemons, aq_lights);

        var bf_daemons;
        var bf_ready = true;
        if (bf_hosts.length > 0)
        {
          bf_daemons = new Array(<?echo $this->bf_dbs_str?>); 
          bf_ready = checkMachinesPWCsAndDaemons("bf", bf_hosts, bf_daemons, "red_light.png");
        }

        var bp_daemons;
        var bp_ready = true;
        if (bp_hosts.length > 0)
        {
          bp_daemons = new Array(<?echo $this->bp_dbs_str?>); 
          bp_ready = checkMachinesPWCsAndDaemons("bp", bp_hosts, bp_daemons, "red_light.png");
        }

        var bs_daemons;
        var bs_ready = true;
        if (bs_hosts.length > 0)
        {
          bs_daemons = new Array(<?echo $this->bs_dbs_str?>); 
          bs_ready = checkMachinesPWCsAndDaemons("bs", bs_hosts, bs_daemons, "red_light.png");
        }

        if ((!(srv_ready && aq_ready && bf_ready && bp_ready && bs_ready)) && (stage3_wait > 0)) {
          stage3_wait--;
          setTimeout('stopMopsrStage3()', 1000);
          return 0;
        }
        stage3_wait = 0;

        // stop the AQ's master control script
        url = "control.lib.php?area=aq&action=stop&daemon=mopsr_master_control&nhosts="+aq_hosts.length+aq_hosts_str;
        daemon_action_request(url);

        // stop the BF's master control script
        if (bf_hosts.length > 0)
        {
          url = "control.lib.php?area=bf&action=stop&daemon=mopsr_bf_master_control&nhosts="+bf_hosts.length+bf_hosts_str;
          daemon_action_request(url);
        }

        // stop the BP's master control script
        if (bp_hosts.length > 0)
        {
          url = "control.lib.php?area=bp&action=stop&daemon=mopsr_bp_master_control&nhosts="+bp_hosts.length+bp_hosts_str;
          daemon_action_request(url);
        }

        // stop the BS's master control script
        if (bs_hosts.length > 0)
        {
          url = "control.lib.php?area=bs&action=stop&daemon=mopsr_bs_master_control&nhosts="+bs_hosts.length+bs_hosts_str;
          daemon_action_request(url);
        }

        // stop the server's master control script
        url = "control.lib.php?area=srv&action=stop&daemon=mopsr_master_control&nhosts=1&host_0="+srv_host
        daemon_action_request(url);

        stage4_wait = 20;
        stopMopsrStage4();
      }

      function stopMopsrStage4()
      {
        poll_2sec_count = 0;
        var srv_daemons = new Array("mopsr_master_control");
        var srv_ready = checkMachinesAndDaemons("srv", srv_hosts, srv_daemons, "red_light.png");

        var aq_daemons = new Array("mopsr_master_control");
        var aq_ready = checkMachinesAndDaemons("aq", aq_hosts, aq_daemons, "red_light.png");

        var bf_daemons;
        var bf_ready = true;
        if (bf_hosts.length > 0)
        {
          var bf_daemons = new Array("mopsr_bf_master_control");
          var bf_ready = checkMachinesAndDaemons("bf", bf_hosts, bf_daemons, "red_light.png");
        }

        var bp_daemons;
        var bp_ready = true;
        if (bp_hosts.length > 0)
        {
          bp_daemons = new Array("mopsr_bp_master_control");
          bp_ready = checkMachinesAndDaemons("bp", bp_hosts, bp_daemons, "red_light.png");
        }

        var bs_daemons;
        var bs_ready = true;
        if (bs_hosts.length > 0)
        {
          bs_daemons = new Array("mopsr_bs_master_control");
          bs_ready = checkMachinesAndDaemons("bs", bs_hosts, bs_daemons, "red_light.png");
        }

        if ((!(srv_ready && aq_ready && bf_ready && bp_ready && bs_ready)) && (stage5_wait > 0)) {
          stage4_wait--;
          setTimeout('stopMopsrStage4()', 1000);
          return 0;
        }
        stage4_wait = 0;

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

      function setDaemon(area, host, pwc, key, value)
      {
        var img_id;

        if (pwc >= 0)
          img_id = "img_"+area+"_"+host+":"+pwc+"_"+key;
        else
          img_id = "img_"+area+"_"+host+"_"+key;

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
            if (key == "mopsr_master_control") {
              set_daemons_to_grey(host);
            }
            img.src = "/images/red_light.png";
          } else {
            img.src = "/images/grey_light.png";
          }
        }
      }

      function process_daemon_info_area (results, area)
      {
        // for each result returned in the XML DOC
        for (i=0; i<results.length; i++) 
        {
          result = results[i];

          this_result = new Array();
          pids = new Array();

          parseXMLValues(result, this_result, pids);

          host = this_result["host"];

          for (key in this_result) 
          {
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
                setDaemon(area, host, pwc_id, key, value)
              }
            } 
            else 
            {
              value = this_result[key];
              setDaemon(area, host, -1, key, value);
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

      function handle_daemon_info_request(xml_request) 
      {
        if (xml_request.readyState == 4) {

          var xmlDoc=xml_request.responseXML;
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement; 

            var i, j, k, result, key, value, span, this_result, pids;

            var srv_results = xmlObj.getElementsByTagName("srv_daemon_info");
            process_daemon_info_area(srv_results, "srv");

            var aq_results = xmlObj.getElementsByTagName("aq_daemon_info");
            process_daemon_info_area(aq_results, "aq");

            var bf_results = xmlObj.getElementsByTagName("bf_daemon_info");
            process_daemon_info_area(bf_results, "bf");

            var bp_results = xmlObj.getElementsByTagName("bp_daemon_info");
            process_daemon_info_area(bp_results, "bp");

            var bs_results = xmlObj.getElementsByTagName("bs_daemon_info");
            process_daemon_info_area(bs_results, "bs");
          }
        }
      }

      function daemon_info_request() 
      {
        var di_http_request;
        var url = "control.lib.php?update=true";
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

      function popUp(URL) 
      {
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

      function pktReset()
      {
        // poll every 2 seconds during a stop
        poll_update = 1000;
        clearTimeout(poll_timeout);
        poll_2sec_count = 0;
        poll_timeout = setTimeout('poll_server()', poll_update);

        // stop server TCS interface
        url = "control.lib.php?script=mopsr_pkt_reset.pl";
        popUp(url);
      }

      function hardReset()
      {
        // poll every 2 seconds during a stop
        poll_update = 1000;
        clearTimeout(poll_timeout);
        poll_2sec_count = 0;
        poll_timeout = setTimeout('poll_server()', poll_update);

        // stop server TCS interface
        url = "control.lib.php?script=mopsr_hard_reset.csh";
        popUp(url);
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
    <input type='button' value='Start' onClick="startMopsr()">
    <input type='button' value='Stop' onClick="stopMopsr()"><br/>

    <input type='button' value='Pkt Reset' onClick="pktReset()"><br/>
    <input type='button' value='Hard Stop' onClick="hardReset()"><br/>
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
      $pids = array("P630", "P786", "P682");
?>
      <table width='100%'>
        <tr>
          <td>
            <table class='control' id="persist_controls">
<?
      for ($i=0; $i < count($server_daemons_persist); $i++) 
      {
        $d = $server_daemons_persist[$i];
        if ($d == "mopsr_raid_pipeline")
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

    $this->openBlockHeader("JS Console");
    
    echo "<div id='jsconsole'></div>";

    $this->closeBlockHeader();

?>
        </td>
      </tr>
      <tr>
        <td>
<?
    $this->openBlockHeader("Usage");
?>
    <p>Instrument Controls can be used to start/stop/ the all required MOPSR daemons</p>
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
  <title>MOPSR Controls</title>
  <link rel='shortcut icon' href='/mopsr/images/mopsr_favicon.ico'/>
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
    <table width='100%' cellspacing='5px' border=0>
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
          <table width='100%' border=0>  
            <tr>
              <td>
                <table class='control' id="srv_controls" border=0>
<?
    $this->printServerDaemonControl("mopsr_master_control", "Master Control", $host, "srv");
    for ($i=0; $i < count($server_daemons); $i++) 
    {
      $d = $server_daemons[$i];
      $this->printServerDaemonControl($d, $server_daemons_hash[$d]["name"], $host, "srv");
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
    $this->openBlockHeader("Acquisition Daemons");

    $acquisition_daemons = explode(" ",$config["CLIENT_DAEMONS"]);
    $acquisition_daemons_hash =  $this->inst->clientLogInfo();
?>

    <table width='100%' border=0>
      <tr>
        <td>

          <table class='control' id='aq_controls'>
            <tr>
              <td>Host</td>
<?
    for ($i=0; $i<count($this->aq_host_list); $i++) 
    {
      $host = str_replace("mpsr-", "", $this->aq_host_list[$i]["host"]);
      $span = $this->aq_host_list[$i]["span"];
      echo "          <td colspan='".$span."'>".$host."</td>\n";
    }
?>
            </tr>
            <tr>
              <td>PWC</td>
<?
    for ($i=0; $i<count($this->aq_list); $i++) 
    {
      $host = $this->aq_list[$i]["host"];
      $pwc  = $this->aq_list[$i]["pwc"];
      echo "          <td style='text-align: center'><span title='".$host."_".$pwc."'>".$pwc."</span></td>\n";
    }
?>
              <td></td>
            </tr>
<?

    $this->printClientDaemonControl("mopsr_master_control", "Master&nbsp;Control", $this->aq_host_list, "daemon&name=".$d, "aq");

    # Data Blocks
    for ($i=0; $i<count($this->aq_dbs); $i++)
    {
      $id = $this->aq_dbs[$i];
      $this->printClientDBControl("DB&nbsp;".$id, $this->aq_list, $id, "aq");
    }

    # Print the client daemons
    for ($i=0; $i<count($acquisition_daemons); $i++) 
    {
      $d = $acquisition_daemons[$i];
      $n = str_replace(" ", "&nbsp;", $acquisition_daemons_hash[$d]["name"]);
      $this->printClientDaemonControl($d, $n, $this->aq_list, "daemon&name=".$d, "aq");
    }
?>
          </table>
        </td>
        <td width='300px' id='aq_output' valign='top'></td>
      </tr>
    </table>

    </td>
    </tr>

<?
    ###########################################################################
    #
    # Beam Former Section
    #
    if (count($this->bf_host_list) > 0)
    {
?>

    <tr>
    <td style='vertical-align: top'>
<?
    $this->openBlockHeader("Beam Former Daemons");

    $beam_former_daemons = explode(" ",$this->bf_cfg["CLIENT_DAEMONS"]);
    $beam_former_daemons_hash =  $this->inst->clientLogInfo();

?>

    <table width='100%' border=0>
      <tr>
        <td>
          <table class='control' id='bf_controls'>
            <tr>
              <td>Host</td>
<?
    for ($i=0; $i<count($this->bf_host_list); $i++)
    {
      $host = str_replace("mpsr-", "", $this->bf_host_list[$i]["host"]);
      $span = $this->bf_host_list[$i]["span"];
      echo "          <td colspan='".$span."' style='text-align: center;'>".$host."</td>\n";
    }
?>
            </tr>

            <tr>
              <td>Chan</td>
<?
    for ($i=0; $i<count($this->bf_list); $i++)
    {
      $host = $this->bf_list[$i]["host"];
      $bf   = $this->bf_list[$i]["bf"];
      echo "          <td style='text-align: center'><span title='".$host."_".$bf."'>".$bf."</span></td>\n";
    }
?>
              <td></td>
            </tr>

<?
    $this->printClientDaemonControl("mopsr_bf_master_control", "Master&nbsp;Control", $this->bf_host_list, "daemon&name=".$d, "bf");

    # Data Blocks
    for ($i=0; $i<count($this->bf_dbs); $i++)
    {
      $id = $this->bf_dbs[$i];
      $this->printClientDBControl("DB&nbsp;".$id, $this->bf_list, $id, "bf");
    }


    # Print the client daemons
    for ($i=0; $i<count($beam_former_daemons); $i++)
    {
      $d = $beam_former_daemons[$i];
      $n = str_replace(" ", "&nbsp;", $beam_former_daemons_hash[$d]["name"]);
      $this->printClientDaemonControl($d, $n, $this->bf_list, "daemon&name=".$d, "bf");
    }
?>
          </table>
        </td>
        <td width='300px' id='bf_output' valign='top'></td>
      </tr>

    </table>
<?
    $this->closeBlockHeader();
?>
      </td>
    </tr>
<?
    }

    ###########################################################################
    #
    # Beam Processor Section
    #
    if (count($this->bp_host_list) > 0)
    {
?>
    <tr>
    <td style='vertical-align: top'>
<?

    $this->openBlockHeader("Beam Processor Daemons");

    $beam_processor_daemons = explode(" ",$this->bp_cfg["CLIENT_DAEMONS"]);
    $beam_processor_daemons_hash =  $this->inst->clientLogInfo();
?>

    <table width='100%' border=0>
      <tr>
        <td>
          <table class='control' id='bp_controls'>
            <tr>
              <td>Host</td>
<?
    for ($i=0; $i<count($this->bp_host_list); $i++)
    {
      $host = str_replace("mpsr-", "", $this->bp_host_list[$i]["host"]);
      $span = $this->bp_host_list[$i]["span"];
      echo "          <td colspan='".$span."' style='text-align: center;'>".$host."</td>\n";
    }
?>
            </tr>

            <tr>
              <td>Chan</td>
<?
    for ($i=0; $i<count($this->bp_list); $i++)
    {
      $host = $this->bp_list[$i]["host"];
      $bp   = $this->bp_list[$i]["bp"];
      echo "          <td style='text-align: center'><span title='".$host."_".$bp."'>".$bp."</span></td>\n";
    }
?>
              <td></td>
            </tr>

<?
    $this->printClientDaemonControl("mopsr_bp_master_control", "Master&nbsp;Control", $this->bp_host_list, "daemon&name=".$d, "bp");

    # Data Blocks
    for ($i=0; $i<count($this->bp_dbs); $i++)
    {
      $id = $this->bp_dbs[$i];
      $this->printClientDBControl("DB&nbsp;".$id, $this->bp_list, $id, "bp");
    }


    # Print the client daemons
    for ($i=0; $i<count($beam_processor_daemons); $i++)
    {
      $d = $beam_processor_daemons[$i];
      $n = str_replace(" ", "&nbsp;", $beam_processor_daemons_hash[$d]["name"]);
      $this->printClientDaemonControl($d, $n, $this->bp_list, "daemon&name=".$d, "bp");
    }
?>
          </table>
        </td>
        <td width='300px' id='bp_output' valign='top'></td>
      </tr>

    </table>
<?
    $this->closeBlockHeader();
?>
      </td>
    </tr>
<?
    } # END BEAM PROCESSOR LIST

    ###########################################################################
    #
    # Beam Smirfer Section
    #
    if (count($this->bs_host_list) > 0)
    {
?>  
    <tr>
    <td style='vertical-align: top'>
<?

    $this->openBlockHeader("Beam Smirfer Daemons");

    $beam_processor_daemons = explode(" ",$this->bs_cfg["CLIENT_DAEMONS"]);
    $beam_processor_daemons_hash =  $this->inst->clientLogInfo();
?>

    <table width='100%' border=0>
      <tr>
        <td>
          <table class='control' id='bs_controls'>
            <tr>
              <td>Host</td>
<?  
    for ($i=0; $i<count($this->bs_host_list); $i++)
    {
      $host = str_replace("mpsr-", "", $this->bs_host_list[$i]["host"]);
      $span = $this->bs_host_list[$i]["span"];
      echo "          <td colspan='".$span."' style='text-align: center;'>".$host."</td>\n";
    }
?>
            </tr>

            <tr>
              <td>Chan</td>
<?  
    for ($i=0; $i<count($this->bs_list); $i++)
    {
      $host = $this->bs_list[$i]["host"];
      $bs   = $this->bs_list[$i]["bs"];
      echo "          <td style='text-align: center'><span title='".$host."_".$bs."'>".$bs."</span></td>\n";
    }
?>
              <td></td>
            </tr>

<?  
    $this->printClientDaemonControl("mopsr_bs_master_control", "Master&nbsp;Control", $this->bs_host_list, "daemon&name=".$d, "bs");

    # Data Blocks
    for ($i=0; $i<count($this->bs_dbs); $i++)
    {
      $id = $this->bs_dbs[$i];
      $this->printClientDBControl("DB&nbsp;".$id, $this->bs_list, $id, "bs");
    }


    # Print the client daemons
    for ($i=0; $i<count($beam_processor_daemons); $i++)
    {
      $d = $beam_processor_daemons[$i];
      $n = str_replace(" ", "&nbsp;", $beam_processor_daemons_hash[$d]["name"]); 
      $this->printClientDaemonControl($d, $n, $this->bs_list, "daemon&name=".$d, "bs");
    }
?>
          </table>
        </td>
        <td width='300px' id='bs_output' valign='top'></td>
      </tr>

    </table>
<?
    $this->closeBlockHeader();
?>
      </td>
    </tr>
<?
    } # END BEAM SMIRFER LIST

?>
    </table>
<?
  }

  function getNodeStatuses($conns)
  {
    $cmd = "daemon_info_xml";
    $sockets = array();
    $results = array();
    $responses = array();

    # open the socket connections
    for ($i=0; $i<count($conns); $i++)
    {
      list ($host, $port) = split(":", $conns[$i]);
      #echo "getNodeStatus: openSocket(".$host.":".$port.")<BR>\n"; flush();
      list ($sockets[$i], $results[$i]) = openSocket($host, $port);
      #echo "getNodeStatus: openSocket: ".$sockets[$i]." ".$results[$i]."<BR>\n"; flush();
    }

    # write the commands
    for ($i=0; $i<count($conns); $i++)
    {
      if ($results[$i] == "ok") 
      {
        #echo "getNodeStatus: [$i] <- $cmd<BR>\n"; flush();
        socketWrite($sockets[$i], $cmd."\r\n");
      } 
      else
      {
        $results[$i] = "fail";
        $responses[$i] = "";
      }
    }

    # read the responses
    for ($i=0; $i<count($conns); $i++)
    {
      if (($results[$i] == "ok") && ($sockets[$i])) 
      {
        #echo "getNodeStatus: [$i] socketRead<BR>\n"; flush();
        list ($result, $response) = socketRead($sockets[$i]);
        #echo "[$i] -> $responses [$result]<BR>\n";
        if ($result == "ok")
          $responses[$i] = $response;
        else
          $responses[$i] = "";
      }
    }

    # close the sockets
    for ($i=0; $i<count($conns); $i++)
    {
      if ($results[$i] == "ok") 
      {
        #echo "[$i] close $host:$port<BR>\n";
        if ($sockets[$i])
        {
          @socket_close($sockets[$i]);
        }
      }
    }

    return $responses;
  }

  #############################################################################
  #
  # print update information for the control page as XML
  #
  function printUpdateXML($get)
  {
    $port = $this->inst->config["CLIENT_MASTER_PORT"];

    # do the server + aq nodes first
    $srv_conns = array();
    array_push ($srv_conns, $this->inst->config["SERVER_HOST"].":".$this->inst->config["CLIENT_MASTER_PORT"]);
    #echo "SRV getNodeStatuses<BR>\n"; flush();
    $srv_statuses = $this->getNodeStatuses($srv_conns);

    $aq_conns = array();
    for ($i=0; $i<count($this->aq_host_list); $i++)
      array_push ($aq_conns, $this->aq_host_list[$i]["host"].":".$this->inst->config["CLIENT_MASTER_PORT"]);
    #echo "AQ getNodeStatuses<BR>\n"; flush();
    $aq_statuses = $this->getNodeStatuses($aq_conns);

    $bf_conns = array();
    #echo "BF getNodeStatuses<BR>\n"; flush();
    for ($i=0; $i<count($this->bf_host_list); $i++)
      array_push ($bf_conns, $this->bf_host_list[$i]["host"].":".$this->bf_cfg["CLIENT_MASTER_PORT"]);
    $bf_statuses = $this->getNodeStatuses($bf_conns);

    $bp_conns = array();
    #echo "BP getNodeStatuses<BR>\n"; flush();
    for ($i=0; $i<count($this->bp_host_list); $i++)
      array_push ($bp_conns, $this->bp_host_list[$i]["host"].":".$this->bp_cfg["CLIENT_MASTER_PORT"]);
    $bp_statuses = $this->getNodeStatuses($bp_conns);

    $bs_conns = array();
    #echo "BS getNodeStatuses<BR>\n"; flush();
    for ($i=0; $i<count($this->bs_host_list); $i++)
      array_push ($bs_conns, $this->bs_host_list[$i]["host"].":".$this->bs_cfg["CLIENT_MASTER_PORT"]);
    $bs_statuses = $this->getNodeStatuses($bs_conns);

    # produce the xml
    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<daemon_infos>\n";

    for ($i=0; $i<count($srv_statuses); $i++)
    {
      $xml .= "<srv_daemon_info>";
      if ((!array_key_exists($i, $srv_statuses)) || ($srv_statuses[$i] == ""))
      {
        list ($host, $port) = split(":", $srv_conns[$i]);
        $xml .= "<host>".$host."</host><mopsr_master_control>0</mopsr_master_control>";
      }
      else 
        $xml .= $srv_statuses[$i];
      $xml .= "</srv_daemon_info>\n";
    }

    for ($i=0; $i<count($aq_statuses); $i++) 
    {
      $xml .= "<aq_daemon_info>";
      if ((!array_key_exists($i, $aq_statuses)) || ($aq_statuses[$i] == ""))
      {
        list ($host, $port) = split(":", $aq_conns[$i]);
        $xml .= "<host>".$host."</host><mopsr_master_control>0</mopsr_master_control>";
      }
      else 
        $xml .= $aq_statuses[$i];

      $xml .="</aq_daemon_info>\n";
    }

    for ($i=0; $i<count($bf_statuses); $i++)
    {
      $xml .= "<bf_daemon_info>";
      if ((!array_key_exists($i, $bf_statuses)) || ($bf_statuses[$i] == ""))
      {
        list ($host, $port) = split(":", $bf_conns[$i]);
        $xml .= "<host>".$host."</host><mopsr_bf_master_control>0</mopsr_bf_master_control>";
      }
      else 
        $xml .= $bf_statuses[$i];

      $xml .="</bf_daemon_info>\n";
    }

    for ($i=0; $i<count($bp_statuses); $i++)
    {
      $xml .= "<bp_daemon_info>";
      if ((!array_key_exists($i, $bp_statuses)) || ($bp_statuses[$i] == ""))
      {
        list ($host, $port) = split(":", $bp_conns[$i]);
        $xml .= "<host>".$host."</host><mopsr_bp_master_control>0</mopsr_bp_master_control>";
      }
      else
        $xml .= $bp_statuses[$i];

      $xml .="</bp_daemon_info>\n";
    }

    for ($i=0; $i<count($bs_statuses); $i++)
    {
      $xml .= "<bs_daemon_info>";
      if ((!array_key_exists($i, $bs_statuses)) || ($bs_statuses[$i] == ""))
      {
        list ($host, $port) = split(":", $bs_conns[$i]);
        $xml .= "<host>".$host."</host><mopsr_bs_master_control>0</mopsr_bs_master_control>";
      }
      else
        $xml .= $bs_statuses[$i];
    
      $xml .="</bs_daemon_info>\n";
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
 
    # which class of daemons we are affecting 
    $area = $get["area"];

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

    if (($nhosts == "") || ($action == "") || ($daemon == "") || ($hosts[0] == "") || ($area == "")) {
      echo "ERROR: malformed GET parameters\n";
      exit(0);
    }

    echo $area."\n";
    echo $unique_id."\n";
    flush();

    # special case for starting/stopping persistent server daemons
    if ($area == "persist")
    {
      if ($action == "start")
      {
        echo "Starting ".$daemon." on ".$this->server_host."\n";
        flush();

        $cmd = "ssh -x -l dada ".$this->server_host." 'server_".$daemon.".pl";
        if ($args != "")
          $cmd .= " ".$args;
        $cmd .= "'";
        $output = array();
        $lastline = exec($cmd, $output, $rval);
      }
      else if ($action == "stop")
      {
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
      }
      else
      {
        $html = "Unrecognized action [".$action."] for daemon [".$daemon."]\n";
        flush();
      }

    }
    else if ((strpos($daemon, "master_control") !== FALSE) && ($action == "start"))
    {
      $html = "Starting master control on";
      if ($nhosts > 2)
      {
        $html .= " ".$hosts[0]." - ".$hosts[$nhosts-1];
      }
      else
      {
        for ($i=0; $i<$nhosts; $i++)
        {
          $html .= " ".$hosts[$i];
        }
      }
      echo $html."\n";
      flush();
      for ($i=0; $i<$nhosts; $i++)
      {
        if ($area == "srv")
          $cmd = "ssh -x -l dada ".$hosts[$i]." 'client_".$daemon.".pl'";
        else
          $cmd = "ssh -x -l mpsr ".$hosts[$i]." 'client_".$daemon.".pl'";
        $output = array();
        $lastline = exec($cmd, $output, $rval);
      }
    }
    else
    {
      $sockets = array();
      $results = array();
      $responses = array();

      if ($area == "bf")
        $port = $this->bf_cfg["CLIENT_MASTER_PORT"];
      else if ($area == "bp")
        $port = $this->bp_cfg["CLIENT_MASTER_PORT"];
      else if ($area == "bs")
        $port = $this->bs_cfg["CLIENT_MASTER_PORT"];
      else
        $port = $this->inst->config["CLIENT_MASTER_PORT"];

      $html = "";

      # all daemons started or stopped
      if ($daemon == "all")
      {
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
      for ($i=0; $i<count($results); $i++) {
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
  function statusLight($area, $host, $daemon, $value, $args, $jsfunc="toggleDaemon") 
  {
    $id = $area."_".$host."_".$daemon;
    $img_id = "img_".$id;
    $link_id = "link_".$id;
    $colour = "grey";
    if ($value == 0) $colour = "red";
    if ($value == 1) $colour = "yellow";
    if ($value == 2) $colour = "green";

    $img = "<img border='0' id='".$img_id."' src='/images/".$colour."_light.png' width='15px' height='15px'>";
    $link = "<a href='javascript:".$jsfunc."(\"".$area."\",\"".$host."\",\"".$daemon."\",\"".$args."\")'>".$img."</a>";

    return $link;

  }

  function printServerDaemonControl($daemon, $name, $host, $area) 
  {
    echo "  <tr>\n";
    echo "    <td style='vertical-align: middle'>".$name."</td>\n";
    echo "    <td style='vertical-align: middle'>".$this->statusLight($area, $host, $daemon, "-1", "")."</td>\n";
    echo "  </tr>\n";
  }

  function printPersistentServerDaemonControl($daemon, $name, $host, $pids) 
  {
    echo "  <tr>\n";
    echo "    <td>".$name."</td>\n";
    if (is_array($pids))
      echo "    <td>".$this->statusLight("persist", $host, $daemon, "-1", "", "toggleDaemonPID")."</td>\n";
    else
      echo "    <td>".$this->statusLight("persist", $host, $daemon, "-1", "", "toggleDaemonPersist")."</td>\n";
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

  function printClientDaemonControl($daemon, $name, $hosts, $cmd, $area) 
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
      if (array_key_exists("bf", $hosts[$i])) 
        $pwc = ":".$hosts[$i]["bf"];
      if (array_key_exists("bp", $hosts[$i])) 
        $pwc = ":".$hosts[$i]["bp"];
      if (array_key_exists("bs", $hosts[$i])) 
        $pwc = ":".$hosts[$i]["bs"];
      
      echo "    <td colspan='".$span."' style='text-align: center;'>".$this->statusLight($area, $host.$pwc, $daemon, -1, "")."</td>\n";
      if (strpos($host_str, $host) === FALSE)
        $host_str .= $host." ";
    }
    $host_str = rtrim($host_str);
    if ($cmd != "" ) {
      echo "    <td style='text-align: center;'>\n";
      echo "      <input type='button' value='Start' onClick=\"toggleDaemons('start', '".$daemon."', '".$host_str."','".$area."')\">\n";
      echo "    </td>\n";
      echo "    <td>\n";
      echo "      <input type='button' value='Stop' onClick=\"toggleDaemons('stop', '".$daemon."', '".$host_str."','".$area."')\">\n";
      echo "    </td>\n";
    }
    echo "  </tr>\n";
  }

  #
  # Print the data block row
  #
  function printClientDBControl($name, $hosts, $id, $area) 
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
      if (array_key_exists("bf", $hosts[$i])) 
        $pwc = ":".$hosts[$i]["bf"];
      if (array_key_exists("bp", $hosts[$i])) 
        $pwc = ":".$hosts[$i]["bp"];
      if (array_key_exists("bs", $hosts[$i])) 
        $pwc = ":".$hosts[$i]["bs"];
      echo "    <td span='".$span."' style='text-align: center;'>".$this->statusLight($area, $host.$pwc, $daemon, -1, "")."</td>\n";
      if (strpos($host_str, $host) === FALSE)
        $host_str .= $host." ";
    }
    $host_str = rtrim($host_str);
    echo "    <td style='text-align: center;'>\n";
    echo "      <input type='button' value='Init' onClick=\"toggleDaemons('start', '".$daemon_on."', '".$host_str."','".$area."')\">\n";
    echo "    </td>\n";
    echo "    <td>\n";
    echo "      <input type='button' value='Dest' onClick=\"toggleDaemons('stop', '".$daemon_off."', '".$host_str."','".$area."')\">\n";
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
