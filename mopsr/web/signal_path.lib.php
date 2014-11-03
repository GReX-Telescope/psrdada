<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class signal_path extends mopsr_webpage 
{
  var $inst = 0;

  var $img_size = "100x75";

  var $num_rx_east = 44;

  var $num_rx_west = 44;

  var $modules_per_rx = 4;

  var $rx_per_pfb = 4;

  var $pfbs = array("-", "EG01", "EG02", "EG03", "EG04", "EG05", "EG06", "EG07", "EG08", "EG09", "EG10", "EG11", "EG12",
                    "WG01", "WG02", "WG03", "WG04", "WG05", "WG06", "WG07", "WG08", "WG09", "WG10", "WG11", "WG12");

  var $modules = array("-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15");

  var $mod_map = array ("B", "G", "Y", "R");

  var $update_secs = 10;

  var $edit = false;

  function signal_path()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "Signal Path Configuration";
    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();
    $this->signal_path_config = $this->inst->config["CONFIG_DIR"]."/mopsr_signal_paths.txt";
    $this->edit = ((isset($_GET["edit"])) && ($_GET["edit"] == "true"));
  }

  /*
  function javaScriptCallback()
  {
    return "signal_path_request();";
  }
  */

  function printJavaScriptHead()
  {

?>
    <script type='text/javascript'>  

      function handle_signal_path_request(xml_request) 
      {

        if (xml_request.readyState == 4)
        {
          var xmlDoc = xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;

            var http_server = xmlObj.getElementsByTagName("http_server")[0].childNodes[0].nodeValue;
            var url_prefix  = xmlObj.getElementsByTagName("url_prefix")[0].childNodes[0].nodeValue;
            var img_prefix  = xmlObj.getElementsByTagName("img_prefix")[0].childNodes[0].nodeValue;
            var signal_path_error = xmlObj.getElementsByTagName("error")[0];

            try {
              document.getElementById("signal_path_error").innerHTML = "[" + signal_path_error.childNodes[0].nodeValue + "]";
            } catch (e) {

            }

            var sp_boards = xmlObj.getElementsByTagName ("sp_board");

            var i, j;
            var excluded_imgs = Array();
            var excluded_tds  = Array();

            // parse XML for each RX ID
            for (i=0; i<sp_boards.length; i++)
            {
              var sp_board = sp_boards[i];
              var sp_board_id = sp_board.getAttribute("id");


              // boards will have params and plots
              var nodes = sp_board.childNodes;
              for (j=0; j<nodes.length; j++)
              {
                if (nodes[j].nodeName == "params")
                {
                  params = nodes[j].childNodes;
                  for (k=0; k<params.length; k++)
                  {
                    param = params[k];
                    if ((param.nodeType == 1) && (param.getAttribute("key") == "board_health"))
                    {
                      var td_id = document.getElementById(sp_board_id);
                      excluded_tds.push(sp_board_id);
                      if (param.getAttribute("state") == "OK")
                        td_id.style.backgroundColor = "#22FF22";
                      if (param.getAttribute("state") == "WARNING")
                        td_id.style.backgroundColor = "#FFFF00";
                      if (param.getAttribute("state") == "ERROR")
                        td_id.style.backgroundColor = "#FF0000";
                    }
                  } 
                }
                else if (nodes[j].nodeName == "plots")
                {
                  var modules = nodes[j].childNodes;
                  for (k=0; k<modules.length; k++)
                  {
                    var module = modules[k];
                    var module_id = module.getAttribute("id");

                    // for each image in this module
                    var images = module.childNodes;
                    for (l=0; l<images.length; l++)
                    {
                      img = images[l];
                      if (img.nodeType == 1)
                      {
                        var type = img.getAttribute("type");
                        if (type == active_img)
                        {
                          var img_id = sp_board_id + "_" + module_id;
                          var imgurl = http_server + "/" + url_prefix + "/" + img_prefix + "/" + img.childNodes[0].nodeValue;
                          if (parseInt(img.getAttribute("width")) < 400)
                          {
                            excluded_imgs.push(img_id);
                            document.getElementById (img_id).src = imgurl;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            reset_other_imgs (excluded_imgs);
            reset_other_tds (excluded_tds);
          }
        }
      }
                  
      function signal_path_request() 
      {
        var url = "signal_path.lib.php?update=true";

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_signal_path_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

    </script>

    <style type="text/css">
    
      table.signal_path {
        border-spacing: 4;
      }

      table.signal_path td {
        padding-top: 2px;
        padding-bottom: 2px;
        padding-left: 1px;
        padding-right: 1px;
        text-align: center;
      }

      table.signal_path img {
        margin-left:  2px;
        margin-right: 2px;
      }

      th {
        padding-right: 10px;
        padding-left: 10px;
      }

    </style>

<?
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("Signal Path Configuration");

    // read the master configuration file
    $sp_config = $this->readConfiguration();
  
   if ($this->edit) { ?>
    <form name="signal_path_form" action="signal_path.lib.php?action=true" method="post">
    <p>
      <input type="submit" value="Save Changes"/>
      <input type="button" value="Cancel Changes" onClick="document.location='signal_path.lib.php?single=true'"/>
    </p>
<?  } else { ?>

      <input type="button" value="Edit Config" onClick="document.location='signal_path.lib.php?single=true&edit=true'"/>

<?  } ?>

    <table border=0 class='signal_path'>
      <tr>
        <th class='pad_right'>Bay</th>
        <th>Blue Input</th>
        <th>Green Input</th>
        <th>Yellow Input</th>
        <th>Red Input</th>
      </tr>
<?
    $num_rows = $this->num_rx_east;

    for ($irow=0; $irow < $num_rows; $irow++)
    {
      $bay = sprintf ("East %02d", $irow + 1);
      $rx_id = sprintf ("E%02d", $irow + 1);

      $cfg = array();

      echo "<tr>\n";
      echo "<th><a href='rx_board.lib.php?single=true&rx_id=".$rx_id."'>".$bay."</a></th>\n";
      for ($imod=0; $imod < $this->modules_per_rx; $imod++)
      {
        $mod_id = $rx_id."-".$this->mod_map[$imod];
        if (array_key_exists($mod_id, $sp_config))
        {
          $cfg = $sp_config[$mod_id];
          if ($this->edit)
            echo "<td>".$this->printSelectionList($mod_id, $cfg["pfb"], $cfg["pfb_input"])."</td>\n";
          else
            if (($cfg["pfb"] == "-") && ($cfg["pfb_input"] == "-"))
              echo "<td>--</td>\n";
            else
              echo "<td>".$cfg["pfb"]."-".sprintf("%02d",$cfg["pfb_input"])."</td>\n";
        }
        else
          echo "<td></td>\n";
      }
      echo "</tr>\n";
    }

    echo "<tr><td colspan=5><hr></td></tr>\n";

    $num_rows = $this->num_rx_west;

    for ($irow=0; $irow < $num_rows; $irow++)
    {
      $bay = sprintf ("West %02d", $irow + 1);
      $rx_id = sprintf ("W%02d", $irow + 1);

      $cfg = array();
      if (array_key_exists($rx_id, $sp_config))
        $cfg = $sp_config[$rx_id];

      echo "<tr>\n";
      echo "<th><a href='rx_board.lib.php?single=true&rx_id=.".$rx_id."'>".$bay."</a></th>\n";
      for ($imod=0; $imod < $this->modules_per_rx; $imod++)
      {
        $mod_id = $rx_id."-".$this->mod_map[$imod];
        if (array_key_exists($mod_id, $sp_config))
        {
          $cfg = $sp_config[$mod_id];
          if ($this->edit)
            echo "<td>".$this->printSelectionList($rx_id."-".$imod, $cfg["pfb"], $cfg["pfb_input"])."</td>\n";
          else
            if (($cfg["pfb"] == "-") && ($cfg["pfb_input"] == "-"))
              echo "<td>--</td>\n";
            else
              echo "<td>".$cfg["pfb"]."-".sprintf("%02d", $cfg["pfb_input"])."</td>\n";
        }
        else
          echo "<td></td>\n";
      }
      echo "</tr>\n";
    }
    echo "</table>\n";

    if ($this->edit) { ?>

    </form>

<?  }
      
    $this->closeBlockHeader();
  }

  function printSelectionList($module, $curr_pfb, $curr_pfb_input)
  {
    $html  = "<select name='".$module."_pfb'>\n";
    foreach ($this->pfbs as $pfb)
      if ($pfb == $curr_pfb)
        $html .= "<option value='".$pfb."' selected>".$pfb."</option>";
      else
        $html .= "<option value='".$pfb."'>".$pfb."</option>";
    $html .= "</select>\n";

    $html .= "<select name='".$module."_pfb_input' curr='".$curr_pfb_input."'>\n";
    foreach ($this->modules as $mod)
    {
      if (strval($mod) == strval($curr_pfb_input))
        $html .= "<option value='".$mod."' selected>".$mod."</option>\n";
      else
        $html .= "<option value='".$mod."'>".$mod."</option>\n";
    }
    $html .= "</select>\n";

    return $html;
  }

  function readConfiguration()
  {
    $config = array();

    // open the configuration file
    $handle = fopen($this->signal_path_config, "r");
    if (!$handle)
      return $config;

    while (!feof($handle))
    {
      $buffer = fgets($handle, 4096);
      list ($mod_id, $pfb_id, $pfb_input) = preg_split('/\s+/', $buffer);
      $config[$mod_id] = array("pfb" => $pfb_id, "pfb_input" => $pfb_input);
    }
    fclose($handle);

    return $config;
  }

  function printUpdateHTML($get)
  {
    echo "printUpdateHTML()<br/>\n";
    print_r($get);
    return;
  }

  function printActionHTML($get)
  {
    $num_rows = $this->num_rx_east;

    $fptr = @fopen($this->signal_path_config, "w");
    if ($fptr)
    {
      for ($irow=0; $irow < $num_rows; $irow++)
      {
        $rx_id = sprintf ("E%02d", $irow + 1);
        for ($imod=0; $imod < $this->modules_per_rx; $imod++)
        {
          $mod_id = $rx_id."-".$imod;
          $pfb = $_POST[$mod_id."_pfb"];
          $pfb_input = $_POST[$mod_id."_pfb_input"];

          fwrite($fptr, $mod_id." ".$pfb." ".$pfb_input."\n");
        }
        if ($irow && (($irow+1) % 4 == 0))
          fwrite ($fptr, "\n");
      }

      $num_rows = $this->num_rx_west;

      for ($irow=0; $irow < $num_rows; $irow++)
      {
        $rx_id = sprintf ("W%02d", $irow + 1);
        for ($imod=0; $imod < $this->modules_per_rx; $imod++)
        {
          $mod_id = $rx_id."-".$imod;
          $pfb = $_POST[$mod_id."_pfb"];
          $pfb_input = $_POST[$mod_id."_pfb_input"];

          fwrite($fptr, $mod_id." ".$pfb." ".$pfb_input."\n");
        }
        if ($irow && (($irow+1) % 4 == 0))
          fwrite ($fptr, "\n");
      }
      fclose ($fptr);
      echo "<script type='text/javascript'>\n";
      echo "document.location='signal_path.lib.php?single=true';\n";
      echo "</script>\n";
    }
    else
    {
      echo  "ERROR: unable to write to ".$this->signal_path_config."<BR>\n";
    }
    return;
  }
}

handleDirect("signal_path");
