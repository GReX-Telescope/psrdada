<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class rxlink_monitor extends mopsr_webpage 
{
  var $inst = 0;

  var $img_size = "100x75";

  var $num_rx_east = 44;

  var $num_rx_west = 44;

  var $update_secs = 10;

  var $ip_to_erx = array();
  var $ip_to_wrx = array();

  var $use_socket_connection = true;

  function rxlink_monitor()
  {
    mopsr_webpage::mopsr_webpage();
    $this->title = "MOPSR RX Link Monitor";

    $this->callback_freq = $this->update_secs * 1000;
    $this->inst = new mopsr();
    
    for ($oct=101; $oct<=144; $oct++)
      $this->ip_to_erx["172.17.228.".$oct] = "E".($oct-100);
    for ($oct=149; $oct<=192; $oct++)
      $this->ip_to_wrx["172.17.228.".$oct] = "W".($oct-148);
  }

  function javaScriptCallback()
  {
    return "rxlink_monitor_request();";
  }

  function printJavaScriptHead()
  {

?>
    <script type='text/javascript'>  

      var active_img = "sp";

      function reset_other_imgs(excluded) 
      {
        var imgs = document.getElementsByTagName('img');
        var i=0;
        for (i=0; i< imgs.length; i++) 
        {
          if (excluded.indexOf(imgs[i].id) == -1)
          {
            imgs[i].src = "/images/red_light.png";
          }
        }
      }

      function handle_rxlink_monitor_request(xml_request) 
      {
        if (xml_request.readyState == 4)
        {
          var xmlDoc = xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj=xmlDoc.documentElement;
            var hosts = xmlObj.getElementsByTagName ("host");
            var excluded_imgs = Array();
            for (i=0; i<hosts.length; i++)
            {
              var nodes = hosts[i].childNodes;
              var state = "unknown";
              var addr = "unknown";

              for (j=0; j<nodes.length; j++)
              {
                if (nodes[j].nodeName == "status")
                {
                  state = nodes[j].getAttribute("state");
                }
                if (nodes[j].nodeName == "address")
                {
                  addr = nodes[j].getAttribute("addr");
                }
              }
              if (state == "up")
              {
                try {
                  document.getElementById (addr).src = "/images/green_light.png";
                  excluded_imgs.push(addr);
                } catch (e) {
                  alert ("problem finding "+addr);
                }
              }
            }
            reset_other_imgs (excluded_imgs);
          }
        }
      }

      function rxlink_monitor_request() 
      {
        var url = "rxlink_monitor.lib.php?update=true";

        if (window.XMLHttpRequest)
          xml_request = new XMLHttpRequest();
        else
          xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        xml_request.onreadystatechange = function() {
          handle_rxlink_monitor_request(xml_request)
        };
        xml_request.open("GET", url, true);
        xml_request.send(null);
      }

    </script>

    <style type="text/css">
    
      table.rxlink_monitor {
        border-spacing: 4;
      }

      table.rxlink_monitor td {
        padding-top: 2px;
        padding-bottom: 2px;
        padding-left: 1px;
        padding-right: 1px;
      }

      table.rxlink_monitor img {
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
    $this->openBlockHeader("Receiver Board Ethernet Link Monitor&nbsp;&nbsp;&nbsp;<span id='rxlink_monitor_error'></span>");
?>
    <table border=0 class='rxlink_monitor'>
      <tr><th class='pad_right'>Bay</th><th colspan=1>East</th><th></th><th colspan=1>West</th></tr>
<?
    $num_rows= max($this->num_rx_east, $this->num_rx_west);

    $keys_e = array_keys($this->ip_to_erx);
    $keys_w = array_keys($this->ip_to_wrx);

    for ($irow=0; $irow < $num_rows; $irow++)
    {
      $bay = sprintf ("%02d", $irow + 1);

      echo "<tr>\n";
      echo "<th>".$bay."</th>";

      $rx_ip = $keys_e[$irow];
      $rx_id = $this->ip_to_erx[$rx_ip];

      echo "<td>\n";
      echo "<img id='".$rx_ip."' src='/images/grey_light.png' width='15px' height='15px' title='".$rx_id."'/>";
      echo "</td>\n";

      echo "<td>&nbsp;</td>\n";

      # west bays
      $rx_ip = $keys_w[$irow];
      $rx_id = $this->ip_to_wrx[$rx_ip];

      echo "<td>\n";
      echo "<img id='".$rx_ip."' src='/images/grey_light.png' width='15px' height='15px' title='".$rx_id."'/>";
      echo "</td>\n";

      echo "</tr>\n";
    }
    echo "</table>\n";
?>
  </center>
<?
    $this->closeBlockHeader();
  }

  function printUpdateHTML($get)
  {
    # generate a temporary name file
    $ofile = tempnam("/tmp", "rxlink");
    $cmd = "nmap -oX  ".$ofile." -n -sn 172.17.228.101-192 > /dev/null";
    system($cmd);

    header('Content-type: text/xml');
    readfile($ofile);
  }
}

handleDirect("rxlink_monitor");
