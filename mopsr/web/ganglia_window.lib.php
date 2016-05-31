<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class ganglia_window extends mopsr_webpage 
{
  var $aq_network;
  var $aq_load;
  var $bf_network;
  var $bf_load;
  var $srvs_network;
  var $srvs_load;

  function ganglia_window()
  {
    mopsr_webpage::mopsr_webpage();

    $inst = new mopsr();

    $gang_base =  "http://".$_SERVER["HTTP_HOST"]."/ganglia/";
    $this->srvs_network    = $gang_base."graph.php?g=network_report&z=small&c=Servers&m=&r=hour&s=descending&hc=4";
    $this->srvs_load       = $gang_base."graph.php?g=load_report&z=small&c=Servers&m=&r=hour&s=descending&hc=4";
    $this->aq_network   = $gang_base."graph.php?g=network_report&z=small&c=AQ%20Nodes&m=&r=hour&s=descending&hc=4";
    $this->aq_load      = $gang_base."graph.php?g=load_report&z=small&c=AQ%20Nodes&m=&r=hour&s=descending&hc=4";
    #$this->aq_gpu_load  = $gang_base."graph.php?g=gpu_load_report&z=small&c=Nodes&m=&r=hour&s=descending&hc=4";
    $this->bf_network   = $gang_base."graph.php?g=network_report&z=small&c=BF%20Nodes&m=&r=hour&s=descending&hc=4";
    $this->bf_load      = $gang_base."graph.php?g=load_report&z=small&c=BF%20Nodes&m=&r=hour&s=descending&hc=4";
    #$this->bf_gpu_load  = $gang_base."graph.php?g=gpu_load_report&z=small&c=BF%20Nodes&m=&r=hour&s=descending&hc=4";
    $this->host           = $inst->config["SERVER_HOST"];
    $this->port           = $inst->config["SERVER_WEB_MONITOR_PORT"];
  }

  function javaScriptCallback()
  {
    return "update_images();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      function poll_server()
      {
        update_images();
        setTimeout('poll_server()', 5000);
      } 

      function update_images() {
        var now = new Date();
        var theTime = now.getTime();
        document.getElementById("srvs_load").src = "<?echo $this->srvs_load?>?"+theTime;
        document.getElementById("srvs_network").src = "<?echo $this->srvs_network?>?"+theTime;
        document.getElementById("aq_load").src = "<?echo $this->aq_load?>?"+theTime;
        //document.getElementById("aq_gpu_load").src = "<?echo $this->aq_gpu_load?>?"+theTime;
        document.getElementById("aq_network").src = "<?echo $this->aq_network?>?"+theTime;
        document.getElementById("bf_load").src = "<?echo $this->bf_load?>?"+theTime;
        //document.getElementById("bf_gpu_load").src = "<?echo $this->bf_gpu_load?>?"+theTime;
        document.getElementById("bf_network").src = "<?echo $this->bf_network?>?"+theTime;
      }

    </script>
<?
  }

  function printHTML()
  {
?>
<html>
<head>
<?
    $this->printJavaScriptHead();
?>
</head>
<body onload='poll_server()'>
<?
    $this->printMainHTML();
?>
</body>
</html>
<?


  }

  /* HTML for this page */
  function printMainHTML() 
  {
?>
    <table cellpadding=2px>
    <tr>
      <td><img id="srvs_load" src="<?echo $this->srvs_load?>"></td>
      <td><img id="aq_load" src="<?echo $this->aq_load?>"></td>
      <td><img id="bf_load" src="<?echo $this->bf_load?>"></td>
    </tr>
    <tr>
      <td><img id="srvs_network" src="<?echo $this->srvs_network?>"></td>
      <td><img id="aq_network" src="<?echo $this->aq_network?>"></td>
      <td><img id="bf_network" src="<?echo $this->bf_network?>"></td>
    </tr>
    <!--<tr>
      <td></td>
      <td><img id="aq_gpu_load" src="<?echo $this->aq_gpu_load?>"></td>
      <td><img id="bf_gpu_load" src="<?echo $this->bf_gpu_load?>"></td>
    </tr>-->
    </table>
<?
  }

  function handleRequest()
  {
    $this->printHTML($_GET);
  }
}
$obj = new ganglia_window();
$obj->handleRequest($_GET);

