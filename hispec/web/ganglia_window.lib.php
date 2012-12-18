<?PHP

include_once("hispec.lib.php");
include_once("hispec_webpage.lib.php");

class ganglia_window extends hispec_webpage 
{

  var $nodes_network;
  var $nodes_load;
  var $srvs_network;
  var $srvs_load;

  function ganglia_window()
  {
    hispec_webpage::hispec_webpage();

    $inst = new hispec();

    $gang_base =  "http://".$_SERVER["HTTP_HOST"]."/ganglia/";
    $this->nodes_network   = $gang_base."graph.php?g=network_report&z=dada_web&c=Nodes&m=&r=hour&s=descending&hc=4";
    $this->nodes_load      = $gang_base."graph.php?g=load_report&z=dada_web&c=Nodes&m=&r=hour&s=descending&hc=4";
    $this->nodes_gpu_load  = $gang_base."graph.php?g=gpu_load_report&z=dada_web&c=Nodes&m=&r=hour&s=descending&hc=4";
    $this->srvs_network = $gang_base."graph.php?g=network_report&z=dada_web&c=Servers&h=hipsr-srv0.hipsr.local&m=&r=hour&s=descending&hc=4";
    $this->srvs_load    = $gang_base."graph.php?g=load_report&z=dada_web&c=Servers&h=hipsr-srv0.hipsr.local&m=&r=hour&s=descending&hc=4";
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
        document.getElementById("nodes_load").src = "<?echo $this->nodes_load?>?"+theTime;
        document.getElementById("nodes_gpu_load").src = "<?echo $this->nodes_gpu_load?>?"+theTime;
        document.getElementById("nodes_network").src = "<?echo $this->nodes_network?>?"+theTime;
        //document.getElementById("parkes_webcam").src = "http://outreach.atnf.csiro.au/visiting/parkes/webcam/parkes.med.jpg?"+theTime;
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
      <td><img id="srvs_network" src="<?echo $this->srvs_network?>"></td>
      <!--
      <td rowspan=2>
        <img id="parkes_webcam" src="http://outreach.atnf.csiro.au/visiting/parkes/webcam/parkes.med.jpg" border=none width='304px' height='228px'>
      </td>
      -->
    </tr>
    <tr>
      <td><img id="nodes_load" src="<?echo $this->nodes_load?>"></td>
      <td><img id="nodes_network" src="<?echo $this->nodes_network?>"></td>
    </tr>
    <tr>
      <td><img id="nodes_gpu_load" src="<?echo $this->nodes_gpu_load?>"></td>
      <td></td>
    </tr>
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

