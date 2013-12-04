<?PHP

include_once("caspsr_webpage.lib.php");
include_once("definitions_i.php");
include_once("functions_i.php");
include_once($instrument.".lib.php");

class ganglia_window extends caspsr_webpage 
{

  var $gpus_ibnetwork;
  var $gpus_network;
  var $gpus_load;
  var $gpus_gpuload;
  var $demuxs_ibnetwork;
  var $demuxs_network;
  var $demuxs_load;

  function ganglia_window()
  {
    caspsr_webpage::caspsr_webpage();

    $inst = new caspsr();

    $gang_base =  "http://".$_SERVER["HTTP_HOST"]."/ganglia/";
    $this->gpus_ibnetwork   = $gang_base."graph.php?g=ibnetwork_report&z=small&c=GPUs&m=&r=hour&s=descending&hc=4";
    $this->gpus_network     = $gang_base."graph.php?g=network_report&z=small&c=GPUs&m=&r=hour&s=descending&hc=4";
    $this->gpus_load        = $gang_base."graph.php?g=load_report&z=small&c=GPUs&m=&r=hour&s=descending&hc=4";
    $this->gpus_gpuload     = $gang_base."graph.php?g=gpuload_report&z=small&c=GPUs&m=&r=hour&s=descending&hc=4";
    $this->demuxs_ibnetwork = $gang_base."graph.php?g=ibnetwork_report&z=small&c=Demuxers&m=&r=hour&s=descending&hc=4";
    $this->demuxs_network   = $gang_base."graph.php?g=network_report&z=small&c=Demuxers&m=&r=hour&s=descending&hc=4";
    $this->demuxs_load      = $gang_base."graph.php?g=load_report&z=small&c=Demuxers&m=&r=hour&s=descending&hc=4";
    $this->host             = $inst->config["SERVER_HOST"];
    $this->port             = $inst->config["SERVER_WEB_MONITOR_PORT"];
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

      function newImage(old_image_id, new_image_url)
      {
        var new_image = new Image();
        new_image.id = old_image_id;
        new_image.src = new_image_url;
        new_image.onload = function() 
        {
          var old_img = document.getElementById(old_image_id);
          var old_img_width = old_img.width;
          var old_img_height = old_img.height;
          new_image.width = old_img_width;
          new_image.height = old_img_height;
          old_img.parentNode.insertBefore(new_image, old_img);
          old_img.parentNode.removeChild(old_img);
        }
      }

      function update_images() {
        var now = new Date();
        var theTime = now.getTime();
        //document.getElementById("demuxs_load").src = "<?echo $this->demuxs_load?>?"+theTime;
        //document.getElementById("gpus_load").src = "<?echo $this->gpus_load?>?"+theTime;

        newImage ("demuxs_network", "<?echo $this->demuxs_network?>?"+theTime);
        newImage ("demuxs_ibnetwork", "<?echo $this->demuxs_ibnetwork?>?"+theTime);
        newImage ("gpus_gpuload", "<?echo $this->gpus_gpuload?>?"+theTime);
        newImage ("gpus_ibnetwork", "<?echo $this->gpus_ibnetwork?>?"+theTime);
        newImage ("parkes_webcam", "ganglia_window.lib.php?update=true&host=<?echo $this->host?>&port=<?echo $this->port?>&"+theTime);


        //document.getElementById("demuxs_network").src = "<?echo $this->demuxs_network?>?"+theTime;
        //document.getElementById("demuxs_ibnetwork").src = "<?echo $this->demuxs_ibnetwork?>?"+theTime;
        //document.getElementById("gpus_gpuload").src = "<?echo $this->gpus_gpuload?>?"+theTime;
        //document.getElementById("gpus_ibnetwork").src = "<?echo $this->gpus_ibnetwork?>?"+theTime;
        //document.getElementById("parkes_webcam").src = "ganglia_window.lib.php?update=true&host=<?echo $this->host?>&port=<?echo $this->port?>&"+theTime;
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
    <table cellpadding=5px>
    <tr>
      <td rowspan=2>
        <img id="parkes_webcam" src="/images/blankimage.gif" border=none width='304px' height='228px'>
      </td>
      <!--<td><img id="demuxs_load" src="<?echo $this->demuxs_load?>"></td>-->
      <td><img id="demuxs_network" src="<?echo $this->demuxs_network?>"></td>
      <td><img id="demuxs_ibnetwork" src="<?echo $this->demuxs_ibnetwork?>"></td>
    </tr>
    <tr>
      <!--<td><img id="gpus_load" src="<?echo $this->gpus_load?>"></td>-->
      <td><img id="gpus_gpuload" src="<?echo $this->gpus_gpuload?>"></td>
      <td><img id="gpus_ibnetwork" src="<?echo $this->gpus_ibnetwork?>"></td>
      <td></td>
    </tr>
    </table>
<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    readfile("http://outreach.atnf.csiro.au/visiting/parkes/webcam/parkes_med.jpg");
    return;

    list ($socket, $result) = openSocket($host, $port);
    if ($result == "ok") {
      $bytes_written = socketWrite($socket, "dish_image\r\n");
      list ($result, $header_one) = socketRead($socket);
      list ($result, $header_two) = socketRead($socket);
      $arr = explode(":",$header_two);
      $arr2 = explode(" ",$arr[1]);
      $size = $arr2[1];
      $read = socket_read($socket, $size, PHP_BINARY_READ);
      header($header_one);
      header($header_two);
      echo $read;
    } else {
      #readfile("http://outreach.atnf.csiro.au/visiting/parkes/webcam/parkes_med.jpg");
    }
  }

  function handleRequest()
  {

    if ($_GET["update"] == "true") {
      $this->printUpdateHTML($_GET);
    } else {
      $this->printHTML($_GET);
    }
  }
}
$obj = new ganglia_window();
$obj->handleRequest($_GET);

