<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class candidate_viewer extends bpsr_webpage 
{
  var $utc_start;

  var $beam;

  var $sample;

  var $filter;

  var $dm;

  var $inst;

  function candidate_viewer()
  {
    bpsr_webpage::bpsr_webpage();
    $this->title = "BPSR Candidate Viewer";
    $this->callback_freq = 16000;
    $this->inst = new bpsr();

    $this->utc_start = isset($_GET["utc_start"]) ? $_GET["utc_start"] : "";
    $this->beam = isset($_GET["beam"]) ? $_GET["beam"] : "";
    $this->sample = isset($_GET["sample"]) ? $_GET["sample"] : "";
    $this->filter = isset($_GET["filter"]) ? $_GET["filter"] : "";
    $this->dm = isset($_GET["dm"]) ? $_GET["dm"] : "";

    #$mod_ten_time = floor($this->time / 10) * 10;
    #$this->mon_file_time = addToDadaTime($this->utc_start, $mod_ten_time);
  }

/*
  function javaScriptCallback()
  {
    return "candidate_viewer_request();";
  }
*/
  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  
    </script>
<?
  }

  function printSideBarHTML()
  {
    $this->openBlockHeader("Observation Information");

?>
    <table border=1 cellpadding=4px>
      <tr>
        <th>Key</th><th>Value</th>
      </tr>
      <tr>
        <td>UTC_START</td>
        <td><?echo $this->utc_start?></td>
      </tr>
      <tr>
        <td>Beam</td>
        <td><?echo $this->beam?></td>
      </tr>
      <tr>
        <td>Sample</td>
        <td><?echo $this->sample?></td>
      </tr>
      <tr>
        <td>Filter</td>
        <td><?echo $this->filter?></td>
      </tr>
      <tr>
        <td>DM</td>
        <td><?echo $this->dm?></td>
      </tr>
    </table>
<?
    $this->closeBlockHeader();
  }


  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("Candidate Event");

    $cand_url = "candidate_viewer.lib.php?update=true";
    $cand_url .= "&utc_start=".$this->utc_start;
    $cand_url .= "&beam=".$this->beam;
    $cand_url .= "&sample=".$this->sample;
    $cand_url .= "&filter=".$this->filter;
    $cand_url .= "&dm=".$this->dm;
?>
    <img src="<?echo $cand_url?>">
<?
    $this->closeBlockHeader();
  }

  # get the candidate plot from the gpu cand server
  function printUpdateHTML($get)
  {
    $host = "hipsr7";
    $port = "55555";

    $params  = "";
    $params .= "utc_start=".$get["utc_start"]." ";
    $params .= "beam=".$get["beam"]." ";
    $params .= "sample=".$get["sample"]." ";
    $params .= "filter=".$get["filter"]." ";
    $params .= "dm=".$get["dm"];

    list ($socket, $result) = openSocket($host, $port);
    $img_data = "";
    if ($result == "ok")
    {
      $bytes_written = socketWrite($socket, $params."\r\n");
      $data = socket_read($socket, 8192, PHP_BINARY_READ);
      $img_data = $data;
      while ($data)
      {
        $data = socket_read($socket, 8192, PHP_BINARY_READ);
        $img_data .= $data;
      }
      if ($socket)
        socket_close($socket);
      $socket = 0;
      header('Content-Type: image/png');
      header('Content-Disposition: inline; filename="image.png"');
      echo $img_data;
    }
    else
    {
      header('Content-Type: image/gif');
      header('Content-Disposition: inline; filename="image.gif"');
      passthru("cat /home/dada/linux_64/web/images/blankimage.gif");
    }
  }
}

handleDirect("candidate_viewer");
