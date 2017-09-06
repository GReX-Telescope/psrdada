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

  var $snr;

  var $nbin;

  var $nchan;

  var $length;

  var $proc_type;

  var $zap_from;

  var $zap_to;

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
    $this->snr = isset($_GET["snr"]) ? $_GET["snr"] : "";
    $this->nchan = isset($_GET["nchan"]) ? $_GET["nchan"] : "0";
    $this->nbin = isset($_GET["nbin"]) ? $_GET["nbin"] : "0";
    $this->length = isset($_GET["length"]) ? $_GET["length"] : "0";
    $this->proc_type = isset($_GET["proc_type"]) ? $_GET["proc_type"] : "cand";
    $this->zap_from = isset($_GET["zap_from"]) ? $_GET["zap_from"] : "-1";
    $this->zap_to = isset($_GET["zap_to"]) ? $_GET["zap_to"] : "-1";

    $this->cand_url = "candidate_viewer.lib.php?update=true";
    if ($this->utc_start != "") $this->cand_url .= "&utc_start=".$this->utc_start;
    if ($this->beam != "") $this->cand_url .= "&beam=".$this->beam;
    if ($this->sample != "") $this->cand_url .= "&sample=".$this->sample;
    if ($this->filter != "") $this->cand_url .= "&filter=".$this->filter;
    if ($this->dm != "") $this->cand_url .= "&dm=".$this->dm;
    if ($this->snr != "") $this->cand_url .= "&snr=".$this->snr;
    if ($this->proc_type != "") $this->cand_url .= "&proc_type=".$this->proc_type;
    if ($this->zap_from != "") $this->cand_url .= "&zap_from=".$this->zap_from;
    if ($this->zap_to != "") $this->cand_url .= "&zap_to=".$this->zap_to;
    $this->cand_url .= "&length=".$this->length;
    $this->cand_url .= "&nchan=".$this->nchan;
    $this->cand_url .= "&nbin=".$this->nbin;
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  
      function update_candidate_image() 
      {
        url = "<?echo $this->cand_url;?>";

        var i = document.getElementById("nchan").selectedIndex;
        var nchan = document.getElementById("nchan").options[i].value;

        var i = document.getElementById("nbin").selectedIndex;
        var nbin = document.getElementById("nbin").options[i].value;

        var i = document.getElementById("length").selectedIndex;
        var length = document.getElementById("length").options[i].value;

        var zap_from = document.getElementById("zap_from").value;
        var zap_to = document.getElementById("zap_to").value;

        url += "&nbin=" + nbin + "&nchan=" + nchan + "&length=" + length + "&zap_from="+zap_from + "&zap_to=" + zap_to;

        document.getElementById("candidate_image").src = url;

      }
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
      <tr>
        <td>SNR </td>
        <td><?echo $this->snr?></td>
      </tr>
      <tr>
        <td>NBIN</td>
        <td>
          <select name="nbin" id="nbin" onchange="update_candidate_image()">
            <option value="0" selected>default</option>
            <option value="4">4</option>
            <option value="8">8</option>
            <option value="16">16</option>
            <option value="32">32</option>
            <option value="64">64</option>
            <option value="128">128</option>
            <option value="256">256</option>
            <option value="512">512</option>
            <option value="1024">1024</option>
          </select>
        </td>
      </tr>
      <tr>
        <td>NCHAN</td>
        <td>
        <select name="nchan" id="nchan" onchange="update_candidate_image()">
          <option value="0" selected>default</option>
          <option value="4">4</option>
          <option value="8">8</option>
            <option value="16">16</option>
            <option value="32">32</option>
            <option value="64">64</option>
            <option value="128">128</option>
            <option value="256">256</option>
            <option value="512">512</option>
            <option value="1024">1024</option>
          </select>
        </td>
      </tr>
      <tr>
        <td>Length</td>
        <td>
          <select name="length" id="length" onchange="update_candidate_image()">
            <option value="0" selected>default</option>
            <option value="0.25">0.25</option>
            <option value="0.5">0.5</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
            <option value="6">6</option>
            <option value="7">7</option>
            <option value="8">8</option>
            <option value="9">9</option>
            <option value="10">10</option>
          </select>
        </td>
      </tr>
      <tr>
        <td>Zap From</td>
        <td><input name="zap_from" id="zap_from" value="<?echo $this->zap_from?>" onchange="update_candidate_image()"/></td>
      </tr>
      <tr>
        <td>Zap To</td>
        <td><input name="zap_to" id="zap_to" value="<?echo $this->zap_to?>" onchange="update_candidate_image()"/></td>
      </tr>
      <tr>
        <td>Proc Type</td>
        <td><?echo $this->proc_type?></td>
      </tr>
    </table>
<?
    $this->closeBlockHeader();
  }


  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("Candidate Event");
?>
    <img id="candidate_image" src="<?echo $this->cand_url?>"/>
<?
    $this->closeBlockHeader();
  }

  # get the candidate plot from the gpu cand server
  function printUpdateHTML($get)
  {
    $host = "caspsr-raid0";
    $port = "55555";

    $params  = "";
    $params .= "utc_start=".$get["utc_start"]." ";
    $params .= "beam=".$get["beam"]." ";
    $params .= "sample=".$get["sample"]." ";
    $params .= "filter=".$get["filter"]." ";
    $params .= "dm=".$get["dm"]." ";
    $params .= "snr=".$get["snr"]." ";
    $params .= "proc_type=".$get["proc_type"]." ";
    $params .= "nchan=".$get["nchan"]." ";
    $params .= "nbin=".$get["nbin"]." ";
    $params .= "length=".$get["length"]." ";
    $params .= "zap_from=".$get["zap_from"]." ";
    $params .= "zap_to=".$get["zap_to"];

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
