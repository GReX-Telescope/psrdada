<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

class live_candidate_viewer extends bpsr_webpage 
{
  var $utc_start;

  var $beam;

  var $sample;

  var $filter;

  var $dm;

  var $snr;

  var $nbin;

  var $nchan;

  var $proc_type;

  var $inst;

  var $cand_url;

  function live_candidate_viewer()
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
    $this->nchan = isset($_GET["nchan"]) ? $_GET["nchan"] : "";
    $this->nbin = isset($_GET["nbin"]) ? $_GET["nbin"] : "";
    $this->length= isset($_GET["length"]) ? $_GET["length"] : "0";
    $this->proc_type = isset($_GET["proc_type"]) ? $_GET["proc_type"] : "cand";

    $this->cand_url = "live_candidate_viewer.lib.php?update=true";
    if ($this->utc_start != "") $this->cand_url .= "&utc_start=".$this->utc_start;
    if ($this->beam != "") $this->cand_url .= "&beam=".$this->beam;
    if ($this->sample != "") $this->cand_url .= "&sample=".$this->sample;
    if ($this->filter != "") $this->cand_url .= "&filter=".$this->filter;
    if ($this->dm != "") $this->cand_url .= "&dm=".$this->dm;
    if ($this->snr != "") $this->cand_url .= "&snr=".$this->snr;
    if ($this->length != "") $this->cand_url .= "&length=".$this->length;
    if ($this->proc_type != "") $this->cand_url .= "&proc_type=".$this->proc_type;

    $this->cand_url .= "&length=".$this->length;
    $this->cand_url .= "&nchan=".$this->nchan;
    $this->cand_url .= "&nbin=".$this->nbin;

    if ($this->utc_start == "")
    {
      $cmd = "find ".$this->inst->config["SERVER_RESULTS_DIR"]." -mindepth 2 -maxdepth 2 ".
             "-type f -name 'all_candidates.dat' -printf '%h\n' | sort | tail -n 1 | awk -F/ '{print \$NF}'";
      $output = array();
      $this->utc_start = exec($cmd, $output, $rval);
    }
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  
      function update_live_candidate_image() 
      {
        url = "<?echo $this->cand_url;?>";

        var i = document.getElementById("nchan").selectedIndex;
        var nchan = document.getElementById("nchan").options[i].value;

        var i = document.getElementById("nbin").selectedIndex;
        var nbin = document.getElementById("nbin").options[i].value;

        var i = document.getElementById("length").selectedIndex;
        var length = document.getElementById("length").options[i].value;

        url += "&nbin=" + nbin + "&nchan=" + nchan + "&length=" + length;

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
        <td>SNR</td>
        <td><?echo $this->snr?></td>
      </tr>
      <tr>
        <td>NBIN</td>
        <td>
          <select name="nbin" id="nbin" onchange="update_live_candidate_image()">
            <option value="0">default</option>
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
        <select name="nchan" id="nchan" onchange="update_live_candidate_image()">
          <option value="0">default</option>
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
         <select name="length" id="length" onchange="update_live_candidate_image()">
          <option value="0">default</option>
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
    <img id='candidate_image' src="<?echo $this->cand_url?>">
<?
    $this->closeBlockHeader();
  }

  # get the candidate plot from the gpu cand server
  function printUpdateHTML($get)
  {
    $cfg = $this->inst->config;

    # determine which host has this beam
    $host = "";
    for ($i=0; (($host == "") && ($i<$cfg["NUM_PWC"])); $i++)
    {
      if ($this->inst->roach["BEAM_".$i] == $this->beam)
      {
        $host = $cfg["PWC_".$i];
      }
    }

    $fil_file = $cfg["CLIENT_ARCHIVE_DIR"]."/".$this->beam."/".$this->utc_start."/".$this->utc_start.".fil";
    $plot_cmd = "trans_freqplus_plot.py ".$fil_file." ".$this->sample." ".$this->dm." ".$this->filter." ".$this->snr;
    if ($this->nchan != "")
      $plot_cmd .= " -nchan ".$this->nchan;
    if ($this->nbin != "")
      $plot_cmd .= " -nbin ".$this->nbin;
    if ($this->length != "")
      $plot_cmd .= " -length ".$this->length;
    $cmd = "ssh -o BatchMode=yes -x ".$cfg["USER"]."@".$host." '".$plot_cmd."'";

    $img_data = `$cmd`;

    header('Content-Type: image/png');
    header('Content-Disposition: inline; filename="image.png"');
    echo $img_data;
  }
}

handleDirect("live_candidate_viewer");
