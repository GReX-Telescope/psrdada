<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

putenv("PATH=/bin:/usr/bin");

class transient_results extends bpsr_webpage 
{

  var $images;

  function transient_results()
  {
    bpsr_webpage::bpsr_webpage();
    $this->title = "BPSR Transient Pipeline";

    $inst = new bpsr();

    # retrieve the list of all transient images from results dir
    $cmd = "find ".$inst->config["SERVER_RESULTS_DIR"]." -mindepth 2 -maxdepth 2 -type f -name '*cands_*x*.png' | sort";
    $results = array();
    $last_line = exec($cmd, $results, $rval);

    $this->images = array();
    for ($i=0; $i<count($results); $i++)
    {
      $bits = explode("/", $results[$i]);
      $nbits = count($bits);
      if ($nbits > 2)
      {
        $utc = $bits[$nbits-2];
        $file = $bits[$nbits-1];
        $this->images[$utc] = $file;
      }
    }
  }

  function javaScriptCallback()
  {
    //return "transient_results_request();";
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      var http_server = "http://<?echo $_SERVER["SERVER_NAME"].":".$_SERVER["SERVER_PORT"]?>"
      var url_prefix = "/bpsr/results/";

      var curr_index = -1;
      var prev_index = -1;
      var next_index = -1;

      var prev_img = new Image();
      var next_img = new Image();

      var utcs = new Array(<?
      $utcs = array_keys($this->images);
      echo "\"".$utcs[0]."\"";
      for ($i=0; $i<count($utcs); $i++)
      {
        echo ",\"".$utcs[$i]."\"";
      }?>);

      var imags = new Array(<?
      $utcs = array_keys($this->images);
      echo "\"".$this->images[$utcs[0]]."\"";
      for ($i=0; $i<count($utcs); $i++)
      {
        echo ",\"".$this->images[$utcs[$i]]."\"";
      }
      ?>);

      var n_imags = imags.length - 1;

      function set_curr_image(new_index)
      {

        curr_index = new_index;
        prev_index = Math.max(0, curr_index - 1);
        next_index = Math.min(n_imags, curr_index + 1);

        var result_url    = http_server + "/bpsr/result.lib.php?single=true&utc_start=" + utcs[curr_index];
        var transient_url = http_server + "/bpsr/transient_viewer.lib.php?single=true&utc_start=" + utcs[curr_index];

        // load the specified new_index image immediately
        document.getElementById("current").src = http_server + url_prefix + utcs[curr_index] + "/" + imags[curr_index];
        document.getElementById("curr_utc").innerHTML = utcs[curr_index] + ": <a href='"+result_url+"'>Results Link</a> <a href='"+transient_url+"'>Transient Link</a>";

        prev_img.src = http_server + url_prefix + utcs[prev_index] + "/" + imags[prev_index];
        document.getElementById("prev_utc").innerHTML = utcs[prev_index];

        next_img.src = http_server + url_prefix + utcs[next_index] + "/" + imags[next_index];
        document.getElementById("next_utc").innerHTML = utcs[next_index];

      }

      function next_obs() 
      {
        // copy curr -> prev
        prev_img.src = document.getElementById("current").src;
        prev_index = curr_index;

        // copy next -> curr 
        document.getElementById("current").src = next_img.src;
        curr_index = next_index;

        // get a new next img in background
        next_index = Math.min(n_imags, curr_index + 1);
        next_img.src = http_server + url_prefix + utcs[next_index] + "/" + imags[next_index];

        document.getElementById("prev_utc").innerHTML = utcs[prev_index];
        document.getElementById("next_utc").innerHTML = utcs[next_index];

        var result_url    = http_server + "/bpsr/result.lib.php?single=true&utc_start=" + utcs[curr_index];
        var transient_url = http_server + "/bpsr/transient_viewer.lib.php?single=true&utc_start=" + utcs[curr_index];
        document.getElementById("curr_utc").innerHTML = utcs[curr_index]+ ": <a href='"+result_url+"'>Results Link</a> <a href='"+transient_url+"'>Transient Link</a>";
      }

      function prev_obs() 
      {
        // copy curr -> next
        next_img.src = document.getElementById("current").src;
        next_index = curr_index;

        // copy prev -> curr
        document.getElementById("current").src = prev_img.src;
        curr_index = prev_index;

        // get new prev img in background
        prev_index = Math.max(0, prev_index - 1);
        prev_img.src = http_server + url_prefix + utcs[prev_index] + "/" + imags[prev_index];

        document.getElementById("prev_utc").innerHTML = utcs[prev_index];
        document.getElementById("next_utc").innerHTML = utcs[next_index];
        
        var result_url    = http_server + "/bpsr/result.lib.php?single=true&utc_start=" + utcs[curr_index];
        var transient_url = http_server + "/bpsr/transient_viewer.lib.php?single=true&utc_start=" + utcs[curr_index];
        document.getElementById("curr_utc").innerHTML = utcs[curr_index]+ ": <a href='"+result_url+"'>Results Link</a> <a href='"+transient_url+"'>Transient Link</a>";
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
  <table cellpadding="10px" border="1">
    <tr>
      <th><input type="button" value="Previous" onClick="prev_obs()"></th>
      <th>Current Obs</th>
      <th><input type="button" value="Next" onClick="next_obs()"></th>
    </tr>
    <tr>
      <td valign=top style='text-align:center;'><span id='prev_utc'></span>
      <td valign=top style='text-align:center;'><span id='curr_utc'></span>
      <td valign=top style='text-align:center;'><span id='next_utc'></span>
    </tr>
    <tr>
      <td colspan=3>
        <img src="" border=0 width=1024 height=768 id="current" TITLE="Current Candidate" alt="alt">
      </td>
    </tr>
  </table>
  <script type='text/javascript'>
    set_curr_image(utcs.length - 1);
    //set_curr_image(0);
  </script>
      
<?
  }
}

handleDirect("transient_results");

