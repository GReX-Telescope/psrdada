<?PHP

include_once("bpsr.lib.php");
include_once("bpsr_webpage.lib.php");

putenv("PATH=/bin:/usr/bin");

class transient_results extends bpsr_webpage 
{

  var $images;

  var $inst;
  
  function transient_results()
  {
    bpsr_webpage::bpsr_webpage();
    $this->title = "BPSR Transient Pipeline";

    $this->inst = new bpsr();

    # retrieve the list of all transient images from results dir
    $cmd = "find ".$this->inst->config["SERVER_RESULTS_DIR"]." -mindepth 2 -maxdepth 2 -type f -name '*cands_*x*.png' | sort";
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
    <style type='text/css'>

      span {
        padding-left: 5px;
      }

      td {
        text-align: center;
      }

    </style>


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
        document.getElementById("curr_utc").innerHTML = utcs[curr_index]
        document.getElementById("links").innerHTML = "<a href='"+result_url+"'>Results Page</a> <a href='"+transient_url+"'>Transient Viewer</a>";
        transient_results_request(utcs[curr_index]);

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
        document.getElementById("curr_utc").innerHTML = utcs[curr_index]
        document.getElementById("links").innerHTML = "<a href='"+result_url+"'>Results Page</a> <a href='"+transient_url+"'>Transient Viewer</a>";
        transient_results_request (utcs[curr_index]);


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
        document.getElementById("curr_utc").innerHTML = utcs[curr_index]
        document.getElementById("links").innerHTML = "<a href='"+result_url+"'>Results Page</a> <a href='"+transient_url+"'>Transient Viewer</a>";
        transient_results_request (utcs[curr_index]);
        
      }

      function handle_transient_results_request(tr_xml_request) 
      {
        if (tr_xml_request.readyState == 4)
        {
          var xmlDoc = tr_xml_request.responseXML
          if (xmlDoc != null)
          {
            var xmlObj = xmlDoc.documentElement;

            var obs = xmlObj.getElementsByTagName("obs");
            for (iob=0; iob<obs.length; iob++)
            {
              var ob = obs[iob];
              var utc_start = ob.getAttribute("utc_start");

              children = ob.childNodes;
              for (i=0; i<children.length; i++)
              {
                child = children[i];
                if (child.nodeType == 1)
                {
                  if (child.nodeName == "pid")
                  {
                    document.getElementById("pid").innerHTML = child.childNodes[0].nodeValue;
                  }
                  if (child.nodeName == "source")
                  {
                    document.getElementById("source").innerHTML = child.childNodes[0].nodeValue;
                  }
                }
              }
            } 
          }
        }
      }
      
      function transient_results_request (utc_start) 
      {
        url = "transient_results.lib.php?update=true&utc_start="+utc_start;

        if (window.XMLHttpRequest)
          tr_xml_request = new XMLHttpRequest();
        else
          tr_xml_request = new ActiveXObject("Microsoft.XMLHTTP");

        tr_xml_request.onreadystatechange = function() {
          handle_transient_results_request(tr_xml_request)
        };
        tr_xml_request.open("GET", url, true);
        tr_xml_request.send(null);
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
      <td align=center>
        <span id='prev_utc'></span><br/>
        <input type="button" name='prev_button' value="Prev" onClick="prev_obs()"/>
      </td>
      <td align=center>
        <b><span id='curr_utc'></span></b>
        <span id='pid'></span>
        <span id='source'></span>
        <br/>
        <div class="clear"/>
        <span id='links'></span>
      </td>
      <td align=center>
        <span id='next_utc'></span><br>
        <input type="button" name='next_button' value="Next" onClick="next_obs()"/>
      </td>
    </tr>
    <tr>
      <td colspan=3>
        <img src="" border=0 width=1024 height=768 id="current" TITLE="Current Candidate" alt="alt">
      </td>
    </tr>
  </table>
  <script type='text/javascript'>
    set_curr_image(utcs.length - 1);
  </script>
<?
  }

  function printUpdateHTML($get)
  {
    $utc = $get["utc_start"];
    $pid = "PXXX";
    $source = "Unknown";

    $results = array();
    $rval = 0;

    # get the PID and SOURCE from the obs.info
    $cmd = "grep ^PID ".$this->inst->config["SERVER_RESULTS_DIR"]."/".$utc."/obs.info | awk '{print \$2}'";
    $last_line = exec($cmd, $results, $rval);
    if ($rval == 0)
    {
      $pid = $last_line;
    }

    $results = array();
    $rval = 0;
    $cmd = "grep ^SOURCE ".$this->inst->config["SERVER_RESULTS_DIR"]."/".$utc."/obs.info | awk '{print \$2}'";
    $last_line = exec($cmd, $results, $rval);
    if ($rval == 0)
    { 
      $source = $last_line;
    }

    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<transient_results>";
    $xml .= "<obs utc_start='".$utc."'>";
    $xml .= "<source>".$source."</source>";
    $xml .= "<pid>".$pid."</pid>";
    $xml .= "</obs>";
    $xml .= "</transient_results>";

    header('Content-type: text/xml');
    echo $xml;
  }
  
}

handleDirect("transient_results");

