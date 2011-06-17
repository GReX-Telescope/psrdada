<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class custom_plot extends apsr_webpage 
{

  var $inst;
  var $base_dir;
  var $obs_dir;
  var $utc_start;
  var $sources = array();
  var $common_options = "-jpC";
  var $pvfl_options = "-p flux -jTF";
  var $pvt_options  = "-p time -jFD";
  var $pvfr_options = "-p freq -jTD";
  var $small_plot = "240x180";
  var $large_plot = "800x600";

  function custom_plot()
  {
    apsr_webpage::apsr_webpage();

    $this->inst = new apsr();
    $this->title = "APSR | Custom Plot";
    $this->utc_start = $_GET["utc_start"];
    if (isset($_GET["basedir"]))
      $this->base_dir = $_GET["basedir"];
    else
      $this->base_dir = $this->inst->config["SERVER_RESULTS_DIR"];
    $this->obs_dir = $this->base_dir."/".$this->utc_start;
  }

  function printJavaScriptHead()
  {
    $this->sources = $this->inst->getObsSources($this->obs_dir);
?>
    <style>
      table.source_info {
        font-size: 10pt;
      }
      table.source_info th {
        text-align: right;
        font-style: bold;
      }
      table.source_info td {
        text-align: left;
      }

    </style>

    <script type='text/javascript'>  

      function regen_all() 
      {
        regen("pvfl");
        regen("pvt");
        regen("pvfr");
      }

      function regen(type) 
      {
        var common_opts = document.getElementById("common_options").value;
        var specific_opts = document.getElementById(type).value;
        var ar_base = "<?echo $this->obs_dir?>/";
        var ar_ext = (type == "pvfr") ? "_f.ar" : "_t.ar";
        var plot_url = "pgwebplot.php?cmd=psrplot "+specific_opts+" "+
                       common_opts+" "+ar_base;
<?
        $srcs = array_keys($this->sources);
        for ($i=0; $i<count($srcs); $i++) {
          $src = str_replace("+", "%2B",$srcs[$i]);
          echo "      var img_".$i."  = document.getElementById('img_".$i."_'+type);\n";
          echo "      img_".$i.".src  = '/images/blackimage.gif';\n";
          echo "      var a_".$i."    = document.getElementById('a_".$i."_'+type);\n";
          echo "      var span_".$i." = document.getElementById('span_".$i."_'+type);\n";
          echo "      img_".$i.".src  = plot_url+'".$src."'+ar_ext+' -g $this->small_plot -s ".
               $this->inst->config["SCRIPTS_DIR"]."/web_style.txt';\n";
          echo "      a_".$i.".href   = plot_url+'".$src."'+ar_ext+' -g $this->large_plot';\n";
        }
?>
      }
    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
    <table cellspacing=0 cellpadding=0 border=0 width="100%">
      <tr>
        <td width='380px' height='60px'><img src='/apsr/images/apsr_logo.png' width='380px' height='60px'></td>
        <td height="60px" align=left style='padding-left: 20px'>
          <span class="largetext">Custom Plots</span>
        </td>
      </tr>
    </table>

    <table cellspacing=0 cellpadding="10px" border=0 width="100%">
      <tr>
        <td valign="top" width="150px">
<?
    $this->openBlockHeader("Plot Options");
?>
          <table class='source_info'>
            <tr>
              <td>Common Options</td>
              <td>
                <input type=text id="common_options" size="16" value="<?echo $this->common_options?>">&nbsp;&nbsp;
                <input type="button" value="Regen All" onClick="regen_all()">
              </td>
            </tr>
            <tr>
              <td>Phase v Flux</td>
              <td>
                <input type=text size="16" id="pvfl" value="<?echo $this->pvfl_options?>">&nbsp;&nbsp;
                <input type="button" onClick="regen('pvfl')" value="Regen">
              </td>
            </tr>
            <tr>
              <td>Phase v Time</td>
              <td>
                <input type=text size="16" id="pvt" value="<?echo $this->pvt_options?>">&nbsp;&nbsp;
                <input type="button" onClick="regen('pvt')" value="Regen">
              </td>
            </tr>
            <tr>
              <td>Phase v Freq</td>
              <td>
                <input type=text size="16" id="pvfr" value="<?echo $this->pvfr_options?>">&nbsp;&nbsp;
                <input type="button" onClick="regen('pvfr')" value="Regen">
              </td>
            </tr>
          </table>
<?
    $this->closeBlockHeader();
?>

        </td> 
      </tr>
      <tr>
        <td>
<?
    $this->openBlockHeader("Generated Results");
  
    $psrs = array_keys($this->sources);
    $web_style = $this->inst->config["SCRIPTS_DIR"]."/web_style.txt";

    list ($small_x, $small_y) = split("x", $this->small_plot);
  
    echo "<table>\n";

    for ($i=0; $i<count($psrs); $i++) 
    {
      $p = $psrs[$i];
      $p = str_replace("+", "%2B",$psrs[$i]);
      echo "<tr>\n";

      echo "  <td width=320><input type='hidden' id='source' value='".$p."'>\n";
      echo "    <table class='source_info'>\n";
      echo "      <tr><th>SOUCE</th><td>".$this->sources[$p]["src"]."</td></tr>\n";
      echo "      <tr><th>INT</th><td>".$this->sources[$p]["int"]."</td></tr>\n";
      echo "      <tr><th>DM</th><td>".$this->sources[$p]["dm"]."</td></tr>\n";
      echo "      <tr><th>P0</th><td>".$this->sources[$p]["p0"]."</td></tr>\n";
      echo "      <tr><th>SNR</th><td>".$this->sources[$p]["snr"]."</td></tr>\n";
      echo "    </table>\n";
      echo "  </td>\n";

      echo "  <td>\n";
      echo "    <a id='a_".$i."_pvfl' href='pgwebplot.php?cmd=psrplot ".$this->pvfl_options." ".$this->common_options." -g $this->large_plot ".$this->obs_dir."/".$p."_t.ar -D -/PNG''>\n";
      echo "    <img id='img_".$i."_pvfl' src='pgwebplot.php?cmd=psrplot ".$this->pvfl_options." ".$this->common_options." -g $this->small_plot ".$this->obs_dir."/".$p."_t.ar -s ".$web_style."' width='".($small_x+1)."px' height='".($small_y+1)."px'></a><br>\n";
      echo "  </td>\n";

      echo "  <td>\n";
      echo "    <a id='a_".$i."_pvt' href='pgwebplot.php?cmd=psrplot ".$this->pvt_options." ".$this->common_options." -g $this->large_plot ".$this->obs_dir."/".$p."_t.ar -D -/PNG'>";
      echo "    <img id='img_".$i."_pvt' src='pgwebplot.php?cmd=psrplot ".$this->pvt_options." ".$this->common_options." -g $this->small_plot  ".$this->obs_dir."/".$p."_t.ar -s ".$web_style."' width='".($small_x+1)."px' height='".($small_y+1)."px'></a><br>\n";
      echo "  </td>\n";

      echo "  <td>\n";
      echo "    <a id='a_".$i."_pvfr' href='pgwebplot.php?cmd=psrplot ".$this->pvfr_options." ".$this->common_options." -g $this->large_plot ".$this->obs_dir."/".$p."_f.ar -D -/PNG'>";
      echo "    <img id='img_".$i."_pvfr' src='pgwebplot.php?cmd=psrplot ".$this->pvfr_options." ".$this->common_options." -g $this->small_plot ".$this->obs_dir."/".$p."_f.ar -s ".$web_style."' width='".($small_x+1)."px' height='".($small_y+1)."px'></a><br>\n";

      echo "  </td>\n";
      echo "</tr>\n";
    }
    echo "</table>\n";
    $this->closeBlockHeader();
?>
        </td>
      </tr>
    </table>
<?
  }
}

handledirect("custom_plot");

