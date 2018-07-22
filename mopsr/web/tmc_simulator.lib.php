<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class tmc_simulator extends mopsr_webpage 
{
  var $psrs = array();
  var $psr_keys = array();
  var $valid_psrs = array();
  var $valid_sources = array();
  var $inst = 0;
  var $passthru = false;
  var $oversampling = false;
  var $ntiedbeams = 1;

  # defaults are critically sampled PFB
  var $bandwidth = 31.25;
  var $cfreq = 834.765625;
  var $nchan = 40;
  var $channel_bandwidth = 0.78125;
  var $oversampling_ratio = 1;
  var $sampling_time = 1.28;
  var $dsb = 1;
  var $nant = 16;
  var $resolution;
  var $hires = 0;

  function tmc_simulator()
  {
    mopsr_webpage::mopsr_webpage();

    $this->title = "MOPSR | TMC Simulator";
    $this->inst = new mopsr();

    # determine if we are configured in low or hi res mode
    $config_name = $this->inst->config["CONFIG_NAME"];
    if (strpos($config_name, "320chan") !== FALSE)
    {
      $this->hires = 1;
      $this->nchan = 320;
      $this->channel_bandwidth = 0.09765625;
      $this->sampling_time = 10.24;
      # AJ TODO remove for original hires mode
      $this->nant = 8;
      $this->cfreq = 835.4980469; # channel 204
      $this->cfreq = 835.5957031; # channel 205
    }

    $this->bf_cfg = $this->inst->configFileToHash(BF_FILE);
    $this->bp_ct_cfg = $this->inst->configFileToHash(BP_CNR_FILE);
    $this->ntiedbeams = $this->bf_cfg["NUM_TIED_BEAMS"];

    # read in the RA/DECs for PSRCAT pulsars
    $this->psrs = $this->inst->getPsrcatPsrs();

    # Trim the list of valid pulsars to those specifiied
    $this->load_valid_psrs();

    # Add in the RA/DECs for calibrators and other sources
    $this->load_sources();

    if ($this->oversampling)
    {
      $this->sampling_time = 1.08;
      $this->channel_bandwidth = "0.9259259259";
    }
    if ($this->passthru)
    {
      $this->nchan = 1;
      $this->chann_bandwidth = 100;
      $this->sampling_time = 0.01;
      $this->dsb = 0;
    }

    $this->resolution = $this->nchan * $this->nant * 2;
    $this->nbeams = $this->bp_ct_cfg["NBEAM"];
  }
  
  function load_valid_psrs()
  {
    # load the calid pulsars from pulsars.list
    $fptr = @fopen("pulsars.list","r");
    if (!$fptr) 
    {
      echo "Could not open file: pulsars.list for reading<BR>\n";
    } 
    else 
    {
      while ($line = fgets($fptr, 1024)) 
      {
        $psr = chop($line);
        array_push ($this->valid_psrs, $psr);
        array_push ($this->valid_sources, $psr);
      }
      fclose ($fptr);
    }
    
  }

  function load_sources ()
  {
    $fptr = @fopen("sources.list","r");
    if (!$fptr)
    {
      echo "Could not open file: sources.list for reading<BR>\n";
    }
    else
    {
      while ($line = fgets($fptr, 1024))
      {
        $line = chop($line);
        //list ($source, $ra, $dec) = explode("\t ", $line);
        list ($source, $ra, $dec) = preg_split('/\s+/', $line);

        array_push ($this->valid_sources, $source);
        $this->psrs[$source]["RAJ"] = $ra;
        $this->psrs[$source]["DECJ"] = $dec;
      }
      fclose ($fptr);
    }
  }

  function printJavaScriptHead()
  {
    $this->psr_keys = array_keys($this->psrs);
?>
    <style type='text/css'>

      td.key {
        text-align: right;
      }
 
      td.val {
        padding-right: 20px;
        text-align: left;
      } 

    </style>


    <script type='text/javascript'>
      var ras = { 'default':'00:00:00.00'<?
      for ($i=0; $i<count($this->psr_keys); $i++)
      {
        $p = $this->psr_keys[$i];
        if (in_array($p, $this->valid_psrs) || in_array($p, $this->valid_sources))
        {
          echo ",'".$p."':'".$this->psrs[$p]["RAJ"]."'";
        }
      }
      ?>};

      var decs = { 'default':'00:00:00.00'<?
      for ($i=0; $i<count($this->psr_keys); $i++)
      {
        $p = $this->psr_keys[$i];
        if (in_array($p, $this->valid_psrs) || in_array($p, $this->valid_sources))
        {
          echo ",'".$p."':'".$this->psrs[$p]["DECJ"]."'";
        }
      }
      ?>};


      function prepareButton() 
      {
        document.getElementById("command").value = "prepare";

        var i = 0;
        var lim = <?echo $this->ntiedbeams?>;
        var psr = "";

        for (j=0; j<=lim; j++)
        {
          i = document.getElementById(j+"_src_list").selectedIndex;
          psr = document.getElementById(j+"_src_list").options[i].value;
          if (psr != "FRB_Transit")
            updateRADEC(j);
          document.getElementById(j+"_source").value = psr;
        }

        document.tmc.submit();
      }

      function startButton()
      {
        document.getElementById("command").value = "start";
        document.tmc.submit();
      }

      function stopButton() {
        document.getElementById("command").value = "stop";
        document.tmc.submit();
      }

      function queryButton() {
        document.getElementById("command").value = "query";
        document.tmc.submit();
      }

      function updateRADEC(prefix) {
        var i = document.getElementById(prefix+"_src_list").selectedIndex;
        var psr = document.getElementById(prefix+"_src_list").options[i].value;
        if (psr == "FRB_Transit") {
          document.getElementById(prefix+"_ra").removeAttribute('readonly');
          document.getElementById(prefix+"_dec").removeAttribute('readonly');
        } else {
          document.getElementById(prefix+"_ra").setAttribute("readonly", "readonly");
          document.getElementById(prefix+"_dec").setAttribute("readonly", "readonly");
        }
        var psr_ra = ras[psr];
        var psr_dec= decs[psr];
        document.getElementById(prefix+"_ra").value = psr_ra;
        document.getElementById(prefix+"_dec").value = psr_dec;
      }
    </script>

<?
  }

  /*************************************************************************************************** 
   *
   * HTML for this page 
   *
   ***************************************************************************************************/
  function printHTML()
  {
    $this->openBlockHeader("TMC Simulator");
?>
    <form name="tmc" target="tmc_interface" method="GET">

    <table border=0 cellpadding=5 cellspacing=0>
      <tr>
        <td width='150px'><b>Telescope</b></td>

        <td class='key'>TRACKING</td>
        <td class='val'><input type="checkbox" name="antenna_tracking"></td>
 
        <td class='key'>MD Angle</td>
        <td class='val'><input type="text" name="md_angle" size="8" value="0.0"> [degrees]</td>

        <td class='key'>NS Tilt</td>
        <td class='val'><input type="text" name="ns_tilt" size="8" value="0.0"> [degrees]</td>
      </tr>     
    </table>

    <!-- Signal Parameters -->
    <table border=0 cellpadding=5 cellspacing=0>
      <tr>
        <td width='150px'><b>Signal</b></td>

        <td class='key'>NCHAN</td>
        <td class='val'><input type="text" name="nchan" size="3" value="<?echo $this->nchan?>" readonly></td>

        <td class='key'>NBIT</td>
        <td class='val'><input type="text" name="nbit" size="2" value="8" readonly></td>
  
        <td class='key'>NDIM</td>
        <td class='val'><input type="text" name="ndim" size="2" value="2" readonly></td>
      
        <td class='key'>NPOL</td>
        <td class='val'><input type="text" name="npol" size="1" value="1" readonly></td>

        <td class='key'>NANT</td>
        <td class='val'><input type="text" id="nant" name="nant" size="3" value="<?echo $this->nant?>"/ readonly></td>

        <td class='key'>BANDWIDTH</td>
        <td class='val'><input type="text" name="bandwidth" value="<?echo $this->bandwidth?>" size="12" readonly></td>

        <td class='key'>FREQ</td>
        <td class='val'><input type="text" name="centre_frequency" size="12" value="<?echo $this->cfreq?>" readonly></td>
      </tr>
    </table>

    <!-- PFB Parameters -->
    <table border=0 cellpadding=5 cellspacing=0>
      <tr>
        <td width='150px'><b>PFB</b></td>

        <td class='key'>OS Ratio</td>
        <td class='val'><input type="text" name="oversampling_ratio" size="16" value="<?echo $this->oversampling_ratio?>" readonly></td>
        
        <td class='key'>TSAMP</td>
        <td class='val'><input type="text" name="sampling_time" size="12" value="<?echo $this->sampling_time?>" readonly></td>

        <td class='key'>CHANBW</td>
        <td class='val'><input type="text" name="channel_bandwidth" size="12" value="<?echo $this->channel_bandwidth?>" readonly></td>
        
        <td class='key'>DSB</td>
        <td class='val'><input type="text" name="dsb" size="1" value="<?echo $this->dsb?>" readonly></td>

        <td class='key'>RESOLUTION</td>
        <td class='val'><input type="text" name="resolution" size="4" value="<?echo $this->resolution?>" readonly></td>
      </tr>
    </table>


    <!-- Observation Parameters -->
    <table border=0 cellpadding=5 cellspacing=0>
      <tr>
        <td width='150px'><b>Common Observation</b></td>

        <td class='key'>OBSERVER</td>
        <td class='val'><input type="text" name="observer" size="6" value="None"></td>

<!--
        <td class='key'>CONFIG</td>
        <td class='val'>
          <select name="config">
            <option value="INDIVIDUAL_MODULES">INDIVIDUAL MODULES</option>
            <option value="CORRELATION">CORRELATION</option>
            <option value="TIED_ARRAY_BEAM">TIED ARRAY BEAM</option>
            <option value="FAN_BEAM">FAN BEAM</option>
            <option value="TIED_ARRAY_FAN_BEAM">TIED ARRAY &amp; FAN BEAM</option>
            <option value="MOD_BEAM">MOD BEAM</option>
            <option value="TIED_ARRAY_MOD_BEAM">TIED ARRAY &amp; MOD BEAM</option>
          </select>
        </td>
-->

        <td class='key'>MODE</td>
        <td class='val'>
          <select name="mode">
            <option value="PSR">PSR</option>
            <option value="CORR">CORR</option>
            <option value="CORR_CAL">CORR_CAL</option>
          </select>
        </td>

        <td class='key'>TOBS</td>
        <td class='val'><input type="text" name="tobs" size="6" value="-1"></td>

      </tr>
    </table>

    <!-- Boresight source parameters-->
    <table border=0 cellpadding=5 cellspacing=0>
      <tr>
        <td width='150px'><b>Boresight [AQ]</b></td>

        <? $prefix = "0"; ?>
        <td class='key'>SOURCE</td>
        <td class='val'>
          <input type="hidden" id="<?echo $prefix?>_source" name="<?echo $prefix?>_source" value="">
          <select id="<?echo $prefix?>_src_list" name="<?echo $prefix?>_src_list" onChange='updateRADEC("0")'>
            <option value=''>--</option>
<?
          for ($j=0; $j<count($this->valid_sources); $j++)
          {
            $p = $this->valid_sources[$j];
            echo "            <option value='".$p."'>".$p."</option>\n";
          }
?>
          </select>
        </td>

        <td class='key'>RA</td>
        <td class='val'><input type="text" id="0_ra" name="0_ra" size="12" value="" readonly></td>

        <td class='key'>DEC</td>
        <td class='val'><input type="text" id="0_dec" name="0_dec" size="12" value="" readonly></td>

        <td class='key'>Project ID</td>
        <td class='val'><input type="text" name="project_id" size="4" value="P999"></td>

        <td class='key'>RFI MITIGATION</td>
        <td class='val'><input type="checkbox" name="rfi_mitigation" checked></td>

        <td class='key'>DELAY TRACKING</td>
        <td class='val'><input type="checkbox" name="delay_tracking" checked></td>
       
        <td class='key'>ANTENNA WEIGHTS</td>
        <td class='val'><input type="checkbox" name="antenna_weights" checked></td>
       
        <td class='key'>PROC FILE</td>
        <td class='val'>
          <select name="0_processing_file">
<?
        if ($this->hires)
          echo '<option value="mopsr.aqdsp.hires.gpu">mopsr.aqdsp.hires.gpu</option>\n';
        else
          echo '<option value="mopsr.aqdsp.gpu">mopsr.aqdsp.gpu</option>\n';
?>
            <option value="mopsr.null">mopsr.null</option>
            <option value="mopsr.dbdisk">mopsr.dbdisk</option>
          </select>
        </td>


      </tr>
    </table>

<?
    for ($i=1; $i<=$this->ntiedbeams; $i++)
    {
      $prefix = $i;
?>

    <table border=0 cellpadding=5 cellspacing=0>
      <tr>
        <td width='150px'><b>Tied Beam <?echo $i?></b></td>

        <td class='key'>ENABLED</td>
        <td class='val'><input type="checkbox" name="<?echo $prefix?>_enabled"></td>

        <td class='key'>Project ID</td>
        <td class='val'><input type="text" name="<?echo $prefix?>_project_id" size="4" value="P999"></td>

        <td class='key'>SOURCE</td>
        <td class='val'>
          <input type="hidden" id="<?echo $prefix?>_source" name="<?echo $prefix?>_source" value="">
          <select id="<?echo $prefix?>_src_list" name="<?echo $prefix?>_src_list" onChange='updateRADEC("<?echo $i?>")'>
            <option value='' selected>--</option>
<?
          for ($j=0; $j<count($this->psr_keys); $j++)
          {
            $p = $this->psr_keys[$j];
            if (in_array($p, $this->valid_psrs))
            {
              echo "            <option value='".$p."'>".$p."</option>\n";
            }
          }
?>
          </select>
        </td>

        <td class='key'>RA</td>
        <td class='val'><input type="text" id="<?echo $prefix?>_ra" name="<?echo $prefix?>_ra" size="12" value="" readonly></td>

        <td class='key'>DEC</td>
        <td class='val'><input type="text" id="<?echo $prefix?>_dec" name="<?echo $prefix?>_dec" size="12" value="" readonly></td>

        <td class='key'>PROC FILE</td>
        <td class='val'>
          <select name="<?echo $prefix?>_processing_file">
            <option value="mopsr.dspsr.cpu">mopsr.dspsr.cpu</option>
            <option value="mopsr.dspsr.cpu.5s">mopsr.dspsr.cpu.5s</option>
            <option value="mopsr.dspsr.cpu.single">mopsr.dspsr.cpu.single</option>
<?          if ($this->hires) { ?>
            <!--<option value="mopsr.dspsr.cpu.cdd.hires">mopsr.dspsr.cpu.cdd.hires</option>-->
<?          } else { ?>
            <option value="mopsr.dspsr.cpu.cdd" selected>mopsr.dspsr.cpu.cdd</option>
<?          } ?>
            <option value="mopsr.null">mopsr.null [discard]</option>
          </select>
        </td>

      </tr>
    </table>
<?  } ?>

    <table border=0 cellpadding=5 cellspacing=0>
      <tr>
        <td width='150px'><b>Correlation</b></td>

        <td class='key'>ENABLED</td>
        <td class='val'><input type="checkbox" name="correlation_enabled"></td>

        <td class='key'>Project ID</td>
        <td class='val'><input type="text" name="correlation_project_id" size="4" value="P999"></td>

        <td class='key'>TYPE</td>
        <td class='val'>
          <select name="correlation_type">
            <option value="FX">FX</option>
            <option value="X">X</option>
          </select>
        </td>

        <td class='key'>DUMP TIME</td>
        <td class='val'><input type="text" name="correlation_dump_time" size="4" value="60"></td>

        <td class='key'>PROC FILE</td>
        <td class='val'>
          <select name="corr_processing_file">
<?          if ($this->hires) {
              echo '<option value="mopsr.calib.hires.pref16.gpu">mopsr.calib.hires.pref16.gpu [limited baselines 352 input]</option>\n';
              echo '<option value="mopsr.calib.hires.gpu">mopsr.calib.hires.gpu</option>\n';
            } else {
              echo '<option value="mopsr.calib.pref16.gpu" selected>mopsr.calib.pref16.gpu [limited baselines 352 input]</option>\n';
              echo '<option value="mopsr.calib.gpu">mopsr.calib.gpu [correlator]</option>\n';
              echo '<option value="mopsr.calib.xgpu">mopsr.calib.xgpu [dev correlator]</option>\n';
            }
?>
            <option value="mopsr.null">mopsr.null [discard]</option>
          </select>
        </td>

      </tr>
    </table>


    <table border=0 cellpadding=5 cellspacing=0>
      <tr>
        <td width='150px'><b>Fan Beam [BP]</b></td>

        <td class='key'>ENABLED</td>
        <td class='val'><input type="checkbox" name="fan_beams_enabled"></td>

        <td class='key'>Project ID</td>
        <td class='val'><input type="text" name="fan_beams_project_id" size="4" value="P999"></td>

        <td class='key'>NBEAMS</td>
        <td class='val'><input type="text" name="nbeams" size="4" value="<?echo $this->nbeams?>" readonly></td>

        <td class='key'>BEAM SPACING</td>
        <td class='val'><input type="text" name="beam_spacing" name="ra" size="12" value="<?echo 4.0 / ($this->nbeams -1)?>"> [degrees]</td>

      </tr>
    </table>

    <table border=0 cellpadding=5 cellspacing=0>
      <tr>      
        <td width='150px'><b>Module Beam [BP]</b></td>
          
        <td class='key'>ENABLED</td>
        <td class='val'><input type="checkbox" name="mod_beams_enabled"></td>
        
        <td class='key'>Project ID</td>
        <td class='val'><input type="text" name="mod_beams_project_id" size="4" value="P999"></td>
      </tr>
    </table>

    <table border=0 cellpadding=5 cellspacing=0>
      <tr>      
        <td width='150px'><b>FRB Injection</b></br><font size="-1">comma delim, no whitespace</font></td>

        <td class='key'>NUMBER</td>
        <td class='val'>
          <select name="num_furbies">
<?
            for ($i=0; $i<=5; $i++)
              echo "<option value='".$i."'>".$i."</option>\n";
?>
          </select> 

        <td class='key'>IDs</td>
        <td class='val'><input type="text" name="furby_ids" id="furby_ids" value="0309" size="24"></input></td>

        <td class='key'>BEAMs</td>
        <td class='val'><input type="text" name="furby_beams" id="furby_beams" value="177" size="24"></input></td>

        <td class='key'>TIMESTAMP</td>
        <td class='val'><input type="text" name="furby_tstamps" id="furby_tstamps" size="32" value="60"></td>

      </tr>

    </table>

    <h3>Controls</h3>
    <table border=0 cellpadding=5 cellspacing=0 width='100%'>

      <tr>
        <td colspan=4>
          <div class="btns" style='text-align: center'>
            <a href="javascript:prepareButton()"  class="btn" > <span>Prepare</span> </a>
            <a href="javascript:startButton()"  class="btn" > <span>Start</span> </a>
            <a href="javascript:stopButton()"  class="btn" > <span>Stop</span> </a>
            <a href="javascript:queryButton()"  class="btn" > <span>Query</span> </a>
          </div>
        </td>
      </tr>
    </table>
    <input type="hidden" id="command" name="command" value="">
    </form>
<?
    $this->closeBlockHeader();

    echo "<br/>\n";

    // have a separate frame for the output from the TMC interface
    $this->openBlockHeader("TMC Interface");
?>
    <iframe name="tmc_interface" src="" width=100% frameborder=0 height='350px'></iframe>
<?
    $this->closeBlockHeader();
  }

  function printTMCResponse($get)
  {
    // Open a connection to the TMC interface script
    $host = $this->inst->config["TMC_INTERFACE_HOST"];
    $port = $this->inst->config["TMC_INTERFACE_PORT"];
    $sock = 0;

    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>";
    $xml .= "<mpsr_tmc_message>";

    if ($get["command"] == "stop")
    {
      $xml .= "<command>stop</command>";
    }
    else if ($get["command"] == "prepare")
    {
      $xml .= "<command>prepare</command>";

      $xml .= "<west_arm_parameters>";
      $xml .=   "<tracking>".$this->checkbox("antenna_tracking", $get)."</tracking>";
      $xml .=   "<ns_tilt units='degrees'>".$get["ns_tilt"]."</ns_tilt>";
      $xml .=   "<md_angle units='degrees'>".$get["md_angle"]."</md_angle>";
      $xml .= "</west_arm_parameters>";
      $xml .= "<east_arm_parameters>";
      $xml .=   "<tracking>".$this->checkbox("antenna_tracking", $get)."</tracking>";
      $xml .=   "<ns_tilt units='degrees'>".$get["ns_tilt"]."</ns_tilt>";
      $xml .=   "<md_angle units='degrees'>".$get["md_angle"]."</md_angle>";
      $xml .= "</east_arm_parameters>";

      $xml .= "<signal_parameters>";
      $xml .=   "<nchan>".$get["nchan"]."</nchan>";
      $xml .=   "<nbit>".$get["nbit"]."</nbit>";
      $xml .=   "<ndim>".$get["ndim"]."</ndim>";
      $xml .=   "<npol>".$get["npol"]."</npol>";
      $xml .=   "<nant>".$get["nant"]."</nant>";
      $xml .=   "<bandwidth units='MHz'>".$get["bandwidth"]."</bandwidth>";
      $xml .=   "<centre_frequency units='MHz'>".$get["centre_frequency"]."</centre_frequency>";
      $xml .= "</signal_parameters>";

      $xml .= "<pfb_parameters>";
      $xml .=   "<oversampling_ratio>".$get["oversampling_ratio"]."</oversampling_ratio>";
      $xml .=   "<sampling_time units='microseconds'>".$get["sampling_time"]."</sampling_time>";
      $xml .=   "<channel_bandwidth units='MHz'>".$get["channel_bandwidth"]."</channel_bandwidth>";
      $xml .=   "<dual_sideband>".$get["dsb"]."</dual_sideband>";
      $xml .=   "<resolution units='bytes'>".$get["resolution"]."</resolution>";
      $xml .= "</pfb_parameters>";

      $xml .= "<observation_parameters>";
      $xml .=   "<observer>".$get["observer"]."</observer>";
      $xml .=   "<tobs>".$get["tobs"]."</tobs>";
      $xml .= "</observation_parameters>";

      $xml .= "<boresight_parameters>";
      $xml .=   "<project_id>".$get["project_id"]."</project_id>";
      $xml .=   "<name epoch='J2000'>".$get["0_source"]."</name>";
      $xml .=   "<ra units='hh:mm:ss'>".$get["0_ra"]."</ra>";
      $xml .=   "<dec units='hh:mm:ss'>".$get["0_dec"]."</dec>";
      $xml .=   "<rfi_mitigation>".$this->checkbox("rfi_mitigation", $get)."</rfi_mitigation>";
      $xml .=   "<antenna_weights>".$this->checkbox("antenna_weights", $get)."</antenna_weights>";
      $xml .=   "<delay_tracking>".$this->checkbox("delay_tracking", $get)."</delay_tracking>";
      $xml .=   "<processing_file>".$get["0_processing_file"]."</processing_file>";
      $xml .= "</boresight_parameters>";
   
      if (array_key_exists("correlation_enabled", $get))
      {
        $xml .= "<correlation_parameters>";
        $xml .=   "<mode>CORR</mode>";
        $xml .=   "<project_id>".$get["correlation_project_id"]."</project_id>";
        $xml .=   "<type>".$get["correlation_type"]."</type>";
        $xml .=   "<processing_file>".$get["corr_processing_file"]."</processing_file>";
        $xml .=   "<dump_time units='seconds'>".$get["correlation_dump_time"]."</dump_time>";
        $xml .= "</correlation_parameters>";
      }

      $ibeam = 1;
      for ($i=1; $i<=$this->ntiedbeams; $i++)
      {
        if (array_key_exists($i."_enabled", $get))
        {
          $key = "tied_beam_".($i-1)."_parameters";
          $xml .= "<".$key.">";
          $xml .=   "<mode>PSR</mode>";
          $xml .=   "<project_id>".$get[$i."_project_id"]."</project_id>";
          $xml .=   "<processing_file>".$get[$i."_processing_file"]."</processing_file>";
          $xml .=   "<name epoch='J2000'>".$get[$i."_source"]."</name>";
          $xml .=   "<ra units='hh:mm:ss'>".$get[$i."_ra"]."</ra>";
          $xml .=   "<dec units='hh:mm:ss'>".$get[$i."_dec"]."</dec>";
          $xml .= "</".$key.">";
          $ibeam++;
        }
      }

      if (array_key_exists("fan_beams_enabled", $get))
      {
        $xml .= "<fan_beams_parameters>";
        $xml .=   "<mode>PSR</mode>";
        $xml .=   "<project_id>".$get["fan_beams_project_id"]."</project_id>";
        $xml .=   "<nbeams>".$get["nbeams"]."</nbeams>";
        $xml .=   "<beam_spacing units='degrees'>".$get["beam_spacing"]."</beam_spacing>";
        $xml .= "</fan_beams_parameters>";
      }

      if (array_key_exists("mod_beams_enabled", $get))
      {
        $xml .= "<mod_beams_parameters>";
        $xml .=   "<mode>PSR</mode>";
        $xml .=   "<project_id>".$get["mod_beams_project_id"]."</project_id>";
        $xml .= "</mod_beams_parameters>";
      }

      $xml .= "<furbies>";
      $xml .=   "<num_furbies>".$get["num_furbies"]."</num_furbies>";
      if ($get["num_furbies"] > 0)
      {
        $xml .=   "<furby_ids>".$get["furby_ids"]."</furby_ids>";
        $xml .=   "<furby_beams>".$get["furby_beams"]."</furby_beams>";
        $xml .=   "<furby_tstamps>".$get["furby_tstamps"]."</furby_tstamps>";
      }
      $xml .= "</furbies>";

    }
    else if ($get["command"] == "start")
    {
      $xml .= "<command>start</command>";
    }
    else if ($get["command"] == "query")
    {
      $xml .= "<command>query</command>";
    }
    else
    {
      $xml .= "<command>ignore</command>";
    }

    $xml .= "</mpsr_tmc_message>\r\n";

    $transmit = true;
    if ($transmit) 
    {
      echo "<html>\n";
      echo "<head>\n";
      for ($i=0; $i<count($this->css); $i++)
        echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
      echo "</head>\n";
      echo "<body>\n";

      echo "<table border=1 width='100%'>\n";
      echo "  <tr>\n";
      echo "    <th>Command</th>\n";
      echo "    <th>Response</th>\n";
      echo "  </tr>\n";
      
      list ($sock,$message) = openSocket($host,$port,2);
      if (!($sock)) {
        $this->printTR("Error: opening socket to TMC interface [".$host.":".$port."]: ".$message, "");
        $this->printTF();
        $this->printFooter();
        return;
      }

      $html_cmd = str_replace("<", "[", $xml);
      $html_cmd = str_replace(">", "]", $html_cmd);
      $html_cmd = str_replace("\n", "<br/>", $html_cmd);

      $xml = str_replace("\n", "", $xml);

      $this->printTR ("Sending", $html_cmd);
      socketWrite ($sock, $xml."\r\n");

      $xml = "";
      list ($result, $xml) = socketRead ($sock);

      $html_response = str_replace("><", "]\n[", $xml);
      $html_response = str_replace("<", "[", $html_response);
      $html_response = str_replace(">", "]", $html_response);
      $html_response = str_replace("\n", "<br/>", $html_response);

      $this->printTR ("Received", $html_response);
      $this->printTF();
      $this->printFooter ();

      socket_close($sock);
      echo "</table>\n";
      echo "</body>\n";
      echo "</html>\n";
      return;
    }
    else
    {
      header("Content-type: text/xml");
      echo $xml."\n";
    }
  }

  function printTR($left, $right) {
    echo " <tr>\n";
    echo "  <td>".$left."</td>\n";
    echo "  <td>".$right."</td>\n";
    echo " </tr>\n";
    echo '<script type="text/javascript">self.scrollBy(0,100);</script>';
    flush();
  }

  function printFooter() {
    echo "</body>\n";
    echo "</html>\n";
  }

  function printTF() {
    echo "</table>\n";
  }

  function checkbox($key, $array)
  {
    if (array_key_exists($key, $array))
      return "true";
    else
      return "false";
  }
}

if (isset($_GET["command"])) {
  $obj = new tmc_simulator();
  $obj->printTMCResponse($_GET);
} else {
  handleDirect("tmc_simulator");
}
