<?PHP
include("definitions_i.php");
include("functions_i.php");
include("bpsr_functions_i.php");

# Get the system configuration (dada.cfg)
$cfg = getConfigFile(SYS_CONFIG,TRUE);
$conf = getConfigFile(DADA_CONFIG,TRUE);
$spec = getConfigFile(DADA_SPECIFICATION, TRUE);

$archives_dir = $cfg["SERVER_ARCHIVE_DIR"];
$results_dir = $cfg["SERVER_RESULTS_DIR"];
$apsr01_dir = "/nfs/apsr01/bpsr/archives";

if (isset($_GET["current_day"])) {
  $current_day = $_GET["current_day"];
} else {
  $current_day = "all";
}

if (isset($_GET["type"])) {
  $type = $_GET["type"];
} else {
  $type = "all";
}

if (isset($_GET["show_test"])) {
  $show_test = $_GET["show_test"];
} else {
  $show_test = "yes";
}

if (isset($_GET["show_survey"])) {
  $show_survey = $_GET["show_survey"];
} else {
  $show_survey = "yes";
}

if (isset($_GET["show_finalized"])) {
  $show_finalized = $_GET["show_finalized"];
} else {
  $show_finalized = "yes";
}

if (isset($_GET["show_archived"])) {
  $show_archived = $_GET["show_archived"];
} else {
  $show_archived = "yes";
}

if (isset($_GET["show_errors_only"])) {
  $show_errors_only = $_GET["show_errors_only"];
} else {
  $show_errors_only = "no";
}

if (isset($_GET["action"])) {
  $action = $_GET["action"];
} else {
  $action = "none";
}


echo "<html>\n";
$title = "BPSR | Archives Status";
include("header_i.php");
echo "<body>\n";

$text = "Mega Archives Page";
include("banner.php");

echo "<center>\n";


# Get the disk listing information
$num_swin_dirs = $cfg["NUM_SWIN_DIRS"];
$num_parkes_dirs = $cfg["NUM_PARKES_DIRS"];
$num_swin_fold_dirs = $cfg["NUM_SWIN_FOLD_DIRS"];
$num_parkes_fold_dirs = $cfg["NUM_PARKES_FOLD_DIRS"];
$num_parkes_dirs = 4;

$swin_disks = array();
$swin_users = array();
$swin_hosts = array();
$swin_fold_disks = array();
$swin_fold_users = array();
$swin_fold_hosts = array();
$parkes_disks = array();
$parkes_users = array();
$parkes_hosts = array();
$parkes_fold_disks = array();
$parkes_fold_users = array();
$parkes_fold_hosts = array();

for ($i=0; $i<$num_swin_dirs; $i++) {
  $array = split(":",$cfg["SWIN_DIR_".$i]);
  array_push($swin_users, $array[0]);
  array_push($swin_hosts, $array[1]);
  array_push($swin_disks, $array[2]);
}
for ($i=0; $i<$num_swin_fold_dirs; $i++) {
  $array = split(":",$cfg["SWIN_FOLD_DIR_".$i]);
  array_push($swin_fold_users, $array[0]);
  array_push($swin_fold_hosts, $array[1]);
  array_push($swin_fold_disks, $array[2]);
}

for ($i=0; $i<$num_parkes_dirs; $i++) {
  $array = split(":",$cfg["PARKES_DIR_".$i]);
  array_push($parkes_users, $array[0]);
  array_push($parkes_hosts, $array[1]);
  array_push($parkes_disks, $array[2]);
}
for ($i=0; $i<$num_parkes_fold_dirs; $i++) {
  $array = split(":",$cfg["PARKES_FOLD_DIR_".$i]);
  array_push($parkes_fold_users, $array[0]);
  array_push($parkes_fold_hosts, $array[1]);
  array_push($parkes_fold_disks, $array[2]);
}

# Get the bookkeeping db information
$array = split(":",$cfg["SWIN_DB_DIR"]);
$swin_db_user = $array[0];
$swin_db_host = $array[1];
$swin_db_dir  = $array[2];
$swin_db_file = $array[2]."/files.db";

$array = split(":",$cfg["PARKES_DB_DIR"]);
$parkes_db_user = $array[0];
$parkes_db_host = $array[1];
$parkes_db_dir  = $array[2];
$parkes_db_file = $array[2]."/files.db";

echo "<span id=\"progress\">\n";
# Get the listing of archives on tape @ swinburne
echo "Retreiving files.db from swin<br>\n";
flush();
$cmd = "ssh -l ".$swin_db_user." ".$swin_db_host." \"cat ".$swin_db_file." | grep '01 HRA' | sort\" | awk -F/ '{print $1}'";
$swin_on_tape = array();
$lastline = exec($cmd, $swin_on_tape, $return_var);

# Get the listing of archives on tape @ parkes
echo "Retreiving files.db from parkes<br>\n";
flush();
$cmd = "ssh -l ".$parkes_db_user." ".$parkes_db_host." \"cat ".$parkes_db_file." | grep '01 HRE' | sort\" | awk -F/ '{print $1}'";
$parkes_on_tape = array();
$lastline = exec($cmd, $parkes_on_tape, $return_var);

# Get the listing of archives in from_parkes @ swinburne
$swin_from_parkes = array();
for ($i=0; $i<$num_swin_dirs; $i++) {
  echo "Retreiving listing of ".$swin_users[$i]." ".$swin_hosts[$i]." ".$swin_disks[$i]."<br>\n";
  flush();
  $cmd = "ssh -l ".$swin_users[$i]." ".$swin_hosts[$i]." 'ls ".$swin_disks[$i]." > /dev/nulll; ls ".$swin_disks[$i]." | grep 20'";
  $array = array();
  $lastline = exec($cmd, $array, $return_var);
  $swin_from_parkes = array_merge($swin_from_parkes, $array);
}

# Get the listing of archives with xfer.complete in from_parkes @ swinburne
$swin_from_parkes_complete = array();
for ($i=0; $i<$num_swin_dirs; $i++) {
  echo "Retreiving listing of ".$swin_users[$i]." ".$swin_hosts[$i]." ".$swin_disks[$i]."/20*/xfer.complete <br>\n";
  flush();
  $cmd = "ssh -l ".$swin_users[$i]." ".$swin_hosts[$i]." 'ls ".$swin_disks[$i]."/20*/xfer.complete' | awk -F / '{print \$6}'";
  $array = array();
  $lastline = exec($cmd, $array, $return_var);
  $swin_from_parkes_complete = array_merge($swin_from_parkes_complete, $array);
}

# Get the listing of archives in from_pakes @ parkes
$parkes_from_parkes = array();
for ($i=0; $i<$num_parkes_dirs; $i++) {
  echo "Retreiving listing of ".$parkes_users[$i]." ".$parkes_hosts[$i]." ".$parkes_disks[$i]."/<br>\n";
  flush();
  $cmd = "ssh -l ".$parkes_users[$i]." ".$parkes_hosts[$i]." 'ls ".$parkes_disks[$i]." > /dev/nulll; ls ".$parkes_disks[$i]." | grep 20'";
  $array = array();
  $lastline = exec($cmd, $array, $return_var);
  $parkes_from_parkes = array_merge($parkes_from_parkes, $array);
}

# Get the listing of archives with xfer.complete in from_pakes @ parkes
$parkes_from_parkes_complete = array();
for ($i=0; $i<$num_parkes_dirs; $i++) {  
  echo "Retreiving listing of ".$parkes_users[$i]." ".$parkes_hosts[$i]." ".$parkes_disks[$i]."/20*/xfer.complete <br>\n";
  flush();
  $cmd = "ssh -l ".$parkes_users[$i]." ".$parkes_hosts[$i]." 'ls ".$parkes_disks[$i]."/20*/xfer.complete' | awk -F / '{print \$6}'";
  $array = array();
  $lastline = exec($cmd, $array, $return_var);
  $parkes_from_parkes_complete = array_merge($parkes_from_parkes_complete, $array);
}

# Get the listing of archives in on_tape disk @ swin
$swin_on_disk = array();
for ($i=0; $i<$num_swin_dirs; $i++) {
  echo "Retreiving listing of ".$swin_users[$i]." ".$swin_hosts[$i]." ".$swin_disks[$i]."/../on_tape/<br>\n";
  flush();
  $cmd = "ssh -l ".$swin_users[$i]." ".$swin_hosts[$i]." 'ls -1 ".$swin_disks[$i]."/../on_tape | grep 20'";
  $array = array();
  $lastline = exec($cmd, $array, $return_var);
  $swin_on_disk = array_merge($swin_on_disk, $array);
}

# Get the listing of archives in on_tape disk @ parkes
$parkes_on_disk = array();
for ($i=0; $i<$num_parkes_dirs; $i++) {
  echo "Retreiving listing of ".$parkes_users[$i]." ".$parkes_hosts[$i]." ".$parkes_disks[$i]."/../on_tape/<br>\n";
  flush();
  $cmd = "ssh -l ".$parkes_users[$i]. " ".$parkes_hosts[$i]." 'ls ".$parkes_disks[$i]."/../on_tape | grep 20'";
  $array = array();
  $lastline = exec($cmd, $array, $return_var);
  $parkes_on_disk = array_merge($parkes_on_disk, $array);
}

# Get the listing of archives in from_parkes_pulsars @ swin
$swin_from_parkes_pulsars = array();
for ($i=0; $i<$num_swin_fold_dirs; $i++) {
  echo "Retreiving listing of ".$swin_fold_users[$i]." ".$swin_fold_hosts[$i]." ".$swin_fold_disks[$i]."/<br>\n";
  flush();
  $cmd = "ssh -l ".$swin_fold_users[$i]." ".$swin_fold_hosts[$i]." 'ls -1 ".$swin_fold_disks[$i]." | grep 20'";
  $array = array();
  $lastline = exec($cmd, $array, $return_var);
  $swin_from_parkes_pulsars = array_merge($swin_from_parkes_pulsars, $array);
}

# Get the listing of archives in from_parkes_pulsars @ parkes
$parkes_from_parkes_pulsars = array();
for ($i=0; $i<$num_parkes_fold_dirs; $i++) {
  echo "Retreiving listing of ".$parkes_fold_users[$i]." ".$parkes_fold_hosts[$i]." ".$parkes_fold_disks[$i]."/<br>\n";
  flush();
  $cmd = "ssh -l ".$parkes_fold_users[$i]." ".$parkes_fold_hosts[$i]." 'ls -1 ".$parkes_fold_disks[$i]." | grep 20'";
  $array = array();
  $lastline = exec($cmd, $array, $return_var);
  $parkes_from_parkes_pulsars = array_merge($parkes_from_parkes_pulsars, $array);
}


echo "Retreiving local archives listing<br>\n";
flush();
$archives = getArchivesArray($archives_dir, $results_dir, $apsr01_dir);

$keys = array_keys($archives);
$days = array("all");

# Now make a drop down list of each day
for ($i=0; $i<count($keys); $i++) {
  $day = substr($keys[$i],0,10);
  if (! in_array($day, $days) ) {
    array_push($days, $day);
  }
}

$num_results = count($keys);

echo "DONE, formatting page<br>\n";
echo "</span>";

if ($action != "delete") {

?>
<script type="text/javascript">

  function mySubmit(value) {
    if (value == "delete_scripts") {
      document.getElementById("action").value = "delete";
    } else {
      document.getElementById("action").value = "none";
    }
    document.day_select.submit();
  }


</script>
<table cellpadding=10>
  <tr>
    <td valign=top align=center>
      <form name="day_select" action="archives.php" method="GET">
      <input type="hidden" id="action" name="action" value="none"></input>

      <table cellpadding=5>
        <tr><td>

          Select a day:
          <select name="current_day" onChange="form.submit()">
          <?
          for ($i=0; $i<count($days); $i++) {
            if ($current_day == $days[$i]) {
              echo "<option selected>".$days[$i]."</option>\n";
            } else {
              echo "<option>".$days[$i]."</option>\n";
            }
          }
          ?>
          </select>

        </td></tr>

        <tr><td>
        Show Test Obs? <input type="radio" name="show_test" value="yes"<? if($show_test == "yes") echo " checked"?>>Yes</input>
                                <input type="radio" name="show_test" value="no"<? if($show_test == "no") echo " checked"?>>No</input>
        </td></tr>

        <tr><td>
        Show Survey Obs? <input type="radio" name="show_survey" value="yes"<? if($show_survey == "yes") echo " checked"?>>Yes</input>
                                <input type="radio" name="show_survey" value="no"<? if($show_survey == "no") echo " checked"?>>No</input>
        </td></tr>

        <tr><td>
        Show Finalized Obs? <input type="radio" name="show_finalized" value="yes"<? if($show_finalized == "yes") echo " checked"?>>Yes</input>
                                <input type="radio" name="show_finalized" value="no"<? if($show_finalized == "no") echo " checked"?>>No</input>
        </td></tr> 

        <tr><td>
        Show Fully Archived? <input type="radio" name="show_archived" value="yes"<? if($show_archived == "yes") echo " checked"?>>Yes</input>
                                <input type="radio" name="show_archived" value="no"<? if($show_archived == "no") echo " checked"?>>No</input>
        </td></tr>

        <tr><td>
        Show Errors Only? <input type="radio" name="show_errors_only" value="yes"<? if($show_errors_only == "yes") echo " checked"?>>Yes</input>
                                <input type="radio" name="show_errors_only" value="no"<? if($show_errors_only == "no") echo " checked"?>>No</input>
        </td></tr> 

        <tr><td colspan=2 align=center><input type="button" value="Submit" onClick="mySubmit('none')"></input></td></tr>
        <tr><td colspan=2 align=center><input type="button" value="Generate Deletion Scripts" onClick="mySubmit('delete_scripts')"></input></td></tr>

      </table> 
      </form>

      <table class="datatable">
        <tr><th colspan=3>Legend</th></tr>
        <tr><th>Key</th><th>Value</th><th>Meaning</th></tr>
        <tr><td>Status</td><td>ARCHIVED</td><td>Backed up to tape at both Parkes &amp; Swinburne</td></tr>
        <tr><td>Status</td><td>FINALIZED</td><td>ARCHIVED, and has been deleted from apsr local disks</td></tr>
        <tr><td>Loc</td><td>A</td><td>Existed in /nfs/archives/bpsr</td></tr>
        <tr><td>Loc</td><td>R</td><td>Existed in /nfs/results/bpsr</td></tr>
        <tr><td>Loc</td><td>L</td><td>Existed in /nfs/apsr01/bpsr/archives/</td></tr>
        <tr><td>Sent</td><td><img src="/images/green_light.png"></td><td>Obs marked as sent.to.&lt;loc&gt;</td></tr>
        <tr><td>Sent</td><td><img src="/images/yellow_light.png"></td><td>Obs ready to send</td></tr>
        <tr><td>Sent</td><td><img src="/images/red_light.png"></td><td>Obs marked as error.to.&lt;loc&gt;</td></tr>
        <tr><td>Tape</td><td><img src="/images/green_light.png"></td><td>Obs written to tape</td></tr>
        <tr><td>Tape</td><td><img src="/images/yellow_light.png"></td><td>Obs received, but not yet written</td></tr>
        <tr><td>Tape</td><td><img src="/images/red_light.png"></td><td>Obs received, not xfer.complete</td></tr>
        <tr><td>Tape</td><td></td><td>Obs not yet received</td></tr>
        <tr><td>Disk</td><td><img src="/images/green_light.png"></td><td>Obs exists in on_tape dir</td></tr>
        <tr><td>Disk</td><td></td><td>Obs does not exist in on_tape dir</td></tr>
      </table>

    </td>
    <td>

<table border=1 cellpadding=3 class="datatable">
<tr>
  <th colspan=6></th>
  <th colspan=4>obs.</th>
  <th colspan=3>Swin</th>
  <th colspan=3>Parkes</th>
  <th colspan=3>Loc</th>
</tr>
<tr>
  <th>Num</th>
  <th>Source</th>
  <th>UTC Start</th>
  <th>Status</th>
  <th>Swin status</th>
  <th>Parkes status</th>
  <th>info</th>
  <th>fin</th>
  <th>del</th>
  <th>prob</th>
  <th>Sent</th>
  <th>Tape</th>
  <th>Disk</th>
  <th>Sent</th>
  <th>Tape</th>
  <th>Disk</th>
  <th>A</th>
  <th>R</th>
  <th>L</th>
</tr>
<?

} else {

  # Sanity check grepping script for files.db
  $files_db_script = "/home/dada/scripts/bpsr_files_db_checker.csh";
  if (file_exists($files_db_script)) {
    unlink($files_db_script);
  }                             
  system('echo "#!/bin/tcsh" > '.$files_db_script);
  chmod($files_db_script, 0755);

  # Script for deletion of archived observations on apsr 1-13
  $bpsr_delete_script = "/home/dada/scripts/bpsr_archived_cleaner.csh";
  if (file_exists($bpsr_delete_script)) {
    unlink($bpsr_delete_script);
  }
  system('echo "#!/bin/tcsh" > '.$bpsr_delete_script);
  chmod($bpsr_delete_script, 0755);

  # Script for deletion of archived observations on apsr 14-17
  $parkes_delete_script = "/home/dada/scripts/parkes_on_tape_master_cleaner.csh";
  if (file_exists($parkes_delete_script)) {
    unlink($parkes_delete_script);
  }
  system('echo "#!/bin/tcsh" > '.$parkes_delete_script);
  chmod($parkes_delete_script, 0755);

?>

  Creating deletion scripts (not running them!) in dada@srv0:~/scripts/

<?

}

$keys = array_keys($archives);
$num = 1;

for ($i=0; $i < count($keys); $i++) {

  $show_this_obs = 1;

  if (($current_day == "all") || (strpos($keys[$i], $current_day) !== FALSE)) {

    $status_string = "";
    $status_swin = "";
    $status_parkes = "";

    $header = array();
    $header_file = $archives[$keys[$i]]["header"];

    $data = $archives[$keys[$i]];

    if (file_exists($header_file)) {
      $header = getConfigFile($header_file,TRUE);
    }

    $nbeams = $data["nbeams"];

    $freq_keys = array_keys($archives[$keys[$i]]);
    $url = "/bpsr/result.php?utc_start=".$keys[$i]."&imagetype=bp";
    $mousein = "onmouseover=\"Tip('<img src=\'".$images[$keys[$i]][0]["bp_400x300"]."\' width=400 height=300>')\"";
    $mouseout = "onmouseout=\"UnTip()\"";

    $of  = ($archives[$keys[$i]]["obs.finished"])*2;
    $oi  = ($archives[$keys[$i]]["obs.info"])*2;
    $od  = ($archives[$keys[$i]]["obs.deleted"])*2;
    $op  = ($archives[$keys[$i]]["obs.problem"])-1;
    $sts = $archives[$keys[$i]]["sent.to.swin"];
    $stp = $archives[$keys[$i]]["sent.to.parkes"];
    $ets = $archives[$keys[$i]]["error.to.swin"];
    $etp = $archives[$keys[$i]]["error.to.parkes"];
    $ots = in_array($keys[$i], $swin_on_tape);
    $otp = in_array($keys[$i], $parkes_on_tape);
    $fps = in_array($keys[$i], $swin_from_parkes);
    $fpsc = in_array($keys[$i], $swin_from_parkes_complete);
    $fpp = in_array($keys[$i], $parkes_from_parkes);
    $fppc = in_array($keys[$i], $parkes_from_parkes_complete);
    $sod = in_array($keys[$i], $swin_on_disk);
    $pod = in_array($keys[$i], $parkes_on_disk);
    $sfpp = in_array($keys[$i], $swin_from_parkes_pulsars);
    $pfpp = in_array($keys[$i], $parkes_from_parkes_pulsars);
    $in_a = $archives[$keys[$i]]["in.nfs.archives"];
    $in_r = $archives[$keys[$i]]["in.nfs.results"];
    $in_l = $archives[$keys[$i]]["in.local.archives"];

    if (!$of) {
      $status_string = "NO obs.finished"; 
    }
    if (!$oi) {
      if ($status_string != "") {
        $status_string .= "<BR>";
      }
      $status_string .= "NO obs.info"; 
    }

    if (!$in_r) {
      if ($status_string != "") {
        $status_string .= "<BR>";
      }
      $status_string .= "NO results dir";
    }

    if ($sts) {
      $to_s = 2;  # sent
    } else if ($ets) {
      $to_s = 0;  # error
      $status_swin = "Send Error";
    } else {
      $to_s = 1;  # pending
    }

    if ($stp) {
      $to_p = 2;  # sent
    } else if ($etp) {
      $to_p = 0;  # error
      $status_parkes = "Send Error";
    } else {
      $to_p = 1;  # pending
    }

    if ($ots) {
      $on_s = 2;  # on a tape
    } else if ($fps) {
      if ($fpsc) {
        $on_s = 1;  # in from_parkes && xfer.complete
      } else {
        $on_s = 0;  # in from_parkes && ! xfer.complete
      }
    } else {
      $ots = 0;
      $on_s = -1;  # not on tape or in from_parkes
    }

    if ($otp) {
      $on_p = 2;  # on a tape
    } else if ($fpp) {
      if ($fppc) {
        $on_p = 1;  # in from_parkes && xfer.complete
      } else {
        $on_p = 0;  # in from_parkes && ! xfer.complete
      }
    } else {
      $otp = 0;
      $on_p = -1;  # not on tape or in from_parkes
    }

    if ($sod) {
      $sod = 2;
    } else {
      $sod = -1;
    }
    if ($pod) {
      $pod = 2;
    } else {
      $pod = -1;
    }

    # Set the status strings to something useful
    if ($of==2 && $to_s==2 && $to_p==2 && $on_s==2 && $on_p==2 && $od==2) {
      $status_string = "FINALIZED";
    }
    if ($of==2 && $to_s==2 && $to_p==2 && $on_s==2 && $on_p==2 && $od<0 && $in_l==2) {
      $status_string = "ARCHIVED";
    }
    if ($to_s==2 && $on_s==2)
      $status_swin = "Archived";

    if ($to_p==2 && $on_p==2)
      $status_parkes = "Archived";

    if ($to_s==1 && $on_s==0)
      $status_swin = "Sending";

    if ($to_p==1 && $on_p==0)
      $status_parkes = "Sending";

    if ($to_s==1 && $on_s==-1 && $of==2)
      $status_swin = "Send";

    if ($to_p==1 && $on_p==-1 && $of==2)
      $status_parkes = "Send";
  
    if ($to_s==2 && $on_s==1)
      $status_swin = "Archive";

    if ($to_p==2 && $on_p==1)
      $status_parkes = "Archive";

    # some consistency checking
    if (($to_s==2) && ($on_s<0) && ($ots!=2)) {
      $status_swin = "Flag Error";
    } 
    if ($to_p==2 && $on_p<0 && $otp!=2) {
      $status_parkes = "Flag Error";
    }

    if (($status_string == "FINALIZED") && ($show_finalized == "no")) {
      $show_this_obs = 0;
    }
    if (($status_string == "ARCHIVED") && ($show_archived == "no")) {
      $show_this_obs = 0;
    }

    $first_char = substr($header["SOURCE"],0,1);
    
    if (($first_char != "G") && ($show_test == "no")) {
      $show_this_obs = 0;
    }

    if (($first_char == "G") && ($show_survey == "no")) {
      $show_this_obs = 0;
    }

    if ($show_errors_only == "yes") {
      # Show if any error state and the obs wasn't previously excluded
      if ((($of == 0) || ($oi == 0) || ($to_s == 0) || ($on_s == 0) || 
          ($to_p == 0) || ($on_p == 0) || ($in_a == 0) || 
          ($in_r == 0)) && ($show_this_obs)) {
        $show_this_obs = 1;
      } else {
        $show_this_obs = 0;
      }
    } else {

    }

    if ($action != "delete") {

      if ($show_this_obs == 1) {

        echo "  <tr class=\"new\">\n";

        /* SOURCE */
        echo "    <td>".$num."</td>\n";

        /* SOURCE */
        echo "    <td>".$header["SOURCE"]."</td>\n";

        /* UTC_START */
        echo "    <td>".$keys[$i]."</td>\n";

        /* OVERALL STATUS*/
        if ($status_string == "FINALIZED") {
          $color = " bgcolor='lightgreen'";
        } else if ($status_string == "ARCHIVED") {
          $color = " bgcolor='#FFFF99;'";
        } else {
          $color = "";
        }
        echo "    <td align=center".$color.">".$status_string."</td>\n";

        /* SWIN STATUS */
        if ($status_swin == "Archived") {
          $color = " bgcolor='lightgreen'";
        } else if (($status_swin == "Send") || ($status_swin == "Archive")) {
          $color = " bgcolor='#FFFF99;'";
        } else if (($status_swin == "Send Error") || ($status_swin == "Flag Error")) {
          $color = " bgcolor='#FF3366;'";
        } else  {
          $color = "";
        }
        echo "    <td align=center".$color.">".$status_swin."</td>\n";

        if ($status_parkes == "Archived") {
          $color = " bgcolor='lightgreen'";
        } else if (($status_parkes == "Send") || ($status_parkes == "Archive")) {
          $color = " bgcolor='#FFFF99;'";
        } else if (($status_parkes == "Send Error") || ($status_parkes == "Flag Error")) {
          $color = " bgcolor='#FF3366;'";
        } else  {
          $color = "";
        } 
        echo "    <td align=center".$color.">".$status_parkes."</td>\n";

        echo "    <td align=center>".statusLight($oi)."</td>\n";
        echo "    <td align=center>".statusLight($of)."</td>\n";
        echo "    <td align=center>".statusLight($od)."</td>\n";
        echo "    <td align=center>".statusLight($op)."</td>\n";
        echo "    <td align=center>".statusLight($to_s)."</td>\n";
        echo "    <td align=center>".statusLight($on_s)."</td>\n";
        echo "    <td align=center>".statusLight($sod)."</td>\n";
        echo "    <td align=center>".statusLight($to_p)."</td>\n";
        echo "    <td align=center>".statusLight($on_p)."</td>\n";
        echo "    <td align=center>".statusLight($pod)."</td>\n";
        echo "    <td align=center>".statusLight($in_a)."</td>\n";
        echo "    <td align=center>".statusLight($in_r)."</td>\n";
        echo "    <td align=center>".statusLight($in_l)."</td>\n";
        echo "  </tr>\n";
        $num++;
      }

    /* generate the scripts for deleting */
    } else {
 
      if ($status_string == "ARCHIVED") {
        # For deletion on apsr 1-> 13
        system("echo '~dada/scripts/bpsr_delete_archived_obs.csh ".$keys[$i]." nice' >> ".$bpsr_delete_script);
        # For deletion on apsr 14 -> 17
        system("echo '~dada/scripts/parkes_delete_on_tape_obs.csh ".$keys[$i]." nice' >> ".$parkes_delete_script);
        # For deletion on apsr 14 -> 17
        system("echo 'set nbeams = `grep \"^".$keys[$i]."\" files.db | wc -l`' >> ".$files_db_script);
        system("echo 'echo \"$keys[$i]: \$nbeams\"' >> ".$files_db_script);
      }
    }
  }

}
?>
</table>

</td></tr>
</table>

<script type="text/javascript">
  document.getElementById("progress").innerHTML = "";
</script>

</body>
</html>

<?

function getArchivesArray($a_dir, $r_dir, $n_dir) {

  $results = array();

  $obs_a = getSubDirs($a_dir);
  $obs_r = getSubDirs($r_dir);
  $obs_n = getSubDirs($n_dir);

  $obs = array_merge($obs_a, $obs_r, $obs_n);

  ksort($obs);

  /* For each observation get a list of frequency channels present */   
  for ($i=0; $i<count($obs); $i++) {

    $results[$obs[$i]] = array();

    $results[$obs[$i]]["header"] = "none";

    if (file_exists($r_dir."/".$obs[$i]."/obs.info")) {
      $results[$obs[$i]]["header"] = $r_dir."/".$obs[$i]."/obs.info";
    }
    if (file_exists($a_dir."/".$obs[$i]."/obs.info")) {
      $results[$obs[$i]]["header"] = $a_dir."/".$obs[$i]."/obs.info";
    }
    if ($results[$obs[$i]]["header"] == "none") {
      if (file_exists($a_dir."/".$obs[$i]."/01/obs.start")) {
        $results[$obs[$i]]["header"] = $a_dir."/".$obs[$i]."/01/obs.start";
      }
    }
    if (file_exists($a_dir."/".$obs[$i]."/sent.to.parkes")) {
      $results[$obs[$i]]["sent.to.parkes"] = 1;
    } else {
      $results[$obs[$i]]["sent.to.parkes"] = 0;
    }
    if (file_exists($a_dir."/".$obs[$i]."/sent.to.swin")) {
      $results[$obs[$i]]["sent.to.swin"] = 1;
    } else {
      $results[$obs[$i]]["sent.to.swin"] = 0;
    }
    if (file_exists($r_dir."/".$obs[$i]."/obs.finished")) {
      $results[$obs[$i]]["obs.finished"] = 1;
    } else {
      $results[$obs[$i]]["obs.finished"] = 0;
    }
    if (file_exists($a_dir."/".$obs[$i]."/obs.problem")) {
      $results[$obs[$i]]["obs.problem"] = 1;
    } else {
      $results[$obs[$i]]["obs.problem"] = 0;
    }
    if (file_exists($r_dir."/".$obs[$i]."/obs.info")) {
      $results[$obs[$i]]["obs.info"] = 1;
    } else {
      $results[$obs[$i]]["obs.info"] = 0;
    }
    if (file_exists($a_dir."/".$obs[$i]."/error.to.swin")) {
      $results[$obs[$i]]["error.to.swin"] = 1;
    } else {
      $results[$obs[$i]]["error.to.swin"] = 0;
    }    
    if (file_exists($a_dir."/".$obs[$i]."/error.to.parkes")) {
      $results[$obs[$i]]["error.to.parkes"] = 1;
    } else {
      $results[$obs[$i]]["error.to.parkes"] = 0;
    }
    $results[$obs[$i]]["in.nfs.archives"] = 0;
    if (in_array($obs[$i],$obs_a)) {
      $results[$obs[$i]]["in.nfs.archives"] = 2;
    }
    $results[$obs[$i]]["in.nfs.results"] = 0;
    if (in_array($obs[$i],$obs_r)) {
      $results[$obs[$i]]["in.nfs.results"] = 2;
    }
    $results[$obs[$i]]["obs.deleted"] = -1;
    if (file_exists($a_dir."/".$obs[$i]."/obs.deleted")) {
      $results[$obs[$i]]["obs.deleted"] = 1;
    }
    $results[$obs[$i]]["in.local.archives"] = 0;
    if (file_exists($n_dir."/".$obs[$i]."/01/aux.tar")) {
      $results[$obs[$i]]["in.local.archives"] = 2;
    }
  } 
  return $results;
}

function statusLight($on) {

  if ($on == 2) {
    return "<img src='/images/green_light.png'>";
  } else if ($on == 1) {
    return "<img src='/images/yellow_light.png'>";
  } else if ($on == 0) {
    return "<img src='/images/red_light.png'>";
  } else {
    return "";
  }
}

function getRemoteListing() {

}

?>
