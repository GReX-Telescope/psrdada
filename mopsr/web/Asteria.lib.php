<?php
function print_summary($pdo) {
  echo "<table>";
  $q = 'SELECT date FROM Updates';

  $stmt = $pdo -> query($q);
  if (!$stmt) {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }

  $updated = $stmt ->fetch();

  $q = 'SELECT utc FROM UTCs ORDER BY utc LIMIT 1';
  $stmt = $pdo -> query($q);
  if (!$stmt) {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }

  $since = $stmt ->fetch();

  $q = 'SELECT COUNT(*) FROM TB_Obs';
  $stmt = $pdo -> query($q);
  if (!$stmt) {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }

  $count = $stmt ->fetch();

  echo "<tr><td>Data since ".substr($since[0], 0, 10)."<td><tr>\n";
  echo "<tr><td>".$count[0]." observations</td></tr>\n";
  echo "<tr><td>Updated at:<br><span class=best_snr>".$updated[0]."</span></td></tr>\n";
?>
      <tr>
        <td colspan=2><a href="/mopsr/Asteria_500.php?single=true">Timing Programme Pulsars</a></td>
      </tr>
      <tr>
        <td colspan=2><a href="/mopsr/Asteria_500_when.php?single=true">Time since last observation</a></td>
      </tr>
      <tr>
        <td colspan=2><a href="/mopsr/Asteria_bests.php?single=true">Best per pulsar</a></td>
      </tr>
      <tr><td><hr/></td><tr>
<?php
  $q = 'SELECT COUNT(*) FROM UTCs JOIN Infos ON UTCs.id = Infos.utc_id WHERE TIMESTAMPDIFF(MINUTE, utc_ts, UTC_TIMESTAMP()) < 24*60';
  $stmt = $pdo -> query ($q);
  if (!$stmt) {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }

  $count_24h = $stmt ->fetch();

  $q = 'SELECT COUNT(*) FROM UTCs JOIN TB_Obs ON UTCs.id = TB_Obs.utc_id AND UTCs.id = TB_Obs.utc_id WHERE TIMESTAMPDIFF(MINUTE, UTCs.utc_ts, UTC_TIMESTAMP()) < 24*60';
  $stmt = $pdo -> query ($q);
  if (!$stmt) {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }
  $count_24h_psrs = $stmt ->fetch();

  $q = 'SELECT COUNT(*) FROM UTCs JOIN TB_Obs ON UTCs.id = TB_Obs.utc_id AND UTCs.id = TB_Obs.utc_id WHERE TIMESTAMPDIFF(MINUTE, UTCs.utc_ts, UTC_TIMESTAMP()) < 24*60 AND TB_Obs.snr > 7';
  $stmt = $pdo -> query ($q);
  if (!$stmt) {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }
  $count_24h_psrs_det = $stmt->fetch();

  $q = 'SELECT COUNT(*) FROM (SELECT COUNT(*) FROM UTCs JOIN TB_Obs ON UTCs.id = TB_Obs.utc_id AND UTCs.id = TB_Obs.utc_id WHERE TIMESTAMPDIFF(MINUTE, UTCs.utc_ts, UTC_TIMESTAMP()) < 24*60 AND TB_Obs.snr > 7 GROUP BY TB_Obs.psr_id) as t';

  $stmt = $pdo -> query ($q);
  if (!$stmt) {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }
  $count_24h_psrs_det_unique = $stmt->fetch();

  $q = 'SELECT SUM(Infos.INT)/60/60 FROM Infos JOIN UTCs JOIN FB_Obs ON UTCs.id = Infos.utc_id AND UTCs.id = FB_Obs.utc_id WHERE TIMESTAMPDIFF(MINUTE, UTCs.utc_ts, UTC_TIMESTAMP()) <= 24*60 AND Infos.FB_ENABLED = true';
  $stmt = $pdo -> query ($q);
  if (!$stmt) {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }
  $FRB_time = $stmt->fetch();

  $q = 'SELECT SUM(Infos.INT)/60/60 FROM Infos JOIN UTCs JOIN FB_Obs ON UTCs.id = Infos.utc_id AND UTCs.id = FB_Obs.utc_id WHERE TIMESTAMPDIFF(MINUTE, UTCs.utc_ts, UTC_TIMESTAMP()) <= 24*60 AND Infos.FB_ENABLED = true AND FB_Obs.gb < 20 AND FB_Obs.gb > -20';
  $stmt = $pdo -> query ($q);
  if (!$stmt) {
    echo "Failed to query:<br>".$q;
    exit(-1);
  }
  $FRB_time_plane = $stmt->fetch();


  echo "<tr><td><span class=best_snr>SUMMARY OF LAST 24hrs</span></td></tr>";
  echo "<tr><td>Total number of observations: ".$count_24h[0]."</td></tr>";
  echo "<tr><td>Total number of pulsars observed: ".$count_24h_psrs[0]."</td></tr>";
  echo "<tr><td>Number of pulsars with S/N > 7: ".$count_24h_psrs_det[0]."</td></tr>";
  echo "<tr><td>Number of unique pulsars with S/N > 7: ".$count_24h_psrs_det_unique[0]."</td></tr>";
  echo "<tr><td><hr></td></tr>";
  echo "<tr><td><span class=best_snr>FRB Statistics</span></td></tr>";
  echo "<tr><td>Total time on sky searching for FRBs: ".round($FRB_time[0], 2)."h";
  echo "<tr><td>Total time on sky searching for FRBs in the plane: ".round($FRB_time_plane[0], 2)."h";
  echo "<tr><td>Total time on sky searching for FRBs out of the plane: ".round($FRB_time[0] - $FRB_time_plane[0], 2)."h";

  $summary_filename = "/home/dada/linux_64/web/mopsr/latest_summary";
  $summary_contents = file_get_contents($summary_filename);
  $summary_array = explode(PHP_EOL, $summary_contents);
  $counter = 0;
  foreach($summary_array as $line) {
    if (strpos($line, "Number of classified") === 0 || strpos($line, "Number of single") === 0 || strpos($line, "Number of RFI") === 0) {
      echo "<tr><td>".$line.'</td></tr>';
    }
  }
    echo "</table>";
}
?>
