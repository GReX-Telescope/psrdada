<?PHP
include("definitions_i.php");
include("functions_i.php");

?>
<html>
<head>
  <meta http-equiv="Refresh" content="2">
</head>
<body>
<?

/* Checks to see if the page needs to be rebuilt */
if (file_exists(CHANGE_INSTRUMENT_FILE)) {

  # echo "changing to $instrument<BR>\n";
  unlink (CHANGE_INSTRUMENT_FILE);

?>

<script type="text/javascript">

  parent.banner.document.location = "statebanner.php";
  parent.statuswindow.document.location = "statuswindow.php";
  parent.infowindow.document.location = "infowindow.php";
  parent.plotwindow.document.location = "plotwindow.php";

  parent.logheader.document.location = "logheader.php";
  parent.logwindow.document.location = "machine_status.php?machine=nexus";

</script>
<?
} 
?> 
</body>
</html>
