<?PHP

include("definitions_i.php");
include("functions_i.php");

?>
<HTML>
<?
$title = "APSR: ATNF Parkes Swinburne Recorder";
include("header_i.php");
?>

<FRAMESET ROWS="1" COLS="800px,*" border=1>
  <FRAMESET COLS=1 ROWS="0px,60px,110px,60px,90px,*" border=1>
    <FRAME name="instrument" src="instrument.php" frameborder=0 marginheight=0 marginwidth=0></FRAME>
    <FRAME name="banner" src="statebanner.php" frameborder=0 marginheight=0 marginwidth=0></FRAME>
    <FRAME name="statuswindow" src="statuswindow.php"></FRAME>
    <FRAME name="controlwindow" src="controlwindow.php"></FRAME>
    <FRAME name="infowindow" src="infowindow.php"></FRAME>
    <FRAME name="plotwindow" src="plotwindow.php"></FRAME>
  </FRAMESET>
  <FRAMESET name=rightwindow COLS=1 ROWS="70px,*" border=0>
    <FRAME name="logheader" src="logheader.php" frameborder=0 marginheight=0 marginwidth=0></FRAME>
    <FRAME name="logwindow" src="machine_status.php?machine=nexus"></FRAME>
  </FRAMESET>
</FRAMESET>
</HTML>
