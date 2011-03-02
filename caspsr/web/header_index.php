<?PHP

include "caspsr_webpage.lib.php";

class result_index extends caspsr_webpage
{
  var $full_path;
  var $utc_start;

  function result_index()
  {
    caspsr_webpage::caspsr_webpage();
    $this->full_path = $_SERVER["SCRIPT_URL"];
    $bits = explode("/", $this->full_path);
    for ($i=count($bits); $i>0; $i--)
    {
      if (($this->utc_start == "") && ($bits[$i] != "")) {
        $this->utc_start = $bits[$i];
      }
    }
  }

  function print_header()
  {
?>
<html>
  <head>
    <title>CASPSR | Directoy Listing of <?echo $this->full_path?></title>
    <link rel="stylesheet" type="text/css" href="/caspsr/Arty/Arty.css">
    <link rel='shortcut icon' href='/caspsr/images/caspsr_favicon.ico'/>
    <style type="text/css">
      table {
        font-size: 16px;
      }
      td {
        font-size: 16px;
      }
    </style>
  </head>
  <body>
<?
    $this->openBlockHeader("Directory Listing for ".$this->utc_start);
  }

  function print_footer()
  {
    $this->closeBlockHeader();
?>
  </body>
</html>
<?
  }
} 

$obj = new result_index;
if ($_GET["type"] == "header")
  $obj->print_header();
else if ($_GET["type"] == "footer")
  $obj->print_footer();
else
  ;
?>
