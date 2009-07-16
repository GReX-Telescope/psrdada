<?PHP

define(INSTRUMENT, "caspsr");
define(CFG_FILE, "caspsr.cfg");
define(CSS_FILE, "caspsr.css");

include("site_definitions_i.php");
include("instrument.lib.php");

class caspsr extends instrument
{

  function caspsr()
  {
    instrument->instrument(INSTRUMENT, CFG_FILE, URL);
  }

  function print_header($title, $refresh=0) 
  {
      echo "<head>\n";
      echo "  <title>".$title."</title>\n";
      echo "  <link rel='STYLESHEET' type='text/css' href='".URL_BASE."/".CSS_FILE.">\n";
      if ($refresh > 0) 
        echo "<meta http-equiv='Refresh' content='".$refresh."'>\n";
      echo "</head>\n";
  }

}
