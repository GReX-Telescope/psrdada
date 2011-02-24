<?PHP

if (!$_DADA_WEBPAGE_LIB_PHP) { $_DADA_WEBPAGE_LIB_PHP = 1;

class dada_webpage 
{
  /* array of css pages requried */
  var $css = array("dada.css");
  var $ejs = array();
  var $title = "PSRDADA";
  var $callback_freq = 4000;
  var $doc_type = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd'>";

  function dada_webpage() 
  {

  }

  function javaScriptCallBack()
  {

  }

  function printJavaScriptHead() 
  {

  }

  function printJavaScriptBody()
  {

  }

  function printHTML()
  {

  }

  function printUpdateHTML()
  {

  }

  function openBlockHeader($block_title)
  {
    echo "<table class='wrapper'>\n";
    echo "  <tr><th class='wrapper'>".$block_title."</th></tr>\n";
    echo "  <tr><td class='wrapper'>\n";
  }

  function closeBlockHeader()
  {
    echo "  </td></tr>\n";
    echo "</table>\n";
  }


} // END CLASS DEFINITION


#
# handle a direct inclusion of the specified class
#
function handleDirect($child_class) 
{

  // if this parameter is defined, output the HTML for the
  // specified pages
  if ($_GET["single"] == "true") 
  {
    $obj = new $child_class();

    echo $obj->doc_type."\n";
    echo "<html>\n";
    echo "  <head>\n";
    echo "    <title>".$obj->title."</title>\n";
    echo "    <link rel='shortcut icon' href='/images/dada_favico.ico'/>\n";

    // css and javascript includes
    for ($i=0; $i<count($obj->css); $i++)
      echo "    <link rel='stylesheet' type='text/css' href='".$obj->css[$i]."'>\n";
    for ($i=0; $i<count($obj->ejs); $i++)
      echo "    <script type='text/javascript' src='".$obj->ejs[$i]."'></script>\n";

    // callbacks for javascript pollings
    if ($obj->javaScriptCallback() != "")
    {
      echo "    <script type='text/javascript'>\n";
      echo "      function poll_server()\n";
      echo "      {\n";
      echo "        ".$obj->javaScriptCallback()."\n";
      echo "        setTimeout('poll_server()', ".$obj->callback_freq.");\n";
      echo "      }\n";
      echo "    </script>\n";
    }

    // javascript head scripts
    if ($obj->printJavaScriptHead() != "")
      $obj->printJavaScriptHead();

    echo "  </head>\n";
  
    // if we need to run the callback
    if ($obj->javaScriptCallback() != "")
      echo "<body onload='poll_server()'>\n";
    else
      echo "<body>\n";

    $obj->printJavaScriptBody();

    $obj->printHTML();

    echo "</body>\n";
    echo "</html>\n";

  } else if ($_GET["update"] == "true") {

    $obj = new $child_class();
    $obj->printUpdateHTML($_GET["host"], $_GET["port"]);

  } else {
    # do nothing :)
  }
}

} // _DADA_WEBPAGE_LIB_PHP

