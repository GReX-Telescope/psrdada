<?PHP

if (!$_CASPSR_WEBPAGE_LIB_PHP) { $_CASPSR_WEBPAGE_LIB_PHP = 1;

class caspsr_webpage 
{

  /* array of css pages requried */
  var $css = array("Arty/Arty.css");
  var $ejs = array();
  var $title = "CASPSR";

  function caspsr_webpage() 
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

  function openBlock($divargs="")
  {
?>
    <div class="Block" <?echo $divargs?>>
      <div class="Block-tl"></div>
      <div class="Block-tr"><div></div></div>
      <div class="Block-bl"><div></div></div>
      <div class="Block-br"><div></div></div>
      <div class="Block-tc"><div></div></div>
      <div class="Block-bc"><div></div></div>
      <div class="Block-cl"><div></div></div>
      <div class="Block-cr"><div></div></div>
      <div class="Block-cc"></div>
      <div class="Block-body">
<?
  }

  function closeBlock()
  {
?>
        </div>
      </div>
<?
  }

  function openBlockHeader($title, $divargs="")
  {
    $this->openBlock($divargs);
?>
        <div class="BlockHeader">
          <div class="header-tag-icon">
            <div class="BlockHeader-text"><?echo $title?></div>
          </div>
          <div class="l"></div>
          <div class="r"><div></div></div>
        </div>
        <div class="BlockContent">
          <div class="BlockContent-body">
            <div>
<?
  }

  function closeBlockHeader()
  {
?>
            </div>
          </div>
        </div>
<?
    $this->closeBlock();
  }
  

} // END CLASS DEFINITION

function handleDirect($child_class) 
{
  if ($_GET["single"] == "true") 
  {
    $obj = new $child_class();
    echo "<html>\n";
    echo "<head>\n";
    echo "  <title>".$obj->title."</title>\n";
    echo "    <link rel='shortcut icon' href='/caspsr/images/caspsr_favicon.ico'/>\n";
    for ($i=0; $i<count($obj->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$obj->css[$i]."'>\n";
    for ($i=0; $i<count($obj->ejs); $i++)
      echo "   <script type='text/javascript' src='".$obj->ejs[$i]."'></script>\n";
    if ($obj->javaScriptCallback() != "") {
      echo "  <script type='text/javascript'>\n";
      echo "    function poll_server()\n";
      echo "    {\n";
      echo "      ".$obj->javaScriptCallback()."\n";
      echo "      setTimeout('poll_server()', 4000);\n";
      echo "    }\n";
      echo "  </script>\n";
    }
    $obj->printJavaScriptHead();
    echo "</head>\n";

    if ($obj->javaScriptCallback() != "")
      echo "<body onload='poll_server()'>\n";
    else
      echo "<body>\n";
    $obj->printJavaScriptBody();
    /*if ($obj->javaScriptCallback() != "") {
      echo "<script type='text/javascript'>\n";
      echo "  Event.observe(window, 'load', function()\n";
      echo "  {\n";
      echo "    poll_server();\n";
      echo "  }, false);\n";
      echo "</script>\n";
    }*/
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

} // _CASPSR_WEBPAGE_LIB_PHP
