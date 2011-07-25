<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class config_edit extends apsr_webpage 
{
  var $inst;
  var $file;
  
  function config_edit()
  {
    apsr_webpage::apsr_webpage();
    $this->inst = new apsr();

    if (isset($_GET["file"])) 
      $this->file = $this->inst->config["CONFIG_DIR"]."/".$_GET["file"];
    else
      $this->file = "";

    $this->title = "APSR | Config File Editor";
  }

  function printJavaScriptHead()
  {
?>
    <style>
      table.confg_edit{
        font-size: 10pt;
      }
      table.config_edit th {
        text-align: right;
        font-style: bold;
      }
      table.config_edit td {
        text-align: left;
      }
    </style>

    <script type='text/javascript'>  

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

    <div style='padding: 10px;'>
<?
    $this->openBlockHeader("Config File: ".basename($this->file));

    // this is the update request
    if (isset($_POST["file"]))
    {
      $this->file = $this->inst->config["CONFIG_DIR"]."/". $_POST["file"];

      $fptr = @fopen($this->file, "w");
      if ($fptr)
      {
        fprintf($fptr, "# Updated by: config_edit.lib.php\n");
        fprintf($fptr, "# Updated: ".date(DADA_TIME_FORMAT)."\n\n");
        foreach ($_POST as $key => $value)
        {
          if ($key != "file")
            fprintf($fptr, str_pad($key, 19)." ".$value."\n");
        }
        fclose($fptr);
      }
      echo "<p>Config file Updated!</p>\n";
    }

    if ($this->file != "") 
    {
      $config = instrument::configFileToHash($this->file);
      echo "<form name='config_edit' action='config_edit.lib.php?single=true' method='POST'>\n";
      echo "<input type='hidden' name='file' value='".basename($this->file)."'>\n";
      echo "<table class='config_edit' width='100%'>\n";
      foreach ($config as $key => $value) 
      {
        echo "<tr><th>".$key."</th><td><input size='128' type='text' name='".$key."' value='".$value."'></th></tr>\n";
      }
      echo "</table>\n";
      echo "<input type='submit' value='Submit Changes'>\n";
      echo "</form>\n";
    }
    else
      echo "<p>No Config file specified</p>\n";
    $this->closeBlockHeader();
?>
    </div>
<?
  }
}

handledirect("config_edit");

