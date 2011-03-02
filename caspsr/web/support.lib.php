<?PHP

include("caspsr_webpage.lib.php");
include("definitions_i.php");
include("functions_i.php");
include($instrument.".lib.php");

class support extends caspsr_webpage 
{

  function support()
  {
    caspsr_webpage::caspsr_webpage();
    $this->title = "CASPSR | Help";
  }

  /* HTML for this page */
  function printHTML() 
  {
    $this->openBlockHeader("CASPSR Help &amp; Support");
?>
    <table>
      <tr>
        <td>
          <a href="https://docs.google.com/document/d/1TqPdFopf5TkQ4uFGPWRfsaxd9zsex4YbDtzDr0NO7bI/edit?hl=en&authkey=CJuapdQO">CASPSR Observer's Guide</a>
        </td>
        <td>Basic information on the use of CASPSR</td>
      </tr>
      <tr>
        <td>
          <a href="http://psrdada.sf.net/support.shtml" style="text-decoration: none"> <font color=green>PSR</font>DADA Support</a>
        </td>
        <td>To report bugs and/or request features</td>
      </tr>
    </table>

<?
    $this->closeBlock();
  }

}

handledirect("support");
