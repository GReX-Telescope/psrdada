<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class support extends apsr_webpage
{

  function support()
  {
    apsr_webpage::apsr_webpage();
  }

  function printHTML()
  {
  
    echo "<div style='padding:10px'>\n"; 
    $this->openBlockHeader("Support");
?>
    <p>APSR Observer's Guide: <a href="https://docs.google.com/document/d/1XQUVPO_eQk3YJntNPFqLYYOzNp8RMt7FSFPIdBQ75vM/edit?hl=en_US&authkey=CPSjyGY" style="text-decoration: none">Google Document Link</a></p>

    <p>
    To report bugs and/or request features: <a href="http://psrdada.sf.net/support.shtml" style="text-decoration: none"> <font color=green>PSR</font>DADA Support</a>
    </p>

<?
    $this->closeBlockHeader();
    echo "</div>\n"; 
  }

}
handledirect("support");
