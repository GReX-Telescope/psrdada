<?PHP

#
# Global configuration independent of machine
#


if (!$_SITE_DEFINITIONS_I_PHP) 
{
  $_SITE_DEFINITIONS_I_PHP = 1;

  if (!(defined('INSTRUMENT')))
  {
    echo "ERROR: Required constant INSTRUMENT [".INSTRUMENT."] not defined<BR>\n";
    exit;
  }

  /* Directory configuration */
  define(DADA_ROOT,  "/home/dada/linux_64");
  define(WEB_BASE,   DADA_ROOT."/web/".INSTRUMENT);
  define(URL_BASE,   "/web/".INSTRUMENT);
  define(URL_FULL,   $_SERVER["HTTP_HOST"]."/".INSTRUMENT);

  /* Time/date configuration */
  define(DADA_TIME_FORMAT,  "Y-m-d-H:i:s");
  define(UTC_TIME_OFFSET,   10);
  define(LOCAL_TIME_OFFSET, (UTC_TIME_OFFSET*3600));

  define(STATUS_OK,    "0");
  define(STATUS_WARN,  "1");
  define(STATUS_ERROR, "2");

  define(LOG_FILE_SCROLLBACK_HOURS,6);
  define(LOG_FILE_SCROLLBACK_SECS,LOG_FILE_SCROLLBACK_HOURS*60*60);

  define(FAVICO_HTML,         "<link rel='shortcut icon' href='/images/favicon.ico'/>");

} //_SITE_DEFINITIONS_I_PHP
?>
