<?PHP

include("definitions_i.php");

if (file_exists(CHANGE_INSTRUMENT_FILE)) {
  echo "change";
} else {
  echo "no change";
}

?>
