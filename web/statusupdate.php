<?PHP 
include("definitions_i.php");
include("functions_i.php");

/* Need to clear PHP's internal cache */
clearstatcache();

$pwc_config = getConfigFile(SYS_CONFIG);
$pwc_status = getAllStatuses($pwc_config);

$pwcc_status = STATUS_OK;
$pwcc_message = "no probs mate";

echo "NUM_PWC:::".$pwc_config["NUM_PWC"].";;;";
for($i=0; $i<$pwc_config["NUM_PWC"]; $i++) {
 echo "PWC_".$i.":::".$pwc_config["PWC_".$i].";;;";
 echo "PWC_".$i."_STATUS:::". $pwc_status["PWC_".$i."_STATUS"].";;;";
 echo "PWC_".$i."_MESSAGE:::".$pwc_status["PWC_".$i."_MESSAGE"].";;;";
 echo "SRC_".$i."_STATUS:::". $pwc_status["SRC_".$i."_STATUS"].";;;";
 echo "SRC_".$i."_MESSAGE:::".$pwc_status["SRC_".$i."_MESSAGE"].";;;";
 echo "SYS_".$i."_STATUS:::". $pwc_status["SYS_".$i."_STATUS"].";;;";
 echo "SYS_".$i."_MESSAGE:::".$pwc_status["SYS_".$i."_MESSAGE"].";;;";
}
