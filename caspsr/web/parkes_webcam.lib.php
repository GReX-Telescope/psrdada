<?php
include ("functions_i.php");
$host = "srv0";
$port = "52024";
list ($socket, $result) = openSocket($host, $port);
$bytes_written = socketWrite($socket, "dish_image\r\n");
$header_one = socketRead($socket);
$header_two = socketRead($socket);
$arr = explode(":",$header_two);
$arr2 = explode(" ",$arr[1]);
$size = $arr2[1];
$read = socket_read($socket, $size, PHP_BINARY_READ);
header($header_one);
header($header_two);
echo $read;
?>
