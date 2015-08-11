<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class tmc_simulator extends mopsr_webpage 
{
  var $psrs = array();
  var $psr_keys = array();
  var $valid_psrs = array();
  var $inst = 0;
  var $passthru = false;
  var $oversampling = false;
  var $sixteen_inputs = true;

  # defaults are critically sampled PFB
  var $bandwidth = 100;
  var $cfreq = 849.609375;
  var $nchan = 128;
  var $channel_bandwidth = 0.78125;
  var $oversampling_ratio = 1;
  var $sampling_time = 1.28;
  var $dsb = 1;
  var $nant = 4;
  var $resolution;

  function tmc_simulator()
  {
    mopsr_webpage::mopsr_webpage();

    $this->title = "MOPSR | TMC Simulator";
    $this->inst = new mopsr();
    $this->valid_psrs = array(

			      "J0001+0001", "J0034-0721", "J0255-5304", "J0401-7608", "J1630-4733","J1715-4034","J1717-3424",
			      "J0437-4715", "J0452-1759", "J0536-7543", "J0534+2200", "J0601-0527", "J1730-3350",
			      "J1745-3040", "J1225-6408",
			      "J0610-2100", "J0613-0200", "J0711-6830", "J0737-3039A", 
			      "J0742-2822", "J0835-4510", "J0820-1350", "J1534-5334","J1614-5048",
                              "J0900-3144", "J0904-7459", "J0941-39", "J0942-5552", "J0953+0755", 
			      "J1001-5939", "J1001-5507", "J1048-5832","J1544-5308", 
			      "J0837+0610","J1202-5820", "J1305-6455",
			      "J1721-3532","J1216-5027", "J1558-5419",
			      "J1707-4053","J1210-5559", "J1559-5545",
			      "J1709-4429", "J1418-3921", "J1453-0620","J1731-4744",
			      "J1357-6228", "J1600-3053", "J1600-5751",
			      "J1651-4246","J1603-7202","J1613-4714",
			      "J1705-3423",
			      "J1709-4429", "J1705-1906",
			      "J1740-3015", "J1738+0333",
			      "J1825-0935",
			      "J1723-2837","J1703-3241",
			      "J1820-0427", "J1648-3256", "J1649-3805",
			      "J1824-1945", "J1231-6303",
			      "J1829-1751","J1836-1008",
			      "J1848-0123", "J1847-0402",
			      "J1900-2600","J1913-0440","J2026-0421",
			      "J2144-3933",
			      "J1359-6038", "J1146-6030", "J1302-6350", "J1306-6617",
			      "J1605-5257", "J1604-4909", "J1319-6056",
			      "J1107-5907", "J0206-4028",
			      "J0907-5157", "J2330-2005",
			      "J0942-5552", "J0151-0635",
			      "J1001-5507", "J0152-1637",
			      "J1056-6258", "J1550-5418",
			      "J1057-5226", "J1253-5820",
			      "J1116-4122",
			      "J1136-5525", "J1824-2452A",
			      "J1401-6357",
			      "J1428-5530", "J1557-4258",
			      "J1430-6623", "J1623-4256", 
			      "J1453-6413", "J1435-5954",
			      "J1456-6843",
			      "J1534-5334",
			      "J1557-4258",
			      "J1600-5044", "J1327-6301", "J1328-4357", "J1341-6220", "J1352-6803",
			      "J1602-5100",
                              "J1018-7154", "J1022+1001", "J1024-0719", "J1034-3224", 
			      "J1045-4509",
			      "J1056-6258", "J1057-5226",
                              "J1103-5355", "J1116-4122", "J1125-5825", "J1125-6014", 
			      "J1141-6545", "J1157-6224",
                              "J1226-6202", "J1243-6423", "J1326-5859",  "J1327-6222",
			      "J1224-6407", "J1243-6423", "J1312-5516", "J1320-5359", "J1326-6700",
			      "J1326-6408",
			      "J1430-6623", 
			      "J1431-4717", "J1439-5501", "J1525-5544",
                              "J1546-4552", "J1600-3053", "J1600-5044", "J1603-7202", 
			      "J1633-5015", "J1643-1224",
                              "J1644-4559", "J1709-1640", "J1731-4744",
			      "J1713+0747", "J1717-4054", "J1718-3718", "J1730-2304",
                              "J1732-5049", "J1744-1134", "J1751-4657", 
			      "J1807-0847", "J1820-0427", "J1825-0935", "J1824-2452", 
			      "J1844+00", "J1857+0943", 
                              "J1909-3744", "J1900-2600", "J1913-0440", 
			      "J1933-6211", "J1939+2134", "J2124-3358",
                              "J2129-5721", "J2145-0750", "J2241-5236", "J1456-6843",
			      "J0630-2834", "J1017-7156", "J0908-4913",
                              "J0738-4042", "J0630-2834", "J0738-4042", "J0742-2822",
                              "J0835-4510", "J0837-4135", "J0953+0755", "J1136+1551", "J1819-1458",
                              "J1453-6413", "J1456-6843", "J1559-4438", "J1644-4559", "J2048-1616",
                              "J1645-0317", "J1752-2806", "J1932+1059", "J1935+1616", 
			      "J1600-5751", 
			      "J1603-7202",
			      "J1603-2531",
			      "J1613-4714",
			      "J1623-2631",
			      "J1623-4256",
			      "J1625-4048",
			      "J1808-0813", 
			      "J1812-1733",
			      "J1812-2102",
			      "J1816-2650",
			      "J1822-2256",
			      "J1637-4553",
			      "J1646-6831",
			      "J1650-1654" ,
			      "J1651-5222",
			      "J1700-3312",
			      "J1703-4851",
			      "J1705-3423",
			      "J1708-3426",
			      "J1720-2933",
			      "J1730-3350",
			      "FRB Transit", 
			      "SUMSS1200M80", "SUMSS1330M76", 
                              "CenA",       "3C273", "HerA", "3C353", "E1", "1hr", "FRB010724",
			      "3hr", "FRB090625", "CDF-S", "4hr", "Prime", "FRB131104", "SN1987A",
			      "9hr", "Polar", "FRB121002", "HD94660",  
			      "CJ1744-5144",
			      "CJ0010-4153",
			      "CJ0025-2602",
                              "CJ0200-3053", 
                              "CJ0252-7104", 
                              "CJ0408-6545", 
                              "CJ0408-7507", 
                              "CJ0440-4333",
			      "CJ0519-4545", 
                              "CJ0522-3627", 
			      "CJ0635-7516",
			      "CJ0743-6726",
			      "CJ0841-7540",
			      "CJ1018-3144",
			      "CJ1020-4251", 
			      "CJ1154-3505", 
			      "CJ1218-4600", 
			      "CJ1248-4118", 
			      "CJ1305-4928", 
			      "CJ1325-4301", 
			      "CJ1424-4913", 
			      "CJ1530-4231",
			      "CJ1556-7914", 
			      "CJ1737-5632",
                              "CJ1819-6345", 
			      "CJ1830-3602", 
			      "CJ1924-2833",
			      "CJ1935-4620",
			      "CJ1957-4222", 
			      "CJ2154-5150",
			      "CJ2253-4057",
			      "CJ2334-4125"
			      );

    if ($this->sixteen_inputs)
    {
      $this->nant = 16;
      $this->bandwidth = 31.25;
      $this->cfreq = 834.765625;
      $this->nchan = 40;
    }

    if ($this->oversampling)
    {
      $this->sampling_time = 1.08;
      $this->channel_bandwidth = "0.9259259259";
    }
    if ($this->passthru)
    {
      $this->nchan = 1;
      $this->chann_bandwidth = 100;
      $this->sampling_time = 0.01;
      $this->dsb = 0;
    }

    $this->resolution = $this->nchan * $this->nant * 2;
  }

  function printJavaScriptHead()
  {
    $this->psrs = $this->inst->getPsrcatPsrs();

    $this->psrs["E1"] = array();     
    $this->psrs["E1"]["RAJ"] = "00:30:00.0";
    $this->psrs["E1"]["DECJ"] = "-44:00:00.0";

    $this->psrs["1hr"] = array();     
    $this->psrs["1hr"]["RAJ"] = "01:15:00.0";
    $this->psrs["1hr"]["DECJ"] = "-50:25:00";

    $this->psrs["FRB010724"] = array();     
    $this->psrs["FRB010724"]["RAJ"] = "01:18:06.0";
    $this->psrs["FRB010724"]["DECJ"] = "-75:12:19.0";

    $this->psrs["3hr"] = array();     
    $this->psrs["3hr"]["RAJ"] = "03:00:00.0";
    $this->psrs["3hr"]["DECJ"] = "-55:25:00.0";

    $this->psrs["FRB090625"] = array();     
    $this->psrs["FRB090625"]["RAJ"] = "03:07:47.2";
    $this->psrs["FRB090625"]["DECJ"] = "-29:55:35.9";

    $this->psrs["CDF-S"] = array();     
    $this->psrs["CDF-S"]["RAJ"] = "03:30:24.0";
    $this->psrs["CDF-S"]["DECJ"] = "-28:06:00.0";

    $this->psrs["SN1987A"] = array();     
    $this->psrs["SN1987A"]["RAJ"] = "05:35:28.0";
    $this->psrs["SN1987A"]["DECJ"] = "-69:16:11.1";

    $this->psrs["4hr"] = array();     
    $this->psrs["4hr"]["RAJ"] = "04:10:00.0";
    $this->psrs["4hr"]["DECJ"] = "-55:00:00.0";

    $this->psrs["Prime"] = array();     
    $this->psrs["Prime"]["RAJ"] = "05:55:07.0";
    $this->psrs["Prime"]["DECJ"] = "-61:21:00.0";

    $this->psrs["FRB131104"] = array();     
    $this->psrs["FRB131104"]["RAJ"] = "06:44:08.7";
    $this->psrs["FRB131104"]["DECJ"] = "-51:16:55.1";

    $this->psrs["9hr"] = array();     
    $this->psrs["9hr"]["RAJ"] = "09:00:00.0";
    $this->psrs["9hr"]["DECJ"] = "-70:00:00.0";

    $this->psrs["Polar"] = array();     
    $this->psrs["Polar"]["RAJ"] = "16:00:00.0";
    $this->psrs["Polar"]["DECJ"] = "-74:00:00.0";

    $this->psrs["FRB121002"] = array();     
    $this->psrs["FRB121002"]["RAJ"] = "18:14:47.4";
    $this->psrs["FRB121002"]["DECJ"] = "-85:11:53.0";

    $this->psrs["FRB Transit"] = array();     
    $this->psrs["FRB Transit"]["RAJ"] = "00:00:00.0";
    $this->psrs["FRB Transit"]["DECJ"] = "-46:00:00";

    $this->psrs["SUMSS1200M80"] = array();     
    $this->psrs["SUMSS1200M80"]["RAJ"] = "12:00:00.0";
    $this->psrs["SUMSS1200M80"]["DECJ"] = "-80:00:00";

    $this->psrs["SUMSS1330M76"] = array();     
    $this->psrs["SUMSS1330M76"]["RAJ"] = "13:30:00.0";
    $this->psrs["SUMSS1330M76"]["DECJ"] = "-76:00:00";

    $this->psrs["CenA"] = array();     
    $this->psrs["CenA"]["RAJ"] = "13:25:27.6";
    $this->psrs["CenA"]["DECJ"] = "-43:01:09";

    $this->psrs["HerA"] = array();     
    $this->psrs["HerA"]["RAJ"] = "16:51:08.0";
    $this->psrs["HerA"]["DECJ"] = "04:59:33";

    $this->psrs["3C273"] = array();
    $this->psrs["3C273"]["RAJ"] = "12:29:06.7";
    $this->psrs["3C273"]["DECJ"] = "02:03:09.0";

    $this->psrs["3C353"] = array();
    $this->psrs["3C353"]["RAJ"] = "17:20:28.0";
    $this->psrs["3C353"]["DECJ"] = "-00:58:47.0";

    $this->psrs["HD94660"] = array();
    $this->psrs["HD94660"]["RAJ"] = "10:55:01.005";
    $this->psrs["HD94660"]["DECJ"] = "-42:15:03.93";

    $this->psrs["CJ0025-2602"] = array();
    $this->psrs["CJ0025-2602"]["RAJ"] = "00:25:49.2";
    $this->psrs["CJ0025-2602"]["DECJ"] = "-26:02:12.6";

    $this->psrs["CJ0200-3053"] = array();
    $this->psrs["CJ0200-3053"]["RAJ"] = "02:00:12.13";
    $this->psrs["CJ0200-3053"]["DECJ"] = "-30:53:25.5";

    $this->psrs["CJ0252-7104"] = array();
    $this->psrs["CJ0252-7104"]["RAJ"] = "02:52:46.3";
    $this->psrs["CJ0252-7104"]["DECJ"] = "-71:04:36.2";

    $this->psrs["CJ0408-6545"] = array();
    $this->psrs["CJ0408-6545"]["RAJ"] = "04:08:20.3";
    $this->psrs["CJ0408-6545"]["DECJ"] = "-65:45:08.5";

    $this->psrs["CJ0408-7507"] = array();
    $this->psrs["CJ0408-7507"]["RAJ"] = "04:08:48.5";
    $this->psrs["CJ0408-7507"]["DECJ"] = "-75:07:20.0";

    $this->psrs["CJ0440-4333"] = array();
    $this->psrs["CJ0440-4333"]["RAJ"] = "04:40:17.07";
    $this->psrs["CJ0440-4333"]["DECJ"] = "-43:33:09.0";

    $this->psrs["CJ0519-4545"] = array();
    $this->psrs["CJ0519-4545"]["RAJ"] = "05:19:26.6";
    $this->psrs["CJ0519-4545"]["DECJ"] = "-45:45:58.9";

    $this->psrs["CJ0522-3627"] = array();
    $this->psrs["CJ0522-3627"]["RAJ"] = "05:22:57.8";
    $this->psrs["CJ0522-3627"]["DECJ"] = "-36:27:31.0";

    $this->psrs["CJ0635-7516"] = array();
    $this->psrs["CJ0635-7516"]["RAJ"] = "06:35:45.05";
    $this->psrs["CJ0635-7516"]["DECJ"] = "-75:16:15.3";

    $this->psrs["CJ0743-6726"] = array();
    $this->psrs["CJ0743-6726"]["RAJ"] = "07:43:32.63";
    $this->psrs["CJ0743-6726"]["DECJ"] = "-67:26:28.6";

    $this->psrs["CJ0841-7540"] = array();
    $this->psrs["CJ0841-7540"]["RAJ"] = "08:41:26.19";
    $this->psrs["CJ0841-7540"]["DECJ"] = "-75:40:31.1";

    $this->psrs["CJ1018-3144"] = array();
    $this->psrs["CJ1018-3144"]["RAJ"] = "10:18:09.19";
    $this->psrs["CJ1018-3144"]["DECJ"] = "-31:44:14.7";

    $this->psrs["CJ1020-4251"] = array();
    $this->psrs["CJ1020-4251"]["RAJ"] = "10:20:03.5";
    $this->psrs["CJ1020-4251"]["DECJ"] = "-42:51:33.0";

    $this->psrs["CJ1154-3505"] = array();
    $this->psrs["CJ1154-3505"]["RAJ"] = "11:54:21.9";
    $this->psrs["CJ1154-3505"]["DECJ"] = "-35:05:32.2";

    $this->psrs["CJ1218-4600"] = array();
    $this->psrs["CJ1218-4600"]["RAJ"] = "12:18:06.0";
    $this->psrs["CJ1218-4600"]["DECJ"] = "-46:00:29.2";

    $this->psrs["CJ1248-4118"] = array();
    $this->psrs["CJ1248-4118"]["RAJ"] = "12:48:49.8";
    $this->psrs["CJ1248-4118"]["DECJ"] = "-41:18:42.2";

    $this->psrs["CJ1305-4928"] = array();
    $this->psrs["CJ1305-4928"]["RAJ"] = "13:05:27.4";
    $this->psrs["CJ1305-4928"]["DECJ"] = "-49:28:06.3";

    $this->psrs["CJ1325-4301"] = array();
    $this->psrs["CJ1325-4301"]["RAJ"] = "13:25:24.0";
    $this->psrs["CJ1325-4301"]["DECJ"] = "-43:01:38.1";

    $this->psrs["CJ1424-4913"] = array();
    $this->psrs["CJ1424-4913"]["RAJ"] = "14:24:32.18";
    $this->psrs["CJ1424-4913"]["DECJ"] = "-49:13:17.16";

    $this->psrs["CJ1556-7914"] = array();
    $this->psrs["CJ1556-7914"]["RAJ"] = "15:56:57.8";
    $this->psrs["CJ1556-7914"]["DECJ"] = "-79:14:03.8";

    $this->psrs["CJ1737-5632"] = array();
    $this->psrs["CJ1737-5632"]["RAJ"] = "17:37:42.85";
    $this->psrs["CJ1737-5632"]["DECJ"] = "-56:32:46.0";

    $this->psrs["CJ1744-5144"] = array();
    $this->psrs["CJ1744-5144"]["RAJ"] = "17:44:25.47";
    $this->psrs["CJ1744-5144"]["DECJ"] = "-51:44:43.1";

    $this->psrs["CJ1819-6345"] = array();
    $this->psrs["CJ1819-6345"]["RAJ"] = "18:19:35.0";
    $this->psrs["CJ1819-6345"]["DECJ"] = "-63:45:48.6";

    $this->psrs["CJ1830-3602"] = array();
    $this->psrs["CJ1830-3602"]["RAJ"] = "18:30:58.8";
    $this->psrs["CJ1830-3602"]["DECJ"] = "-36:02:30.3";

    $this->psrs["CJ1935-4620"] = array();
    $this->psrs["CJ1935-4620"]["RAJ"] = "19:35:57.2";
    $this->psrs["CJ1935-4620"]["DECJ"] = "-46:20:43.1";

    $this->psrs["CJ1924-2833"] = array();
    $this->psrs["CJ1924-2833"]["RAJ"] = "19:24:50.2";
    $this->psrs["CJ1924-2833"]["DECJ"] = "-28:33:39.4";

    $this->psrs["CJ2334-4125"] = array();
    $this->psrs["CJ2334-4125"]["RAJ"] = "23:34:26.1";
    $this->psrs["CJ2334-4125"]["DECJ"] = "-41:25:25.8";

    $this->psrs["CJ1530-4231"] = array();
    $this->psrs["CJ1530-4231"]["RAJ"] = "15:30:14.28";
    $this->psrs["CJ1530-4231"]["DECJ"] = "-42:31:53.6";

    $this->psrs["CJ1957-4222"] = array();
    $this->psrs["CJ1957-4222"]["RAJ"] = "19:57:15.19";
    $this->psrs["CJ1957-4222"]["DECJ"] = "-42:22:19.8";

    $this->psrs["CJ0010-4153"] = array();
    $this->psrs["CJ0010-4153"]["RAJ"] = "00:10:52.44";
    $this->psrs["CJ0010-4153"]["DECJ"] = "-41:53:10.8";

    $this->psrs["CJ2154-5150"] = array();
    $this->psrs["CJ2154-5150"]["RAJ"] = "21:54:07.28";
    $this->psrs["CJ2154-5150"]["DECJ"] = "-51:50:18.1";

    $this->psrs["CJ2253-4057"] = array();
    $this->psrs["CJ2253-4057"]["RAJ"] = "22:53:03.37";
    $this->psrs["CJ2253-4057"]["DECJ"] = "-40:57:47.5";

    $this->psrs["CJ0440-4333"] = array();
    $this->psrs["CJ0440-4333"]["RAJ"] = "04:40:17.06";
    $this->psrs["CJ0440-4333"]["DECJ"] = "-43:33:09.0";






    $this->psr_keys = array_keys($this->psrs);
?>
    <style type='text/css'>

      td.key {
        text-align: right;
      }
 
      td.val {
        padding-right: 20px;
        text-align: left;
      } 

    </style>


    <script type='text/javascript'>
      var ras = { 'default':'00:00:00.00'<?
      for ($i=0; $i<count($this->psr_keys); $i++)
      {
        $p = $this->psr_keys[$i];
        if (in_array($p, $this->valid_psrs))
        {
          echo ",'".$p."':'".$this->psrs[$p]["RAJ"]."'";
        }
      }
      ?>};

      var decs = { 'default':'00:00:00.00'<?
      for ($i=0; $i<count($this->psr_keys); $i++)
      {
        $p = $this->psr_keys[$i];
        if (in_array($p, $this->valid_psrs))
        {
          echo ",'".$p."':'".$this->psrs[$p]["DECJ"]."'";
        }
      }
      ?>};


      function prepareButton() 
      {
        document.getElementById("command").value = "prepare";

        var i = 0;
        var psr = "";

        updateRADEC();

        i = document.getElementById("src_list").selectedIndex;
        psr = document.getElementById("src_list").options[i].value;

        document.getElementById("source").value = psr;
        document.tmc.submit();
      }

      function startButton()
      {
        document.getElementById("command").value = "start";
        document.tmc.submit();
      }

      function stopButton() {
        document.getElementById("command").value = "stop";
        document.tmc.submit();
      }

      function queryButton() {
        document.getElementById("command").value = "query";
        document.tmc.submit();
      }

      function updateRADEC() {
        var i = document.getElementById("src_list").selectedIndex;
        var psr = document.getElementById("src_list").options[i].value;
        var psr_ra = ras[psr];
        var psr_dec= decs[psr];
        document.getElementById("ra").value = psr_ra;
        document.getElementById("dec").value = psr_dec;
      }
    </script>

<?
  }

  /*************************************************************************************************** 
   *
   * HTML for this page 
   *
   ***************************************************************************************************/
  function printHTML()
  {
    $this->openBlockHeader("TMC Simulator");
?>
    <form name="tmc" target="tmc_interface" method="GET">
    <table border=0 cellpadding=5 cellspacing=0 width='100%'>
      <tr>

        <td class='key'>SOURCE</td>
        <td class='val'>
          <input type="hidden" id="source" name="source" value="">
          <select id="src_list" name="src_list" onChange='updateRADEC()'>
<?
          for ($i=0; $i<count($this->psr_keys); $i++)
          {
            $p = $this->psr_keys[$i];
            if (in_array($p, $this->valid_psrs))
            {
              if ($p == "J0835-4510")
                echo "            <option value='".$p."' selected>".$p."</option>\n";
              else
                echo "            <option value='".$p."'>".$p."</option>\n";
            }
          }
?>
          </select>
        </td>


        <td class='key'>AQ PROC</td>
        <td class='val'>
          <select name="aq_processing_file">
            <option value="mopsr.aqdsp.gpu">mopsr.aqdsp.gpu</option>
            <option value="mopsr.aqdsp.unscaled.gpu">mopsr.aqdsp.unscaled.gpu</option>
            <option value="mopsr.null">mopsr.null</option>
            <option value="mopsr.dspsr.cpu">mopsr.dspsr.cpu</option>
            <option value="mopsr.dspsr.cpu.odd">mopsr.dspsr.cpu.odd</option>
            <option value="mopsr.dspsr.cpu.sk">mopsr.dspsr.cpu.sk</option>
            <option value="mopsr.dspsr.gpu">mopsr.dspsr.gpu</option>
            <option value="mopsr.dspsr.gpu.sk" <? if (!$this->passthru) echo "selected"?>>mopsr.dspsr.gpu.sk</option>
            <option value="mopsr.dspsr.gpu.fb" <? if ($this->passthru) echo "selected"?>>mopsr.dspsr.gpu.fb</option>
            <option value="mopsr.dspsr.gpu.1fold">mopsr.dspsr.gpu.1fold</option>
            <option value="mopsr.dspsr.gpu.10fold">mopsr.dspsr.gpu.10fold</option>
            <option value="mopsr.dbdisk">mopsr.dbdisk</option>
            <option value="mopsr.dbib">mopsr.dbib</option>
          </select>
        </td>

        <td class='key'>BANDWIDTH</td>
        <td class='val'><input type="text" name="bandwidth" value="<?echo $this->bandwidth?>" size="12"></td>

        <td class='key'>NPOL</td>
        <td class='val'><input type="text" name="npol" size="1" value="1" readonly></td>

      </tr>
      <tr>

        <td class='key'>MODE</td>
        <td class='val'>
          <select name="mode">
            <option value="PSR">PSR</option>
            <option value="CORR">CORR</option>
            <option value="CORR_CAL">CORR_CAL</option>
          </select>
        </td>

        <td class='key'>BF PROC</td>
        <td class='val'>
          <select name="bf_processing_file">
            <option value="mopsr.calib.gpu">mopsr.calib.gpu [correlator]</option>
            <option value="mopsr.calib.pref8.gpu">mopsr.calib.pref8.gpu [limited baselines 176 input]</option>
            <option value="mopsr.calib.pref16.gpu">mopsr.calib.pref16.gpu [limited baselines 352 input]</option>
            <option value="mopsr.dspsr.cpu">mopsr.dspsr.cpu [tied-array beam]</option>
            <option value="mopsr.dspsr.cpu.cdd">mopsr.dspsr.cpu.cdd [CDD tied-array beam]</option>
            <option value="mopsr.digifil.cpu">mopsr.digifil.cpu [tied-array beam]</option>
            <!--<option value="mopsr.dspsr.cpu.tied">mopsr.dspsr.cpu.tied [tied-array beam **]</option>-->
            <!--<option value="mopsr.digifil.tied">mopsr.digifil.tied [tied-array filterbank **]</option>-->
            <option value="mopsr.bfdsp.gpu">mopsr.bfdsp.gpu [tiled beams]</option>
            <option value="mopsr.dbdisk.channel">mopsr.dbdisk.channel [write baseband]</option>
            <option value="mopsr.null">mopsr.null [discard]</option>
          </select>
        </td>

        <td class='key'>RESOLUTION</td>
        <td class='val'><input type="text" name="resolution" size="4" value="<?echo $this->resolution?>"></td>

        <td class='key'>NBIT</td>
        <td class='val'><input type="text" name="nbit" size="2" value="8" readonly></td>

      </tr>
      <tr>

        <td class='key'>TYPE</td>
        <td class='val'>
          <select name="type">
            <option value="TRACKING">TRACKING</option>
            <option value="TRANSITING">TRANSITING</option>
            <option value="STATIONARY">STATIONARY</option>
          </select>
        </td>

        <td class='key'>BP PROC</td>
        <td class='val'>
          <select name="bp_processing_file">
            <option value="mopsr.heimdall">mopsr.heimdall [search transients]</option>
            <option value="mopsr.dbdisk.beams">mopsr.dbdisk.beams [write baseband]</option>
            <option value="mopsr.null">mopsr.null [discard]</option>
          </select>
        </td>


        <td class='key'>NANT</td>
        <td class='val'><input type="text" id="nant" name="nant" size="3" value="<?echo $this->nant?>"/></td>

        <td class='key'>NDIM</td>
        <td class='val'><input type="text" name="ndim" size="2" value="2" readonly></td>
      

      </tr>
      <tr>

        <td class='key'>CONFIG</td>
        <td class='val'>
          <select name="config">
            <option value="INDIVIDUAL_MODULES">INDIVIDUAL MODULES</option>
            <option value="CORRELATION">CORRELATION</option>
            <option value="TIED_ARRAY_BEAM">TIED ARRAY BEAM</option>
            <option value="FAN_BEAM">FAN BEAM</option>
            <option value="TIED_ARRAY_FAN_BEAM">TIED ARRAY &amp; FAN BEAM</option>
          </select>
        </td>


        <td class='key'>Project ID</td>
        <td class='val'><input type="text" name="project_id" size="4" value="P000"></td>
      
        <td class='key'>OS Ratio</td>
        <td class='val'><input type="text" name="oversampling_ratio" size="16" value="<?echo $this->oversampling_ratio?>"></td>

        <td class='key'>FREQ</td>
        <td class='val'><input type="text" name="centre_frequency" size="12" value="<?echo $this->cfreq?>"></td>

      </tr>
  
      <tr>

        <td class='key'>OBSERVER</td>
        <td class='val'><input type="text" name="observer" size="6" value="AJ"></td>

        <td class='key'>RA</td>
        <td class='val'><input type="text" id="ra" name="ra" size="12" value="08:35:20.61149" readonly></td>

        <td class='key'>NCHAN</td>
        <td class='val'><input type="text" name="nchan" size="3" value="<?echo $this->nchan?>"></td>

        <td class='key'>DSB</td>
        <td class='val'><input type="text" name="dsb" size="1" value="<?echo $this->dsb?>"></td>

      </tr>

      <tr>
        <td class='key'>TOBS</td>
        <td class='val'><input type="text" name="tobs" size="6" value=""></td>

        <td class='key'>DEC</td>
        <td class='val'><input type="text" id="dec" name="dec" size="12" value="-45:10:34.8751" readonly></td>

        <td class='key'>TSAMP</td>
        <td class='val'><input type="text" name="sampling_time" size="12" value="<?echo $this->sampling_time?>"></td>

        <td class='key'>CHANBW</td>
        <td class='val'><input type="text" name="channel_bandwidth" size="12" value="<?echo $this->channel_bandwidth?>"></td>

      </tr>

      <tr>
        <td class='key'>MD Angle</td>
        <td class='val'><input type="text" name="md_angle" size="8" value="0.0"> [degrees]</td>

        <td class='key'>NS Tilt</td>
        <td class='val'><input type="text" name="ns_tilt" size="8" value="0.0"> [degrees]</td>

        <td colspan=4></td>

      </tr>

      <tr>
        <td colspan=8><hr></td>
      </tr>
      
      <tr>
        <td colspan=4>
          <div class="btns" style='text-align: center'>
            <a href="javascript:prepareButton()"  class="btn" > <span>Prepare</span> </a>
            <a href="javascript:startButton()"  class="btn" > <span>Start</span> </a>
            <a href="javascript:stopButton()"  class="btn" > <span>Stop</span> </a>
            <a href="javascript:queryButton()"  class="btn" > <span>Query</span> </a>
          </div>
        </td>
        <td colspan=4 style='text-align: right;'>
          <font size="-1">* has no effect on MOPSR, for future use</font>
        </td>
    </table>
    <input type="hidden" id="command" name="command" value="">
    </form>
<?
    $this->closeBlockHeader();

    echo "<br/>\n";

    // have a separate frame for the output from the TMC interface
    $this->openBlockHeader("TMC Interface");
?>
    <iframe name="tmc_interface" src="" width=100% frameborder=0 height='350px'></iframe>
<?
    $this->closeBlockHeader();
  }

  function printTMCResponse($get)
  {

    // Open a connection to the TMC interface script
    $host = $this->inst->config["TMC_INTERFACE_HOST"];
    $port = $this->inst->config["TMC_INTERFACE_PORT"];
    $sock = 0;

    echo "<html>\n";
    echo "<head>\n";
    for ($i=0; $i<count($this->css); $i++)
      echo "   <link rel='stylesheet' type='text/css' href='".$this->css[$i]."'>\n";
    echo "</head>\n";

    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<mpsr_tmc_message>\n";

    if ($get["command"] == "stop")
    {
      $xml .= "<command>stop</command>\n";
    }
    else if ($get["command"] == "prepare")
    {
      $xml .= "<command>prepare</command>\n";
      $xml .= "<source_parameters>\n";
      $xml .= "  <name epoch='J2000'>".$get["source"]."</name>\n";
      $xml .=   "<ra units='hh:mm:ss'>".$get["ra"]."</ra>\n";
      $xml .=   "<dec units='hh:mm:ss'>".$get["dec"]."</dec>\n";
      $xml .=   "<ns_tilt units='degrees'>".$get["ns_tilt"]."</ns_tilt>\n";
      $xml .=   "<md_angle units='degrees'>".$get["md_angle"]."</md_angle>\n";
      $xml .= "</source_parameters>\n";
      $xml .= "<signal_parameters>\n";
      $xml .=   "<nchan>".$get["nchan"]."</nchan>\n";
      $xml .=   "<nbit>".$get["nbit"]."</nbit>\n";
      $xml .=   "<ndim>".$get["ndim"]."</ndim>\n";
      $xml .=   "<npol>".$get["npol"]."</npol>\n";
      $xml .=   "<nant>".$get["nant"]."</nant>\n";
      $xml .=   "<bandwidth units='MHz'>".$get["bandwidth"]."</bandwidth>\n";
      $xml .=   "<centre_frequency units='MHz'>".$get["centre_frequency"]."</centre_frequency>\n";
      $xml .= "</signal_parameters>\n";
      $xml .= "<pfb_parameters>\n";
      $xml .=   "<oversampling_ratio>".$get["oversampling_ratio"]."</oversampling_ratio>\n";
      $xml .=   "<sampling_time units='microseconds'>".$get["sampling_time"]."</sampling_time>\n";
      $xml .=   "<channel_bandwidth units='MHz'>".$get["channel_bandwidth"]."</channel_bandwidth>\n";
      $xml .=   "<dual_sideband>".$get["dsb"]."</dual_sideband>\n";
      $xml .=   "<resolution units='bytes'>".$get["resolution"]."</resolution>\n";
      $xml .= "</pfb_parameters>\n";
      $xml .= "<observation_parameters>\n";
      $xml .=   "<observer>".$get["observer"]."</observer>\n";
      $xml .=   "<aq_processing_file>".$get["aq_processing_file"]."</aq_processing_file>\n";
      $xml .=   "<bf_processing_file>".$get["bf_processing_file"]."</bf_processing_file>\n";
      $xml .=   "<bp_processing_file>".$get["bp_processing_file"]."</bp_processing_file>\n";
      $xml .=   "<mode>".$get["mode"]."</mode>\n";
      $xml .=   "<project_id>".$get["project_id"]."</project_id>\n";
      $xml .=   "<tobs>".$get["tobs"]."</tobs>\n";
      $xml .=   "<type>".$get["type"]."</type>\n";
      $xml .=   "<config>".$get["config"]."</config>\n";
      $xml .= "</observation_parameters>\n";
    }
    else if ($get["command"] == "start")
    {
      $xml .= "<command>start</command>\n";
    }
    else if ($get["command"] == "query")
    {
      $xml .= "<command>query</command>\n";
    }
    else
    {
      $xml .= "<command>ignore</command>\n";
    }

    $xml .= "</mpsr_tmc_message>\r\n";

?>
</head>
<body>
<table border=1 width='100%'>
 <tr>
  <th>Command</th>
  <th>Response</th>
 </tr>
<?
    list ($sock,$message) = openSocket($host,$port,2);
    if (!($sock)) {
      $this->printTR("Error: opening socket to TMC interface [".$host.":".$port."]: ".$message, "");
      $this->printTF();
      $this->printFooter();
      return;
    }

    $html_cmd = str_replace("<", "[", $xml);
    $html_cmd = str_replace(">", "]", $html_cmd);
    $html_cmd = str_replace("\n", "<br/>", $html_cmd);

    $xml = str_replace("\n", "", $xml);

    $this->printTR ("Sending", $html_cmd);
    socketWrite ($sock, $xml."\r\n");

    $xml = "";
    list ($result, $xml) = socketRead ($sock);

    $html_response = str_replace("><", "]\n[", $xml);
    $html_response = str_replace("<", "[", $html_response);
    $html_response = str_replace(">", "]", $html_response);
    $html_response = str_replace("\n", "<br/>", $html_response);

    $this->printTR ("Received", $html_response);
    $this->printTF();
    $this->printFooter ();

    socket_close($sock);
    return;
  }

  function printTR($left, $right) {
    echo " <tr>\n";
    echo "  <td>".$left."</td>\n";
    echo "  <td>".$right."</td>\n";
    echo " </tr>\n";
    echo '<script type="text/javascript">self.scrollBy(0,100);</script>';
    flush();
  }

  function printFooter() {
    echo "</body>\n";
    echo "</html>\n";
  }

  function printTF() {
    echo "</table>\n";
  }

}

if (isset($_GET["command"])) {
  $obj = new tmc_simulator();
  $obj->printTMCResponse($_GET);
} else {
  handleDirect("tmc_simulator");
}
