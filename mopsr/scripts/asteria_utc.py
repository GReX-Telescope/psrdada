#!/usr/bin/env python

# Stefan Oslowski, Chris Flynn

from argparse import ArgumentParser
import MySQLdb as sql
import glob
import subprocess
import sys

from math import sqrt
import time
from datetime import datetime
from os.path import getmtime

import re
import numpy as np

import ephem
from math import degrees

"""
This script goes through a list of recent observations and if they weren't yet
processed it will extract basic metadata about them. It then stores the results
in a database which can be later used by a php script to display the results on
a webpage
"""

prepDBsQuery = '''
DROP TABLE IF EXISTS Pulsars;
DROP TABLE IF EXISTS TB_Obs;
DROP TABLE IF EXISTS TB_Headers;
DROP TABLE IF EXISTS Infos;
DROP TABLE IF EXISTS UTCs;
DROP TABLE IF EXISTS Updates;

CREATE TABLE Pulsars (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
  name TEXT UNIQUE,
  dm REAL,
  period REAL,
  max_snr_in5min REAL DEFAULT 0.0,
  max_snr_obs_id INTEGER
);

CREATE TABLE TB_Obs (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
  psr_id INTEGER,
  utc_id INTEGER,
  bw REAL,
  freq REAL,
  nbin INTEGER,
  MJD_start TEXT,
  snr REAL,
  tint REAL,
  UNIQUE(psr_id, utc_id)
);

CREATE TABLE TB_Headers (
id INTEGER NOT NULL PRIMARY KEY UNIQUE,
ANTENNA_WEIGHTS  INTEGER,
AQ_PROC_FILE     TEXT,
BF_PROC_FILE     TEXT,
BP_PROC_FILE     TEXT,
BW               REAL,
BYTES_PER_SECOND INTEGER,
CHAN_OFFSET      INTEGER,
CONFIG           TEXT,
CORR_ENABLED     INTEGER,
DEC              TEXT,
DELAY_CORRECTED  INTEGER,
DELAY_TRACKING   INTEGER,
DSB              INTEGER,
FB_BEAM_SPACING  REAL,
FB_ENABLED       INTEGER,
FILE_SIZE        INTEGER,
FREQ             REAL,
HDR_SIZE         INTEGER,
HDR_VERSION      REAL,
INSTRUMENT       TEXT,
MB_ENABLED       INTEGER,
MD_ANGLE         INTEGER,
MODE             TEXT,
NANT             INTEGER,
NBEAM            INTEGER,
NBIT             INTEGER,
NCHAN            INTEGER,
NDIM             INTEGER,
NPOL             INTEGER,
NS_TILT          REAL,
OBSERVER         TEXT,
OBSERVING_TYPE   TEXT,
OBS_OFFSET       INTEGER,
[ORDER]          TEXT,
OSAMP_RATIO      INTEGER,
PFB_ID           TEXT,
PHASE_CORRECTED  INTEGER,
PID              TEXT,
PKT_START        BIGINT,
PROC_FILE        TEXT,
RA               TEXT,
RANKED_MODULES   TEXT,
RESOLUTION       INTEGER,
SOURCE           TEXT,
TB0_ENABLED      INTEGER,
TB1_ENABLED      INTEGER,
TB2_ENABLED      INTEGER,
TB3_ENABLED      INTEGER,
TELESCOPE        TEXT,
TOBS             INTEGER,
TRACKING         INTEGER,
TSAMP            REAL,
UT1_OFFSET       REAL,
UTC_START        TEXT,
UTC_STOP         TEXT
);

CREATE TABLE IF NOT EXISTS `Infos` (
  id INTEGER NOT NULL PRIMARY KEY AUTO_INCREMENT UNIQUE,
  ANTENNA_WEIGHTS  INTEGER,
  AQ_PROC_FILE     TEXT,
  BW               REAL,
  CONFIG           TEXT,
  CORR_ENABLED     INTEGER,
  `DEC`            TEXT,
  DELAY_TRACKING   INTEGER,
  FB_ENABLED       INTEGER,
  FB_IMG           TEXT,
  FREQ             REAL,
  `INT`            INTEGER,
  MB_ENABLED       INTEGER,
  MD_ANGLE         INTEGER,
  MODE             TEXT,
  NANT             INTEGER UNSIGNED,
  NBIT             INTEGER UNSIGNED,
  NCHAN            INTEGER UNSIGNED,
  NDIM             INTEGER UNSIGNED,
  NPOL             INTEGER UNSIGNED,
  NS_TILT          REAL,
  NUM_PWC          INTEGER,
  OBSERVER         TEXT,
  PID              TEXT,
  RA               TEXT,
  RFI_MITIGATION   INTEGER,
  SOURCE           TEXT,
  TB0_ENABLED      INTEGER,
  TB1_ENABLED      INTEGER,
  TB2_ENABLED      INTEGER,
  TB3_ENABLED      INTEGER,
  TB0_IMAGE        TEXT,
  TB1_IMAGE        TEXT,
  TB2_IMAGE        TEXT,
  TB3_IMAGE        TEXT,
  TB0_SOURCE       TEXT,
  TB1_SOURCE       TEXT,
  TB2_SOURCE       TEXT,
  TB3_SOURCE       TEXT,
  TRACKING         INTEGER,
  UTC_START        TEXT
);

CREATE TABLE UTCs (
  id BIGINT NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
  utc TEXT UNIQUE
);

CREATE TABLE Updates (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
  date TEXT
);
INSERT INTO Updates (date) VALUES ("never");

CREATE TABLE IF NOT EXISTS `Cal_solutions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `utc_id` bigint(20) DEFAULT NULL,
  `solution_date` datetime,
  `ref_ant` int(11),
  `best_ants` text,
  `script_version_id` TEXT,
  PRIMARY KEY (`id`)
);

CREATE TABLE IF NOT EXISTS `Cal_phases` (
  `utc_id` int(11),
  `solution_id` int(11),
  `ant_id` int,
  `pfb_id` int,
  `delay` REAL,
  `phase` REAL,
  `weight` REAL,
  `weight_err` REAL,
  `SEFD` REAL,
  `snr` REAL,
  PRIMARY KEY (`utc_id`, `solution_id`, `ant_id`)
);

CREATE TABLE IF NOT EXISTS `Antennas` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(6),
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
);

CREATE TABLE IF NOT EXISTS `PFBs` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(7),
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
);

CREATE TABLE IF NOT EXISTS `Cal_script_versions` (
  `id` int NOT NULL AUTO_INCREMENT,
  `version` varchar(40),
  PRIMARY KEY (`id`),
  UNIQUE KEY `version` (`version`)
);
'''

get_id_DM_period_SNR = '''
SELECT id, dm, period, max_snr_in5min FROM Pulsars WHERE name = %s
'''

get_CORR_SOURCE_query = '''
SELECT CORR_SOURCE FROM Infos WHERE utc_id = %s
'''

set_name_DM_period = '''
INSERT INTO Pulsars (name, dm, period)
  VALUES ( %s, %s, %s )
'''

set_basic_meta_Obs = '''
INSERT IGNORE INTO TB_Obs (psr_id, utc_id, snr, tint)
  VALUES (%s, %s, %s, %s)
'''

set_all_meta_Obs = '''
INSERT INTO TB_Obs (psr_id, utc_id, snr, tint, bw, freq, nbin, MJD_start)
  VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE snr = %s,
  tint = %s, bw = %s, freq = %s, nbin = %s, MJD_start = %s
'''

update_all_meta_Obs = '''
UPDATE TB_Obs SET psr_id = %s, snr = %s, tint = %s, bw = %s, freq = %s,
nbin = %s, MJD_start = %s WHERE id = %s
'''

check_if_Obs_processed = '''
SELECT COUNT(*) FROM TB_Obs
  WHERE psr_id = %s AND utc_id = %s
'''

insert_solution_meta = '''
INSERT INTO Cal_solutions (utc_id, solution_date, ref_ant, best_ants,
  script_version_id) VALUES (%s, %s, %s, %s, %s)
'''

insert_per_ant_solution = '''
INSERT INTO Cal_phases (utc_id, solution_id, ant_id, pfb_id, delay, phase,
  weight, weight_err, SEFD, snr) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
  %s, %s)
'''

insert_annot = '''
UPDATE UTCs SET annotation = %s
  WHERE id = %s
'''

insert_state = '''
UPDATE UTCs SET state_id  = (SELECT id FROM States WHERE state = %s)
  WHERE id = %s
'''

# in most cases the UTC will be new hence probably insert or
# ignore is more efficient than first checking.
# No longer true as MySQL doesn't allow unique keyword on TEXT
insert_utc = '''
INSERT IGNORE INTO UTCs (utc)
  VALUES (%s)
'''

insert_ant = '''
INSERT IGNORE INTO Antennas (name)
  VALUES (%s)
'''

insert_pfb = '''
INSERT IGNORE INTO PFBs (name)
  VALUES (%s)
'''

insert_scriptid = '''
INSERT IGNORE INTO Cal_script_versions(version)
  VALUES (%s)
'''

update_utc_ts = '''
UPDATE UTCs SET utc_ts = TIMESTAMP(%s) WHERE id = %s
'''

get_utcid = '''
SELECT id from UTCs WHERE utc = %s
'''

get_solution_id = '''
SELECT id FROM Cal_solutions WHERE solution_date = %s AND utc_id = %s
'''

get_antid = '''
SELECT id FROM Antennas WHERE name = %s
'''

get_scriptid = '''
SELECT id FROM Cal_script_versions WHERE version = %s
'''

get_pfbid = '''
SELECT id FROM PFBs WHERE name = %s
'''

get_infosid = '''
SELECT id from Infos WHERE utc_id = %s
'''


def parse_cfg(cfg_file, tags=None):
    """Written by W. Farah, with a slight tweak by S. Oslowski"""
    """Function that returns config file with given tags as dictionar

    Parameters
    ----------
    cfg_file : str
        full directory to config file
    tags : list
        list of tags to search the cgf_file

    Returns
    -------
    config_dict : dict
        dictionary with keys given in tags, and values
        extracted from cfg_file. If one tag doesn't exist,
        value corresponded will be None, else value is of
        type str, or list if multiple values exist for
        same key.
    """
    if tags is None:
        tags = []
        with open(cfg_file) as o:
            for line in o:
                if line[0] in ["\n", "#"]:
                    continue
                tags.append(line.split()[0])
    config_dict = {}
    with open(cfg_file) as o:
        for line in o:
            if line[0] in ["\n", "#"]:
                continue
            for tag in tags:
                if tag in line:
                    i = line.split()
                    if tag != i[0]:
                        continue
                    config_dict[tag] = []
                    for ii in i[1:]:
                        if "#" in ii:
                            break
                        config_dict[tag].append(ii)
                    if len(config_dict[tag]) == 1:
                        config_dict[tag] = config_dict[tag][0]
                    tags.remove(tag)
    for tag in tags:
        # logging.warning("Couldn't parse <"+tag+"> from "+cfg_file)
        config_dict[tag] = None
    return config_dict


def insert_psr_into_DB(psr, cur, conn):
    cmd = "/home/dada/linux_64/bin/psrcat -all -nohead -nonumber -c 'DM P0' "\
            + psr
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=True)
    (psrcat_out, psrcat_err) = process.communicate()
    if psrcat_err:
        print "error while running psrcat"
        print psrcat_err
        sys.exit(process.returncode)
    else:
        try:
            DM = psrcat_out.split()[0]
            period = psrcat_out.split()[3]
            cur.execute(set_name_DM_period, (psr, DM, period, ))
            conn.commit()
        except IndexError as ie:
            print "Index error in insert_psr_into_DB"
            print ie
            print "Tried:", cmd
            print psrcat_out
            print psrcat_err


def extract_snr_tint_bw_freq_nbin_int0mjd(dir, archive_fn, verbose):
    archive = glob.glob(archive_fn)
    if archive:
        if verbose:
            print "Found", archive_fn
        cmd = ['/home/dada/linux_64/bin/psrstat', '-c',
               'snr=pdmp,snr,length,bw,freq,nbin,int[0]:mjd',
               '-qQ', '-j', 'DFTp', archive_fn]
        if verbose:
            print "Running cmd:", cmd
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=False)
        (psrstat_out, psrstat_err) = process.communicate()
        if psrstat_err:
            print "Error while running psrstat"
            print psrstat_err
            sys.exit(process.returncode)
        out = psrstat_out.split()
        return float(out[0]), float(out[1]), float(out[2]), float(out[3]),\
            int(out[4]), out[5]
    else:
        if verbose:
            print "No archive present for", dir
        return -1., -1.


def insert_obs_into_DB(psr, UTC, snr, tint, cur):
    cur.execute('SELECT id FROM Pulsars WHERE name = %s', [psr, ])
    psr_id = cur.fetchone()[0]

    cur.execute(set_basic_meta_Obs, [psr_id, utc_id, snr, tint, ])
    conn.commit()


def insert_full_obs_into_DB(id_last, psr, UTC, snr, tint, bw, freq, nbin,
                            mjd_start, cur):
    cur.execute('SELECT id FROM Pulsars WHERE name = %s', [psr, ])
    psr_id = cur.fetchone()[0]

    cur.execute(update_all_meta_Obs, [psr_id, snr, tint, bw, freq, nbin,
                mjd_start, id_last])
    conn.commit


def insert_ant_into_DB(ant, cur, verbose):
    cur.execute(get_antid, [ant, ])
    answer = cur.fetchone()
    if not answer:
        if verbose:
            print "ant", ant, "not yet in DB"
        cur.execute(insert_ant, [ant, ])
        cur.execute(get_antid, [ant, ])
        answer = cur.fetchone()
    ant_id = answer[0]
    return ant_id


def insert_pfb_into_DB(pfb, cur, verbose):
    cur.execute(get_pfbid, [pfb, ])
    answer = cur.fetchone()
    if not answer:
        if verbose:
            print "pfb", pfb, "not yet in DB"
        cur.execute(insert_pfb, [pfb, ])
        cur.execute(get_pfbid, [pfb, ])
        answer = cur.fetchone()
    pfb_id = answer[0]
    return pfb_id


def insert_script_into_DB(script_version, cur, verbose):
    cur.execute(get_scriptid, [script_version, ])
    answer = cur.fetchone()
    if not answer:
        if verbose:
            print "script", script_version, "not yet in DB"
        cur.execute(insert_scriptid, [script_version, ])
        cur.execute(get_scriptid, [script_version, ])
        answer = cur.fetchone()
    script_id = answer[0]
    return script_id


def insert_solution_into_DB(utc_id, solution_date, ref_ant, best_ants,
                            script_version_id):
    cur.execute(insert_solution_meta, [utc_id, solution_date, ref_ant,
                best_ants, script_version_id])
    cur.execute(get_solution_id, [solution_date, utc_id, ])
    return cur.fetchone()[0]


def insert_utc_into_DB(UTC, cur, verbose):
    cur.execute(get_utcid, [UTC, ])
    answer = cur.fetchone()
    if not answer:
        if verbose:
            print "UTC not yet in DB"
        cur.execute(insert_utc, (UTC, ))
        cur.execute(get_utcid, [UTC, ])
        answer = cur.fetchone()
    utc_id = answer[0]
    cur.execute(update_utc_ts, [UTC, utc_id, ])
    return utc_id


def insert_infos_into_DB(top_dir, utc, utc_id, cur, config_dir, verbose):
    cur.execute(get_infosid, [utc_id, ])
    answer = cur.fetchone()
    if not answer:
        if verbose:
            print "Infos not yet in DB for", utc
        parse_infos(top_dir, utc, utc_id, cur, config_dir, verbose)
    cur.execute(get_infosid, [utc_id, ])
    answer = cur.fetchone()
    if not answer:
        print "Failed to inject Infos for", utc
        return -1
    else:
        return answer[0]


def replace_max_snr(psr, snr, utc_id, cur, verbose):
    cur.execute('UPDATE Pulsars SET max_snr_in5min = %s, max_snr_obs_id = %s'
                + ' WHERE name = %s', [snr, utc_id, psr, ])


def rescale_snr_to_5min(snr, tint):
    if tint > 0:
        return snr * sqrt(300./tint)
    else:
        return -1.


def ingest_calib_out(calib_out_fn, UTC, utc_id, ants, pfb, dels, phs, w, w_e,
                     sefds, snrs, cur, verbose):
    script_version, ref_ant, best_ants =\
            parse_calib_out_header(calib_out_fn, verbose)
    ref_ant_id = insert_ant_into_DB(ref_ant, cur, verbose)
    solution_date = str(datetime.utcfromtimestamp(getmtime(calib_out_fn)))
    script_version_id = insert_script_into_DB(script_version, cur, verbose)
    solution_id = insert_solution_into_DB(utc_id, solution_date, ref_ant_id,
                                          best_ants, script_version_id)

    for i in xrange(len(ants)):
        ant_id = insert_ant_into_DB(ants[i], cur, verbose)
        pfb_id = insert_pfb_into_DB(pfb[i], cur, verbose)
        cur.execute(insert_per_ant_solution, [utc_id, solution_id, ant_id,
                    pfb_id, dels[i], phs[i], w[i], w_e[i], sefds[i],
                    snrs[i], ])

    return 0


# def insert_phases_into_DB(utc_id, solution_id, ant_id, pfb_id, delay, phae,
#                           weigh, weight_error, sefd, snr):
#     return 0


def parse_calib_out_header(calib_out_fn, verbose):
    script_version = "unknown"
    ref_ant = "-1"
    best_ants = ""
    try:
        with open(calib_out_fn, "r") as fh:
            line = fh.readline().rstrip("\n")
            line_split = line.split(" ")
            if len(line_split) == 4 and line_split[0] == "#calib.py_version":
                script_version = line_split[2]

            line = fh.readline().rstrip("\n")
            line_split = line.split(" ")
            if len(line_split) == 2 and line_split[0] == "#ref":
                ref_ant = line_split[1]

            line = fh.readline().rstrip("\n")
            line_split = line.split(" ")
            if len(line_split) > 2 and line_split[0] == "#best":
                for iant in xrange(2, len(line_split)):
                    if iant == 2:
                        best_ants = line_split[iant]
                    else:
                        best_ants = best_ants + " " + line_split[iant]

    except Exception:
        print "parse_calib_out_header: couldn't read calib.out", calib_out_fn
    return script_version, ref_ant, best_ants


def parse_calib_out(top_dir, UTC, utc_id, cal, cur, verbose):
    calib_out_fn = top_dir + "/" + cal + "/calib.out"
    calib_out = glob.glob(calib_out_fn)
    if len(calib_out) == 1:
        ants, pfb, dels, phs, w, w_e, sefds, snrs = \
            np.genfromtxt(calib_out_fn, dtype="str", unpack=True)
        return ingest_calib_out(calib_out_fn, UTC, utc_id, ants, pfb, dels,
                                phs, w, w_e, sefds, snrs, cur, verbose)
    else:
        print "Couldn't find calib.out for", UTC, "checked", calib_out_fn
        return -1


def get_CORR_source(utc_id, cur, verbose):
    cur.execute(get_CORR_SOURCE_query, [utc_id, ])
    answer = cur.fetchone()
    return answer[0]


def parse_infos(top_dir, UTC, utc_id, cur, config_dir, verbose):
    obs_info_fns = glob.glob(top_dir + "/obs.info")
    if len(obs_info_fns) == 1:
        infos = parse_cfg(obs_info_fns[0])
        query = 'INSERT INTO Infos ('
        values = ""

        accepted_keys = np.loadtxt(config_dir + "/obs.info_params.only_common",
                                   dtype='str')
        for key in infos.keys():
            try:
                if key in accepted_keys:
                    query += "`" + key + "`,"
                    # handle missing entries and strings with spaces:
                    if isinstance(infos[key], list):
                        if not infos[key]:  # no entry for this key
                            values += '"",'
                        else:  # flatten the list
                            values += '"' + " ".join(infos[key]) + '",'
                    elif infos[key].lower() == "true":
                        values += '1,'
                    elif infos[key].lower() == "false":
                        values += '0,'
                    else:
                        string_re = re.search('[a-zA-Z]', infos[key])
                        if string_re or ":" in infos[key] or "," in infos[key]:
                            values += '"' + infos[key] + '",'
                        else:
                            values += infos[key] + ','
            except Exception as e:
                print e
                print "key:", key
                print "value:", infos[key]

        query += "utc_id)"
        values += str(utc_id) + ")"

        query += ' VALUES (' + values
        try:
            cur.execute(query)
        except (sql.OperationalError, sql.ProgrammingError) as err:
            print "got sql error for", UTC
            print err
            print query
            with open("sqlite_errors.log", "a") as fh:
                fh.write(" " + UTC + " " + str(sys.exc_info()[1])+"\n")
                fh.write(query + "\n")
            sys.exit(-1)

        for psr_id in range(4):
            key_en = "TB%d_ENABLED" % psr_id
            key_source = "TB%d_SOURCE" % psr_id
            if infos[key_en] == "true":
                psr = infos[key_source]
                parse_psr_TB(psr, top_dir, utc_id, cur, config_dir, verbose)

        src = ""
        if "CORR_ENABLED" in infos:
            if infos["CORR_ENABLED"] == "true":
                if "CORR_SOURCE" in infos:
                    src = infos["CORR_SOURCE"]
                    parse_header(src, src, top_dir, utc_id, cur, config_dir
                                 + "/obs.corr_header_params.only_common",
                                 "Corr_Obs", "Corr_Headers", verbose)
                else:
                    print UTC, "missing CORR_SOURCE"
        if "FB_ENABLED" in infos:
            if infos["FB_ENABLED"] == "true":
                if "SOURCE" in infos:
                    src = infos["SOURCE"]
                FB_Obs_id = parse_header(src, "FB", top_dir, utc_id, cur,
                                         config_dir
                                         + "/obs.fb_header_params.only_common",
                                         "FB_Obs", "FB_Headers", verbose)
                set_galb(FB_Obs_id, utc_id, cur, verbose)
        if "MB_ENABLED" in infos:
            if infos["MB_ENABLED"] == "true":
                if "SOURCE" in infos:
                    src = infos["SOURCE"]
                parse_header(src, "FB", top_dir, utc_id, cur, config_dir
                             + "/obs.mb_header_params.only_common", "MB_Obs",
                             "MB_Headers", verbose)
        return 0

    else:
        print "More than one or no obs.info found for", UTC, "in", top_dir
        return -1


def set_galb(FB_Obs_id, utc_id, cur, verbose):
    q = "SELECT FB_Headers.RA, FB_Headers.DEC FROM FB_Headers JOIN FB_Obs ON"\
         + " FB_Headers.utc_id = FB_Obs.utc_id WHERE FB_Obs.id = %s"
    cur.execute(q, [FB_Obs_id, ])
    answer = cur.fetchone()
    RA = answer[0]
    DEC = answer[1]

    radec = ephem.Equatorial(RA, DEC)
    g = ephem.Galactic(radec)
    gb = degrees(g.lat)

    q = "UPDATE FB_Obs SET gb=%s WHERE id = %s"
    cur.execute(q, [gb, FB_Obs_id, ])
    return 0


def parse_annotation(top_dir, utc, utc_id, cur, verbose):
    if verbose:
        print "parsing obs.txt for", utc
    with open(top_dir + "/obs.txt") as fh:
            annot = fh.read()
            try:
                cur.execute(insert_annot, [annot, utc_id])
            except (sql.OperationalError, sql.ProgrammingError) as err:
                print "got sql err while trying to insert annotion for", utc
                print err
                with open("sqlite_errors.log", "a") as fh:
                    fh.write(utc + "\n")
                    fh.write(err + "\n")


def parse_state(top_dir, utc, utc_id, cur, verbose):
    state = "unknown"
    for possible_state in ["processing", "finished", "transferred",
                           "completed", "failed"]:
        if len(glob.glob(top_dir + "/obs." + possible_state)) > 0:
            state = possible_state
            break
    try:
        cur.execute(insert_state, [state, utc_id])
    except (sql.OperationalError, sql.ProgrammingError) as err:
        print "got sql err while trying to insert annotion for", utc
        print err
        with open("sqlite_errors.log", "a") as fh:
            fh.write(utc + "\n")
            fh.write(err + "\n")


def parse_psr_TB(psr, top_dir, utc_dir, cur, config_dir, verbose):
    _dir = top_dir + "/" + psr + "/"
    cur.execute(get_id_DM_period_SNR, [psr])
    answer = cur.fetchone()
    if not answer:
        if verbose:
            print "Pulsar not yet in DB, extracting DM, period"
        insert_psr_into_DB(psr, cur, conn)
    cur.execute(get_id_DM_period_SNR, [psr])
    answer = cur.fetchone()
    # psr_id = answer[0]
    max_snr = answer[3]
    for psrfile in glob.glob(_dir + "/" + psr + "_t.tot"):
        TB_Obs_id = parse_header(psr, psr, top_dir, utc_id, cur, config_dir
                                 + "/obs.tb_header_params.only_common",
                                 "TB_Obs", "TB_Headers", verbose)
        snr, tint, bw, freq, nbin, int0mjd = \
            extract_snr_tint_bw_freq_nbin_int0mjd(_dir, psrfile, verbose)
        insert_full_obs_into_DB(TB_Obs_id, psr, utc_id, snr, tint, bw, freq,
                                nbin, int0mjd, cur)
        snr_5min = rescale_snr_to_5min(snr, tint)
        if verbose:
            print "max_snrs", max_snr, tint, snr_5min
        if tint > 60 and snr_5min > max_snr:
                if verbose:
                    print "Replacing max SNR", psr, max_snr, utc_id
                replace_max_snr(psr, snr_5min, utc_id, cur, verbose)
                max_snr = snr_5min

    # Read and handle obs.header if exists:
    # if args.parseHeaders:


def construct_header_query(src, utc_id, fn, allowed_keys_fn, id_table, table,
                           verbose):
    # obs_id_q = 'SELECT MAX(id) from ' + id_table
    # cur.execute(obs_id_q)
    # obs_id = cur.fetchone()[0]

    cfg = parse_cfg(fn)
    # query = 'INSERT IGNORE INTO ' + table + ' (id, utc_id, '
    query = 'INSERT IGNORE INTO ' + table + ' (utc_id, '
    # values = str(obs_id) +"," + str(utc_id) + ","
    values = str(utc_id) + ","

    accepted_keys = np.loadtxt(allowed_keys_fn, dtype='str')
    for key in cfg.keys():
        if key in accepted_keys:
            query += "`" + key + "`,"
            # handle special case of a problem with UT1_OFFSET
            # having two entries on 2016-09-12
            if isinstance(cfg[key], list):
                if key == "UT1_OFFSET":
                    cfg[key] = cfg[key][0]
                else:
                    with open("asteria_error.log", "a") as fh:
                        fh.write("cfg[key] is a list" + src + " " + str(utc_id)
                                 + " " + key)
                    print "cfg[key] is a list " + src + " " + str(utc_id)\
                        + " " + key
                    continue
            if cfg[key].lower() == "true":
                values += '1,'
            elif cfg[key].lower() == "false":
                values += '0,'
            else:
                string_re = re.search('[a-zA-Z]', cfg[key])
                if string_re or ":" in cfg[key] or "," in cfg[key]:
                    values += '"' + cfg[key] + '",'
                else:
                    values += cfg[key] + ','
        else:
            with open("headers_missing.log", "a") as fh:
                fh.write(src + " " + str(utc_id) + " " + key
                         + " not present in DB")

    query = query[0:-1] + ")"  # replace last comma with closing parenthesis
    values = values[0:-1] + ")"  # replace last comma with closing parenthesis
    query += ' VALUES (' + values
    return query


def parse_header(src, src_dir, top_dir, _utc_id, cur, allowed_keys_fn,
                 id_table, table, verbose):
    obs_table_prefill_query = "INSERT INTO " + id_table + " (utc_id) VALUES ("\
                              + str(_utc_id) + ")"
    cur.execute(obs_table_prefill_query)
    id_table_id = cur.lastrowid
    obs_head_fns = glob.glob(top_dir + "/" + src_dir + "/obs.header")
    if len(obs_head_fns) == 1:
        query = construct_header_query(src, _utc_id, obs_head_fns[0],
                                       allowed_keys_fn, id_table, table,
                                       verbose)
        try:
            cur.execute(query)
        except (sql.OperationalError, sql.ProgrammingError) as err:
            print "got sql error for", src, _utc_id
            print err
            with open("sqlite_errors.log", "a") as fh:
                fh.write(src + " " + str(_utc_id) + " "
                         + str(sys.exc_info()[1]) + "\n")
                fh.write(query + "\n")
    return id_table_id


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="Verbose", action="store_true",
                        help="Turn on verbose output")
    parser.add_argument("--data-top-dir", dest="dataTopPath",
                        help="Path to observations",
                        default="/data/mopsr/results/")
    #  parser.add_argument("-m", "--mysql-config-file", dest="mysqlConfig",
    #      help="Path to mysql config file", default="")
    parser.add_argument("-c", "--config-dir", dest="configDir",
                        help="Path to mysql config file", default="")
    parser.add_argument("-U", "--UTC", dest="UTC",
                        help="UTC of the observation to ingest")
    parser.add_argument("-H", "--headers", dest="headers",
                        help="Ingest obs.header")
    parser.add_argument("-C", "--CAL", dest="ingestCal", action="store_true",
                        help="Ingest calibration solution", default=False)

    args = parser.parse_args()

    # Connect to the DB:
    if (len(args.configDir) == 0):
        print "You need to specify the config directory"
        sys.exit(-1)
    elif not args.UTC:
        print "You need to specify the UTC of the observation to ingest"
        sys.exit(-1)

    mysql_cfg = parse_cfg(args.configDir + "/mysql-config.php")
    # The keys and values are a bit funny because
    # we use the php mysql config file
    _host = mysql_cfg['define("MYSQL_HOST",'].replace('");', '').lstrip('"')
    _db = mysql_cfg['define("MYSQL_DB",'].replace('");', '').lstrip('"')
    _db = _db.strip('"')
    _passwd = mysql_cfg['define("MYSQL_PWD",'].replace('");', '').lstrip('"')
    _user = mysql_cfg['define("MYSQL_USER",'].replace('");', '').lstrip('"')
    conn = sql.connect(host=_host, db=_db, passwd=_passwd, user=_user)
    cur = conn.cursor()

    # Find a list of dirs to process
    dirs = glob.glob(args.dataTopPath + "/" + args.UTC)
    if len(dirs) != 1:
        print "Found ", len(dirs), "directories matching the obs, expected 1"
        sys.exit(-1)

    if args.Verbose:
        print "Found ", len(dirs), "dirs to process"
    _dir = dirs[0]
    if args.Verbose:
        print "Processing ", args.UTC

    utc_id = insert_utc_into_DB(args.UTC, cur, args.Verbose)
    if len(glob.glob(_dir + "/obs.txt")) > 0:
        parse_annotation(_dir, args.UTC, utc_id, cur, args.Verbose)
    parse_state(_dir, args.UTC, utc_id, cur, args.Verbose)
    infos_id = insert_infos_into_DB(_dir, args.UTC, utc_id, cur,
                                    args.configDir, args.Verbose)
    if args.ingestCal:
        cal = get_CORR_source(utc_id, cur, args.Verbose)
        if cal is not None:
            parse_calib_out(_dir, args.UTC, utc_id, cal, cur, args.Verbose)
        else:
            print "CORR_SOURCE missing for", args.UTC, "could not ingest cal"

    cur_time = time.strftime('%a %d %b %Y %H:%M %Z')
    cur.execute('UPDATE Updates SET date = %s WHERE id = 1', [cur_time, ])
    conn.commit()
