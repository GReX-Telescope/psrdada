#include "dada_client.h"
#include "dada_hdu.h"
#include "dada_def.h"

#include "ascii_header.h"
#include "daemon.h"
#include "mopsr_def.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <float.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include "futils.h"


#define ID_MAX_LEN 12

int quit_threads = 0;

int compare (const void * a, const void * b);
int compare (const void * a, const void * b)
{
  return ( *(float *)a - *(float *)b );
}

int64_t dbfurbydb_SFT (dada_client_t *, void *, uint64_t, uint64_t);

void usage()
{
  fprintf (stdout,
           "mopsr_dbfurbydb [options] bp_node in_key out_key\n"
	   "		  add furby in SFT ordered float data\n"
           //"              data type change from 32-bit float to 8-bit unsigned int\n"
           " -D path      complete path to the furby database directory\n\t\tDef: /home/dada/furby_databse/"
           " -s           1 transfer, then exit\n"
           " -z           use zero copy transfers\n"
           " -c           copy live data snippets into dada files whenever furbies are added\n"
           " -v           verbose mode\n"
           " in_key       DADA key for input data block\n"
           " out_key      DADA key for output data blocks\n");
}

typedef struct {

  dada_hdu_t *  out_hdu;
  key_t         out_key;
  uint64_t      out_block_size;
  char *        out_block;

  // number of bytes read
  uint64_t bytes_in;
  uint64_t bytes_out;
  unsigned bitrate_factor;

  // verbose output
  int verbose;

  unsigned int nsig;
  unsigned int nant;
  unsigned int nbeam;
  unsigned int nchan;
  unsigned int ndim; 
  unsigned int nbit_in;
  unsigned int nbit_out;
  float bw;

  unsigned quit;
  char order[4];

  char furby_database[256];
  int first_beam;
  int last_beam;
  
  uint64_t passed_samps;
  int furbies_added;
  
  int inj_furbys;
  float furby_BW;
  char **furby_ids;
  int * furby_beams; 
  float * furby_tstamps; 
  int furby_nsamps;
 
  int * furby_samp_indices;
  int * furby_beam_indices;
  int furby_chansamp;
  float * furby_buffer;

  int N_furbys_to_add_here;
  int bad_requests;
  int bp_node;

  FILE **FURBY_ptrs;
  int copy_furby;
  float * tmp_buff;
  int tmp_buff_size;

} mopsr_dbfurbydb_t;


#define DADA_DBFURBYDB_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", 0, 0, 0}

/*! Function that opens the data transfer target */
int dbfurbydb_open (dada_client_t* client)
{
  // the mopsr_dbfurbydb specific data
  mopsr_dbfurbydb_t* ctx = (mopsr_dbfurbydb_t *) client->context;

  // status and error logging facilty
  multilog_t* log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbfurbydb_open()\n");

  char output_order[4];

  // header to copy from in to out
  char * header = 0;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: HDU (key=%x) lock_write on HDU\n", ctx->out_key);
  if (dada_hdu_lock_write (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot lock write DADA HDU (key=%x)\n", ctx->out_key);
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: dada_hdu_lock_write\n");

  if (ascii_header_get (client->header, "BW", "%f", &(ctx->bw)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no BW\n");
    return -1;
  }

  uint64_t nsamps_per_block;
  if(ctx->bitrate_factor !=1)
  {
    multilog(log, LOG_ERR, "open: Bitrate factor has to be = 1 to be able to add furbies\n");
    return -1;
  }

  // get the transfer size (if it is set)
  int64_t transfer_size = 0;
  ascii_header_get (client->header, "TRANSFER_SIZE", "%"PRIi64, &transfer_size);

  // get the number of antenna and beams
  if (ascii_header_get (client->header, "NANT", "%d", &(ctx->nant)) != 1)
  {
    ctx->nant = 1;
  }
  if (ascii_header_get (client->header, "NBEAM", "%d", &(ctx->nbeam)) != 1)
  {
    ctx->nbeam = 1;
  }
  if (ascii_header_get (client->header, "NCHAN", "%u", &(ctx->nchan)) != 1)
  {           
    multilog (log, LOG_ERR, "open: header with no NCHAN\n");
    return -1;                
  }

  if (ctx->nant == 1 && ctx->nbeam > 1)
    ctx->nsig = ctx->nbeam;
  else if (ctx->nant > 1 && ctx->nbeam == 1)
    ctx->nsig = ctx->nant;
  else
  {
    multilog (log, LOG_ERR, "open: cannot add furbies when nant=%d and nbeam=%d\n", ctx->nant, ctx->nbeam);
    return -1;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "open: using nant=%d, nbeam=%d, nsig=%d nchan=%d\n", ctx->nant, ctx->nbeam, ctx->nsig, ctx->nchan);

  if (ascii_header_get (client->header, "NBIT", "%u", &(ctx->nbit_in)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no NBIT\n");
    return -1;
  }

  //ctx->nbit_out = 8;
  ctx ->nbit_out = 32;

  if (ascii_header_get (client->header, "NDIM", "%u", &(ctx->ndim)) != 1)
  {           
    multilog (log, LOG_ERR, "open: header with no NDIM\n");
    return -1;                
  }                             

  if (ascii_header_get (client->header, "ORDER", "%s", &(ctx->order)) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no ORDER\n");
    return -1;
  }
  else
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: ORDER=%s\n", ctx->order);
    if ((strcmp(ctx->order, "SFT") == 0) && client->io_block_function)
    {
      client->io_block_function = dbfurbydb_SFT;
      strcpy (output_order, "SFT");
    }
    else
    {
      multilog (log, LOG_ERR, "open: input ORDER=%s is not supported\n", ctx->order);
      return -1;
    }
  }

  char UTC_START[32];
  if (ascii_header_get (client->header, "UTC_START", "%s", UTC_START) == 1)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "open: UTC_START=%s\n", UTC_START);
  }
  else
  {
    multilog (log, LOG_INFO, "open: UTC_START=UNKNOWN\n");
    if(ctx->copy_furby)
    {
      multilog(log, LOG_ERR, "open: FATAL: Cannot copy live furby data into files if UTC_START is UNKOWN\n");
      return -1;
    }
  }

  uint64_t resolution, new_resolution;
  if (ascii_header_get (client->header, "RESOLUTION", "%"PRIu64, &resolution) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no RESOLUTION\n");
    return -1;
  }

  uint64_t bytes_per_second, new_bytes_per_second;
  if (ascii_header_get (client->header, "BYTES_PER_SECOND", "%"PRIu64, &bytes_per_second) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no BYTES_PER_SECOND\n");
    return -1;
  }

  uint64_t file_size, new_file_size;
  if (ascii_header_get (client->header, "FILE_SIZE", "%"PRIu64, &file_size) != 1)
  {
    multilog (log, LOG_ERR, "open: header with no FILE_SIZE\n");
    return -1;
  }

  float tsamp;
  if (ascii_header_get (client->header, "TSAMP", "%f", &tsamp) != 1)
  {
    multilog (log, LOG_ERR, "open: failed to read TSAMP from header\n");
    return -1;
  }

  float samples_per_second = 1000000.0f/ tsamp;

  // TSAMP origin

  // get the header from the input data block
  uint64_t header_size = ipcbuf_get_bufsz (client->header_block);
  assert( header_size == ipcbuf_get_bufsz (ctx->out_hdu->header_block) );

  header = ipcbuf_get_next_write (ctx->out_hdu->header_block);
  if (!header) 
  {
    multilog (log, LOG_ERR, "open: could not get next header block\n");
    return -1;
  }

  //================================================== VG ================================================


  if (ascii_header_get (client->header, "INJECTED_FURBYS", "%i", &(ctx->inj_furbys)) > 0)
  {
  
    ctx->passed_samps=0;
    ctx->furbies_added=0;
    //-------------------- Reading the config file ---------------------
    char * DADA_ROOT = getenv("DADA_ROOT");
    char cfg_path[128];
    sprintf(&cfg_path[0], "%s/share/mopsr_bp_cornerturn.cfg", DADA_ROOT);
    //sprintf(&cfg_path[0], "%s/mopsr_bp_cornerturn.cfg", "/home/vgupta/Codes/Redesign");


    // Get size of config file, allocate resources and read in file
    int cfg_size = filesize(cfg_path);
    char * bp_cornerturn_cfg = calloc(cfg_size + 1, sizeof *bp_cornerturn_cfg);

    if (fileread(cfg_path, bp_cornerturn_cfg, cfg_size) < 0)
    {
      multilog(log, LOG_ERR, "open: config file (%s) couldn't be read\n", cfg_path);
      return -1;
    }


    char temp[32] = {0};

    // Parse BEAM_FIRST_RECV
    sprintf(&temp[0],"BEAM_FIRST_RECV_%i",ctx->bp_node);
    if (ascii_header_get(bp_cornerturn_cfg, temp, "%i", &(ctx->first_beam)) != 1)
    {
      multilog(log, LOG_ERR,"ERROR: could not parse '%s' from cfg file (%s)\n",
          temp,cfg_path);
      return -1;
    }

    // Parse BEAM_LAST_RECV
    sprintf(&temp[0],"BEAM_LAST_RECV_%i",ctx->bp_node);
    if (ascii_header_get(bp_cornerturn_cfg, temp, "%i", &(ctx->last_beam)) != 1)
    {
      multilog(log, LOG_ERR,"ERROR: could not parse '%s' from cfg file (%s)\n",
          temp,cfg_path);
      return -1;
    }
    //--------------------------------------------------------------------

    if (ctx->verbose)
      multilog(log, LOG_INFO, "open: First_beam is: %i, Last_beam is: %i\n", ctx->first_beam, ctx->last_beam);

    
    if(ctx->verbose)
	multilog(log, LOG_INFO, "open: parsed Injected_furbies = %i\n", ctx->inj_furbys);

   if (ctx->inj_furbys > 0)
   {

    char * f_ids;
    char * f_beams;
    char * f_tstamps;
    f_ids = malloc(sizeof(char) * ctx->inj_furbys * 50);
    f_beams = malloc(sizeof(char) * ctx->inj_furbys * 50);
    f_tstamps = malloc(sizeof(char) * ctx->inj_furbys * 50);

    if (ascii_header_get (client->header, "FURBY_IDS", "%s", f_ids) != 1)
    {
      multilog (log, LOG_ERR, "open: FURBY_INJECTION with no FURBY_IDS\n");
      return -1;
    }
    
    if (ascii_header_get (client->header, "FURBY_BEAMS", "%s", f_beams) != 1)
    {
      multilog (log, LOG_ERR, "open: FURBY_INJECTION with no FURBY_INJECTION_BEAMS\n");
      return -1;
    }

    //if (ascii_header_get (client->header, "FURBY_TSTAMPS", "%s", &(ctx->furby_tstamps)) != 1)
    if (ascii_header_get (client->header, "FURBY_TSTAMPS", "%s", f_tstamps) != 1)
    {
      multilog (log, LOG_ERR, "open: FURBY_INJECTION with no FURBY_INJECTION_TSTAMPS\n");
      return -1;
    }

    //ctx->furby_ids = (int *)malloc(sizeof(int) * ctx->inj_furbys);
    ctx->furby_ids = malloc(sizeof(char*) * ctx->inj_furbys);
    ctx->furby_beams = (int *)malloc(sizeof(int) * ctx->inj_furbys);
    ctx->furby_tstamps = (float *)malloc(sizeof(float) * ctx->inj_furbys);

    int ff = 0;
    char * temp_a;
    char * temp_b;
    char * temp_c;

    if(ctx->verbose)
      multilog(log, LOG_INFO, "open: Parsed furby parameters: \nFurby_IDs:\t%s\nFurby_Beams:\t%s\nFurby_tstamps:\t%s\n",f_ids, f_beams, f_tstamps);

    for (ff=0; ff<ctx->inj_furbys; ff++)
    {
	temp_a = strsep(&(f_ids), ",");
	temp_b = strsep(&(f_beams), ",");
	temp_c = strsep(&(f_tstamps), ",");

	//checking in case there is an extra-comma by mistake (as a replacement of white-space).
	while( strcmp(temp_a, "") == 0)
	{
	  multilog(log, LOG_ERR, "open: Found an empty furby_id at index: %i. Trying to ignore it.\n", ff+1);
	  if(f_ids == NULL)
	  {
	    multilog(log, LOG_ERR, "open: Insufficient furby_ids. Required: %i, found: %i\n", ctx->inj_furbys, ff);
	    return -1;
	  }
	  else
	    temp_a = strsep(&(f_ids), ",");
	}

	while( strcmp(temp_b, "") == 0)
	{
	  multilog(log, LOG_ERR, "open: Found an empty furby_beam at index: %i. Trying to ignore it.\n", ff+1);
	  if(f_beams == NULL)
	  {
	    multilog(log, LOG_ERR, "open: Insufficient furby_beams. Required: %i, found: %i\n", ctx->inj_furbys, ff);
	    return -1;
	  }
	  else
	    temp_b = strsep(&(f_beams), ",");
	}

	while( strcmp(temp_c, "") == 0)
	{
	  //checking in case there is an extra-comma by mistake (as a replacement of white-space).
	  multilog(log, LOG_ERR, "open: Found an empty furby_tstsamp at index: %i. Trying to ignore it.\n", ff+1);
	  if(f_tstamps == NULL)
	  {
	    multilog(log, LOG_ERR, "open: Insufficient furby_tstamps. Required: %i, found: %i\n", ctx->inj_furbys, ff);
	    return -1;
	  }
	  else
	    temp_c = strsep(&(f_tstamps), ",");
	}


        ctx->furby_ids[ff] = malloc((ID_MAX_LEN+1) * sizeof(char));
	strcpy(ctx->furby_ids[ff], temp_a);
        ctx->furby_beams[ff] = atoi(temp_b) -1 ;	//Subtracting one because all other beam_numbers are 0 indexed, while user requests a 1 indexed beam_number.
        ctx->furby_tstamps[ff] = atof(temp_c);
	
	if(ctx->verbose)
	  multilog(log, LOG_INFO, "open: parsed -> ID: %s  ===  BEAM: %i  ===  TSTAMP: %f\n", ctx->furby_ids[ff], ctx->furby_beams[ff]+1, ctx->furby_tstamps[ff]);
    }


    char **ord_f_ids;
    int * ord_f_beams;
    float * ord_f_tstamps;

    ord_f_ids = malloc(sizeof(char *) * ctx->inj_furbys);
    ord_f_beams = (int *)malloc(sizeof(int) * ctx->inj_furbys);
    ord_f_tstamps = (float *)malloc(sizeof(float) * ctx->inj_furbys);

    int kk= 0;
    int smallest = 0;
    for (kk = 0; kk < ctx->inj_furbys; kk++)
    { 
      ord_f_ids[kk] = malloc(sizeof(char) * (ID_MAX_LEN+1));
      smallest = kk;
      for (ff = kk+1; ff<ctx->inj_furbys; ff++)
      {
	if(ctx->furby_tstamps[ff] < ctx->furby_tstamps[smallest])
	    smallest = ff;
      }
      ord_f_beams[kk] = ctx->furby_beams[smallest];  ctx->furby_beams[smallest] = ctx->furby_beams[kk];
      strcpy(ord_f_ids[kk], ctx->furby_ids[smallest]);   strcpy(ctx->furby_ids[smallest], ctx->furby_ids[kk]);
      ord_f_tstamps[kk] = ctx->furby_tstamps[smallest];   ctx->furby_tstamps[smallest] = ctx->furby_tstamps[kk];
    }


    for(ff=0; ff<ctx->inj_furbys; ff++)
    {	
	if( (ord_f_beams[ff] >= ctx->first_beam) & (ord_f_beams[ff] <= ctx->last_beam) )
	    ctx->N_furbys_to_add_here++;
    }

    if (ctx->verbose)
        multilog(log, LOG_INFO, "open: The number of furbies to be added on this bp_node : %i\n", ctx->N_furbys_to_add_here);
    
    if(ctx->N_furbys_to_add_here > 0)
    {
	
	int furby_nchan;
	int furby_nbits;
	char furby_order[4];
	FILE * fu;
	char furby_header[16384];
        int furby_header_size;

	char furby_tmp[1024];
	sprintf(&furby_tmp[0], "%s/furby_%s", ctx->furby_database, ctx->furby_ids[0]);
	
	if(ctx->verbose)
	  multilog(log, LOG_INFO, "open: Opening %s to read the header parameters\n", furby_tmp);

	fu = NULL;
	fu = fopen(furby_tmp, "r");			//reading the header of first furby to be added and assuming all the other furbies have same header parameters.
	if(fu==NULL)
	{
	  multilog(log, LOG_ERR, "open: FATAL: Could not open (%s) to read the header parameters.\n", furby_tmp);
	  return -1;
	}

	fread(furby_header, 16384, 1, fu);
	fclose(fu);

	
        if( (ascii_header_get(furby_header, "NBIT", "%i", &furby_nbits))!= 1)
	{
	    multilog(log, LOG_ERR, "open: furby_header with no 'NBIT'\n");
	    return -1;
	}
	if( (ascii_header_get(furby_header, "BW", "%f", &(ctx->furby_BW)))!= 1)
	{
	    multilog(log, LOG_ERR, "open: furby_header with no 'BW'\n");
	    return -1;
	}

	if( (ascii_header_get(furby_header, "NCHAN", "%i", &furby_nchan)) !=1 )
	{
	    multilog(log, LOG_ERR, "open: furby_header with no 'NCHAN'\n");
	    return -1;
	}

	if( (ascii_header_get(furby_header, "NSAMPS", "%i", &(ctx->furby_nsamps)))!=1)
	{
	    multilog(log, LOG_ERR, "open: furby_header with no 'NSAMPS'\n");
	    return -1;
	}

        if( (ascii_header_get(furby_header, "HDR_SIZE", "%i", &furby_header_size)) !=1)
        {
          multilog(log, LOG_ERR, "open: furby_header with no 'HDR_SIZE'\n");
          return -1;
        }

	ctx->furby_samp_indices = (int *)malloc(sizeof(int) * (ctx->N_furbys_to_add_here+1));
	ctx->furby_beam_indices = (int *)malloc(sizeof(int) * (ctx->N_furbys_to_add_here+1));
	ctx->furby_chansamp = furby_nchan * ctx->furby_nsamps;
	int bytes_per_furby = ctx->furby_chansamp * furby_nbits/8;

	ctx->furby_buffer = (float *)malloc(ctx->N_furbys_to_add_here * bytes_per_furby);

        memset(ctx->furby_beam_indices, 0, sizeof(int) * (ctx->N_furbys_to_add_here+1));
        memset(ctx->furby_samp_indices, 0, sizeof(int) * (ctx->N_furbys_to_add_here+1));
        memset(ctx->furby_buffer, 0, ctx->N_furbys_to_add_here * bytes_per_furby);	//setting all memory to zero, so that any failed attempt to add furby which slips through the checks below will be added as zeros (No additional data) into the live data stream
        //This is keeping in mind that in the worst case, it is better to not have a fake FRB when we want one, rather than have one Fake FRB when we did not expect one.

	if(ctx->furby_buffer == NULL)
	{
	  multilog(log, LOG_ERR, "open: FATAL: Could not allocate memory to load in %i furby(s)\n", ctx->N_furbys_to_add_here);
	  return -1;
	}

	if( (furby_nchan!=320) || (furby_nbits!=32) )
	{
	  multilog(log, LOG_ERR, "open: FATAL: Unsupported format of the furbies to be added: NBIT: %i, NCHAN: %i. Only 320 channel and 32-bit floats are supported currently.\n", furby_nchan, furby_nbits);
	  return -1;
	}

        char fbuffer[128];
        char fdir[128];
        if(ctx->copy_furby)
        {
          ctx->FURBY_ptrs = (FILE **) malloc(sizeof(FILE *) * ctx->N_furbys_to_add_here);
        }

	//loading the furbies to add onto RAM now.
	multilog(log, LOG_INFO, "open: Attempting to load  %i furby(s) onto memory to be added in live data stream\n", ctx->N_furbys_to_add_here);
	
	uint64_t f_index=0;
	int kk = 0;
	for (ff=0; ff<ctx->inj_furbys; ff++)
	{
	if ((ord_f_beams[ff] >= ctx->first_beam) & (ord_f_beams[ff] <= ctx->last_beam) )
	{
	  //get the sample index where to add the furby; ignore furbies with incorrect samples indices.
	  ctx->furby_samp_indices[kk] = (int)(ord_f_tstamps[ff] * samples_per_second); 
	  if( ctx->furby_samp_indices[kk] < (int)(ctx->furby_nsamps/2))		//Cannot add furby in the first 0.5 seconds
	  {
	    multilog(log, LOG_ERR, "open: ERR:: Furby_samp_index(%i) < Furby_nsamps/2 (%i)\nCannot add furby %s.\n", ctx->furby_samp_indices[kk], (int)(ctx->furby_nsamps/2), furby_tmp);
	    ctx->N_furbys_to_add_here--; ctx->bad_requests++;
	    continue;
	    //return -1;
	  }
	  
          nsamps_per_block = ctx->out_block_size / (ctx->nchan * ctx->nsig * ctx->ndim * (ctx->nbit_in/8));
          uint64_t minimum_gap = ctx->furby_nsamps + nsamps_per_block;  //Cannot have two consecutive furbies touching or in the same data block
          if(kk>=1)
          {
            if( (ctx->furby_samp_indices[kk] - ctx->furby_samp_indices[kk-1]) <= minimum_gap ) 		//seperation between furbies has to be more than 1 second.
	    {
	      multilog(log, LOG_ERR, "open: ERR:: Cannot have two coincident injected furbys: requested samp_index: (%i) too close to previous samp_index: (%i). Minimum gap = %"PRIu64" samples\n", ctx->furby_samp_indices[kk], ctx->furby_samp_indices[kk-1], minimum_gap);
              ctx->N_furbys_to_add_here--; ctx->bad_requests++;
	      continue;
	    }
          }

	  sprintf(&furby_tmp[0], "%s/furby_%s", ctx->furby_database, ord_f_ids[ff]);
	  fu = fopen(furby_tmp, "r");
	  if (fu == NULL)
	  {
	    multilog(log, LOG_ERR, "open: ERR:: Could not open %s to read header parameters\n", furby_tmp);
	    ctx->N_furbys_to_add_here--; ctx->bad_requests++;
	    continue;
	  }

	  fread(furby_header, 16384, 1, fu);
	  if( (ascii_header_get(furby_header, "ORDER", "%s", &furby_order))!= 1)
	  {
	    multilog(log, LOG_ERR, "open: ERR:: furby_header (%s) with no 'ORDER'\n", furby_tmp);
	    ctx->N_furbys_to_add_here--; ctx->bad_requests++;
	    continue;
	    //return -1;
	  }

	  if( strcmp(furby_order, "TF")!=0 )
	  {  
	    multilog(log, LOG_ERR, "open: ERR:: The furby has to be in TF order, while %s has %s order\n", furby_tmp, furby_order);
	    ctx->N_furbys_to_add_here--; ctx->bad_requests++;
	    continue;
	    //return -1;
	  }

	  int check = 0;
	  fseek(fu, furby_header_size, SEEK_SET);
	  check = fread(&(ctx->furby_buffer[f_index]), bytes_per_furby, 1, fu);		//reading the furby into memory buffer directly from file

	  if(check != 1)
	  {
	    multilog(log, LOG_ERR, "open: ERR:: Could not read the furby_data from (%s) into memory\n", furby_tmp);
	    ctx->N_furbys_to_add_here--; ctx->bad_requests++;
	    continue;
	    //return -1;
	  }
	  ctx->furby_beam_indices[kk] = ord_f_beams[ff];
	  fclose(fu);
	  f_index += ctx->furby_chansamp;
	  if(ctx->verbose)
	    multilog(log, LOG_INFO, "open: Loaded %s into memory buffer to be added in BEAM_%i at SAMPLE: %i\n", furby_tmp, ctx->furby_beam_indices[kk]+1, ctx->furby_samp_indices[kk]);

          if(ctx->copy_furby)
          {
            //Opening files to copy the inejcted furby + live data.        
            sprintf(fdir, "/data/mopsr/rawdata/BP%02d/%s/FB/BEAM_%03d", ctx->bp_node, UTC_START, ctx->furby_beam_indices[kk]+1);
            sprintf(fbuffer, "mkdir -p %s", fdir);
            system(fbuffer);

            sprintf(fbuffer, "%s/FURBY_%s_B%03d_S%i.dada", fdir, UTC_START, ctx->furby_beam_indices[kk]+1, ctx->furby_samp_indices[kk]);
            multilog(log, LOG_INFO, "open: Opening %s\n", fbuffer);
            ctx->FURBY_ptrs[kk] = fopen(fbuffer, "w");

            if(ctx->FURBY_ptrs[kk] == NULL)
            {
              multilog(log, LOG_ERR, "open: Could not open: %s to write the furby+live data into it\n", fbuffer);
              return -1;
            }
            //Copying the header of the input furby into the header of output FURBY
            fwrite(furby_header, furby_header_size, 1, ctx->FURBY_ptrs[kk]);
          }

	  kk++;
          fu = NULL;
	}
	}
	multilog(log, LOG_INFO, "open: %i furby(s) will be added on this BP node: %i. No. of bad requests: %i.\n", ctx->N_furbys_to_add_here, ctx->bp_node, ctx->bad_requests);

        if(ctx->copy_furby)
        {
          ctx->tmp_buff_size = sizeof(float) * ctx->nchan * nsamps_per_block;
          ctx->tmp_buff = (float *)malloc(ctx->tmp_buff_size);
        }

    }
    else
    {
	multilog(log, LOG_INFO, "open: No furbies to add in the beams on this BP node: %i\n", ctx->bp_node);
    }

   }
  }
  else
  {
    multilog(log, LOG_INFO, "dbfurbydb_open: Could not parse INJECTED_FURBYS from header.\n No furby injection will happen.\n");
    ctx->inj_furbys = 0;
    ctx->N_furbys_to_add_here = 0;
    //multilog(log, LOG_ERR, "open: FATAL:  Could not parse INJECTED_FURBYS from the header");
    //return -1;
  }

  //=========================================================================================================

  // copy the header from the in to the out
  memcpy ( header, client->header, header_size );

  if (ascii_header_set (header, "ORDER", "%s", output_order) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write ORDER=%s to header\n", output_order);
    return -1;
  }

  new_bytes_per_second = bytes_per_second / ctx->bitrate_factor;
  if (ascii_header_set (header, "BYTES_PER_SECOND", "%"PRIu64, new_bytes_per_second) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write BYTES_PER_SECOND=%"PRIu64" to header\n", bytes_per_second);
    return -1;
  }

  new_file_size = file_size / ctx->bitrate_factor;
  if (ascii_header_set (header, "FILE_SIZE", "%"PRIu64, new_file_size) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write FILE_SIZE=%"PRIu64" to header\n", file_size);
    return -1;
  }

  new_resolution = resolution / ctx->bitrate_factor;
  if (ascii_header_set (header, "RESOLUTION", "%"PRIu64, new_resolution) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write RESOLUITON=%"PRIu64" to header\n", new_resolution);
    return -1;
  }

  if (ascii_header_set (header, "NBIT", "%"PRIu16, ctx->nbit_out) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write NBIT=%"PRIu16" to header\n", ctx->nbit_out);
    return -1;
  }

  // TODO fix corner turn to deal with this
  if (ascii_header_set (header, "NBEAM", "%d", ctx->nsig) < 0)
  {
    multilog (log, LOG_ERR, "open: failed to write NBEAM=%d to header\n", ctx->nsig);
    return -1;
  }

  int nant = 1;
  if (ascii_header_set (header, "NANT", "%d", nant) < 0)
  {     
    multilog (log, LOG_ERR, "open: failed to write NBEAM=%d to header\n", ctx->nsig);
    return -1;    
  }                 

  // mark the outgoing header as filled
  if (ipcbuf_mark_filled (ctx->out_hdu->header_block, header_size) < 0)  {
    multilog (log, LOG_ERR, "Could not mark filled Header Block\n");
    return -1;
  }
  if (ctx->verbose) 
    multilog (log, LOG_INFO, "open: HDU (key=%x) opened for writing\n", ctx->out_key);

  //VG: not sure about these two things below.. They are not used anywhere in this script. Do we need to change their values because we are changing output from 8 bit to 32 bit? Check with AJ
  client->transfer_bytes = transfer_size; 
  client->optimal_bytes = 64*1024*1024;

  ctx->bytes_in = 0;
  ctx->bytes_out = 0;
  client->header_transfer = 0;

  return 0;
}

int dbfurbydb_close (dada_client_t* client, uint64_t bytes_written)
{
  mopsr_dbfurbydb_t* ctx = (mopsr_dbfurbydb_t*) client->context;
  
  multilog_t* log = client->log;

  unsigned ichan, isig, i;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: bytes_in=%"PRIu64", bytes_out=%"PRIu64"\n",
                    ctx->bytes_in, ctx->bytes_out );

  // unlock write on the datablock (end the transfer)
  if (ctx->verbose)
    multilog (log, LOG_INFO, "close: dada_hdu_unlock_write\n");

  if (dada_hdu_unlock_write (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "dbfurbydb_close: cannot unlock DADA HDU (key=%x)\n", ctx->out_key);
    return -1;
  }

  if(ctx->furby_buffer)
    free(ctx->furby_buffer);

  if(ctx->tmp_buff)
    free(ctx->tmp_buff);

  multilog(log, LOG_INFO, "dbfurbydb_close: Succesfully added %i furby(s), %i were left to add. %i were bad injection requests.\n", ctx->furbies_added, (ctx->N_furbys_to_add_here - ctx->furbies_added), ctx->bad_requests);

  if(ctx->copy_furby)
  {
    int i;
    multilog(log, LOG_INFO, "close: Closing all the file ptrs to FURBYs\n");
    for(i= 0;i<ctx->N_furbys_to_add_here; i++)
    {
      if(ctx->FURBY_ptrs[i])
        fclose(ctx->FURBY_ptrs[i]);
      ctx->FURBY_ptrs[i]=0;
    }
  }

  return 0;
}

/*! Pointer to the function that transfers data to/from the target */
int64_t dbfurbydb_write (dada_client_t* client, void* data, uint64_t data_size)
{
  mopsr_dbfurbydb_t* ctx = (mopsr_dbfurbydb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: to_write=%"PRIu64"\n", data_size);

  // write dat to all data blocks
  ipcio_write (ctx->out_hdu->data_block, data, data_size);

  ctx->bytes_in += data_size;
  ctx->bytes_out += data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write: read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, data_size);
 
  return data_size;
}



// add furbies in SFT ordered data
int64_t dbfurbydb_SFT (dada_client_t * client, void *in_data , uint64_t data_size, uint64_t block_id)
{
  mopsr_dbfurbydb_t* ctx = (mopsr_dbfurbydb_t*) client->context;

  multilog_t * log = client->log;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbfurbydb_SFT: data_size=%"PRIu64", block_id=%"PRIu64"\n",
              data_size, block_id);

  const uint64_t nsamp  = data_size / (ctx->nchan * ctx->nsig * ctx->ndim * (ctx->nbit_in/8));

  //  assume 32-bit, detected (ndim == 1) data
  float * in = (float *) in_data;
  uint64_t out_block_id, isamp;
  unsigned ichan, isig, i;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbfurbydb_SFT: nsamp=%lu\n", nsamp);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "write_block_SFT_to_SFT: ipcio_open_block_write()\n");
  ctx->out_block = ipcio_open_block_write(ctx->out_hdu->data_block, &out_block_id);
  if (!ctx->out_block)
  {
    multilog (log, LOG_ERR, "dbfurbydb_SFT: ipcio_open_block_write failed %s\n", strerror(errno));
    return -1;
  }


  uint64_t out_data_size = data_size / ctx->bitrate_factor;
//====================================== VG ==================================

  uint64_t beam_size = data_size / ctx->nsig;
  int chansamp = nsamp * ctx->nchan;

  float * out = (float *) ctx->out_block;
  unsigned furby_chan;
  unsigned ibeam;
  uint64_t index=0;
  uint64_t curr_samp=0;
  uint64_t furby_start_samp, furby_end_samp;
  uint64_t furby_index=0;
  int skip_loop = 0;
  int switched = 0;

  int buff_index = 0;

  if(ctx->verbose)
      multilog(log, LOG_INFO, "dbfurbydb_SFT: passed samp = %"PRIu64"\n", ctx->passed_samps);

  if((ctx->N_furbys_to_add_here < 1) || (ctx->furbies_added == ctx->N_furbys_to_add_here))
  {
    memcpy(out, in, data_size);
    skip_loop = 1;
  }
  //Checking if the last sample in this data block is less that the first sample of the next furby to be added
  else if((ctx->passed_samps + nsamp) < (ctx->furby_samp_indices[ctx->furbies_added] - (int)(ctx->furby_nsamps/2)))
  {
    memcpy(out, in, data_size);
    skip_loop = 1;
  }
  else
  {
    skip_loop = 0;
  }

  if(!skip_loop)
  {

    for(ibeam=ctx->first_beam; ibeam<(ctx->last_beam +1); ibeam++)
    {

      isig = ibeam - ctx->first_beam;
      if (ibeam == ctx->furby_beam_indices[ctx->furbies_added] )
      {
         for(ichan = 0; ichan < ctx->nchan; ichan++)
         {
           if ( (ctx->bw * ctx->furby_BW) < 0)
             furby_chan=ctx->nchan-(ichan+1);
           else
             furby_chan=ichan;
   
           for(isamp = 0; isamp < nsamp; isamp++)
           {
             index = ( isig * ctx->nchan + ichan ) * nsamp + isamp;
             buff_index = isamp * ctx->nchan + furby_chan;   //Saving in TF order and in the same frequency order(Hi -> Lo / Lo -> Hi) as the input furby
             furby_start_samp = ctx->furby_samp_indices[ctx->furbies_added] - (int)(ctx->furby_nsamps/2);
             furby_end_samp = ctx->furby_samp_indices[ctx->furbies_added] + (int)(ctx->furby_nsamps/2) -1;
             curr_samp = ctx->passed_samps + isamp;

             out[index] = in[index];
             
             if ( (curr_samp >= furby_start_samp) & (curr_samp <= furby_end_samp) )
             {
               furby_index = ctx->furbies_added * ctx->furby_chansamp + (curr_samp - furby_start_samp)*ctx->nchan + furby_chan;
               
               out[index] = in[index] + ctx->furby_buffer[furby_index];
             }

             if(ctx->copy_furby)
             {
               ctx->tmp_buff[buff_index] = out[index];
             }
             
             if ( (curr_samp == furby_end_samp) & (ichan == (ctx->nchan - 1)) )
             {
               ctx->furbies_added++;
               switched = 1;
               multilog(log, LOG_INFO, "dbfurbydb_SFT: Added one Furby in BEAM_%i starting at sample: %"PRIu64", ending at sample: %"PRIu64".\nTotal furbies added uptill now: %i\n", ibeam+1, furby_start_samp, furby_end_samp, ctx->furbies_added);
             }//if_last_samp

           }//isamp
         }//ichan

         if(ctx->copy_furby)
         {
           if(switched)
             fwrite(ctx->tmp_buff, ctx->tmp_buff_size, 1, ctx->FURBY_ptrs[ctx->furbies_added-1]);
           else
             fwrite(ctx->tmp_buff, ctx->tmp_buff_size, 1, ctx->FURBY_ptrs[ctx->furbies_added]);
         }
  
      }//if_injection_beam
      else
      {
        memcpy(&(out[isig * chansamp]) , &(in[isig * chansamp]), beam_size);
      } 

    }//ibeam

  }//skip_loop

  ctx->passed_samps+=nsamp;

//============================================================================
   

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbfurbydb_SFT close_block_write written=%"PRIu64"\n", out_data_size);
  if (ipcio_close_block_write (ctx->out_hdu->data_block, out_data_size) < 0)
  {
    multilog (log, LOG_ERR, "dbfurbydb_SFT ipcio_close_block_write failed\n");
    return -1;
  }

  ctx->bytes_in += data_size;
  ctx->bytes_out += out_data_size;

  if (ctx->verbose)
    multilog (log, LOG_INFO, "dbfurbydb_SFT read %"PRIu64", wrote %"PRIu64" bytes\n", data_size, out_data_size);

  return data_size;
}

int main (int argc, char **argv)
{
  mopsr_dbfurbydb_t dbfurbydb = DADA_DBFURBYDB_INIT;

  mopsr_dbfurbydb_t * ctx = &dbfurbydb;

  dada_hdu_t* hdu = 0;

  dada_client_t* client = 0;

  /* DADA Logger */
  multilog_t* log = 0;

  /* Flag set in daemon mode */
  char daemon = 0;

  // number of transfers
  unsigned single_transfer = 0;

  // use zero copy transfers
  unsigned zero_copy = 0;

  // input data block HDU key
  key_t in_key = 0;

  int arg = 0;

  ctx->verbose = 0;
  
  strcpy(&(ctx->furby_database[0]), "/home/dada/furby_database/");
  
  //ctx->bitrate_factor = 4;
  ctx->bitrate_factor = 1;
  ctx->copy_furby = 0;
  
  while ((arg=getopt(argc,argv,"dsD:vcz")) != -1)
  {
    switch (arg) 
    {
      case 'd':
        daemon = 1;
        break;

      case 's':
        single_transfer = 1;
        break;

      case 'D':
        if (strcpy(ctx->furby_database, optarg) == NULL)
        {
          fprintf(stderr,"ERROR: Could not parse (%s)\n", optarg);
	  return EXIT_FAILURE;
        }
        break;

      case 'v':
        ctx->verbose++;
        break;
        
      case 'z':
        zero_copy = 1;
        break;

      case 'c':
        ctx->copy_furby = 1;
        break;
        
      default:
        usage ();
        return 0;
      
    }
  }

  int num_args = argc-optind;
  int i = 0;
      
  if ((argc-optind) != 3)
  {
    fprintf(stderr, "mopsr_dbfurbydb: 3 arguments required\n");
    usage();
    exit(EXIT_FAILURE);
  } 


  if (ctx->verbose)
    fprintf (stderr, "parsing BP_NODE number %s\n", argv[optind]);
  if (sscanf (argv[optind], "%i", &ctx->bp_node) != 1)
  {
    fprintf (stderr, "mopsr_dbfurbydb: could not parse BP_NODE from %s\n", argv[optind]);
    return EXIT_FAILURE;
  }
  if (ctx->verbose)
    fprintf (stderr, "parsing input key=%s\n", argv[optind+1]);
  if (sscanf (argv[optind+1], "%x", &in_key) != 1) {
    fprintf (stderr, "mopsr_dbfurbydb: could not parse in key from %s\n", argv[optind+1]);
    return EXIT_FAILURE;
  }

  // read output DADA key from command line arguments
  if (ctx->verbose)
    fprintf (stderr, "parsing output key %s\n", argv[optind+2]);
  if (sscanf (argv[optind+2], "%x", &(ctx->out_key)) != 1) {
    fprintf (stderr, "mopsr_dbfurbydb: could not parse out key from %s\n", argv[optind+2]);
    return EXIT_FAILURE;
  }

  if (ctx->verbose)
    multilog(log, LOG_INFO, "Using %s as path to furby database\n", ctx->furby_database);


  log = multilog_open ("mopsr_dbfurbydb", 0);

  multilog_add (log, stderr);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: creating in hdu\n");

  // setup input DADA buffer
  hdu = dada_hdu_create (log);
  dada_hdu_set_key (hdu, in_key);
  if (dada_hdu_connect (hdu) < 0)
  {
    fprintf (stderr, "mopsr_dbfurbydb: could not connect to input data block\n");
    return EXIT_FAILURE;
  }

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: lock read key=%x\n", in_key);
  if (dada_hdu_lock_read (hdu) < 0)
  {
    fprintf(stderr, "mopsr_dbfurbydb: could not lock read on input data block\n");
    return EXIT_FAILURE;
  }

  // get the block size of the DADA data block
  uint64_t block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) hdu->data_block);

  // setup output data block
  ctx->out_hdu = dada_hdu_create (log);
  dada_hdu_set_key (ctx->out_hdu, ctx->out_key);
  if (dada_hdu_connect (ctx->out_hdu) < 0)
  {
    multilog (log, LOG_ERR, "cannot connect to DADA HDU (key=%x)\n", ctx->out_key);
    return -1;
  }
  ctx->out_block = 0;
  ctx->out_block_size = ipcbuf_get_bufsz ( (ipcbuf_t *) ctx->out_hdu->data_block);

  if (ctx->verbose)
    multilog (log, LOG_INFO, "main: ctx->out_block_size=%"PRIu64"\n", ctx->out_block_size);
  if (zero_copy && (ctx->out_block_size != (block_size / ctx->bitrate_factor)))
  {
    multilog (log, LOG_ERR, "output block size [%"PRIu64"]  must be input block size [%"PRIu64"] / %u\n", ctx->out_block_size, block_size, ctx->bitrate_factor);
   return EXIT_FAILURE;
  }
 
  client = dada_client_create ();

  client->log           = log;
  client->data_block    = hdu->data_block;
  client->header_block  = hdu->header_block;
  client->open_function = dbfurbydb_open;
  client->io_function   = dbfurbydb_write;

  if (zero_copy)
  {
    client->io_block_function = dbfurbydb_SFT;
  }
  else
  {
    multilog (log, LOG_ERR, "currently zero copy must be used\n");
    return EXIT_FAILURE;
  }

  client->close_function = dbfurbydb_close;
  client->direction      = dada_client_reader;

  client->context = &dbfurbydb;

  client->quiet = (ctx->verbose > 0) ? 0 : 1;

  while (!client->quit)
  {
    if (ctx->verbose)
      multilog (log, LOG_INFO, "main: dada_client_read()\n");

    if (dada_client_read (client) < 0)
      multilog (log, LOG_ERR, "Error during transfer\n");

    if (ctx->verbose)
      multilog (log, LOG_INFO, "main: dada_hdu_unlock_read()\n");

    if (dada_hdu_unlock_read (hdu) < 0)
    {
      multilog (log, LOG_ERR, "could not unlock read on hdu\n");
      return EXIT_FAILURE;
    }

    if (single_transfer || ctx->quit)
      client->quit = 1;

    if (!client->quit)
    {
      if (dada_hdu_lock_read (hdu) < 0)
      {
        multilog (log, LOG_ERR, "could not lock read on hdu\n");
        return EXIT_FAILURE;
      }
    }
  }

  if (dada_hdu_disconnect (hdu) < 0)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
