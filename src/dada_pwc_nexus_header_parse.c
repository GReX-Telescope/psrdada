#include "dada_pwc_nexus.h"
#include "ascii_header.h"
#include "futils.h"

#include <string.h>
#include <stdlib.h>

int dada_pwc_config_header (const char* prefix, char* header,
			    const char* config)
{
  char parm_name  [64] = "";
  char parm_value [64] = "";

  char* parms[] = { "FREQ", "BW", 0 };
  char** parm = 0;

  if (!config) {
    fprintf (stderr, "dada_pwc_config_header no config");
    return -1;
  }
  if (!header) {
    fprintf (stderr, "dada_pwc_config_header no header");
    return -1;
  }

  for (parm=parms; *parm != 0; parm++) {

    sprintf (parm_name, "%s%s", prefix, *parm);

    if (ascii_header_get (config, parm_name, "%s", parm_value) < 0)
      fprintf (stderr, "dada_pwc_config_header WARNING %s not found\n",
	       parm_name);
    else if (ascii_header_set (header, *parm, "%s", parm_value) < 0) {
      fprintf (stderr, "dada_pwc_config_header ERROR setting %s=%s\n",
	       *parm,parm_value);
      return -1;
    }
      
  }

  return 0;
}


int dada_pwc_nexus_header_parse (dada_pwc_nexus_t* n, const char* buffer)
{
  char node_name [16] = "";
  dada_node_t* node = 0;

  unsigned inode, nnode = nexus_get_nnode ((nexus_t*) n);

  /* First set up the common header parameters */
  if (n->header_template)
    fileread (n->header_template, n->pwc->header, n->pwc->header_size);
  else
    n->pwc->header[0] = '\0';

  if (dada_pwc_config_header (node_name, n->pwc->header, buffer) < 0)
    return -1;

  for (inode=0; inode < nnode; inode++) {

    node = (dada_node_t*) n->nexus.nodes[inode];
    if (node->header_size < n->pwc->header_size) {
      node->header = realloc (node->header, n->pwc->header_size);
      node->header_size = n->pwc->header_size;
    }

    strcpy (node->header, n->pwc->header);
    sprintf (node_name, "Band%d_", inode);
    if (dada_pwc_config_header (node_name, node->header, buffer) < 0)
      return -1;
    
  }
    
  return 0;
}

