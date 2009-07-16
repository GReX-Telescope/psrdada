#ifndef __CASPSR_DEF_H
#define __CASPSR_DEF_H

#include "dada_def.h"

/* default port to connect to udpdb command interface */
#define CASPSR_DEFAULT_UDPDB_PORT 4001

#define CASPSR_DEFAULT_PWC_PORT     49022
#define CASPSR_DEFAULT_PWC_LOGPORT  49023

/* Number of UDP packets per buffer_function */
#define CASPSR_NUM_UDP_PACKETS 16384

#define CASPSR_UDPGEN_LOG  49200
#define CASPSR_CLOCK 400
#define CASPSR_DEFAULT_BW 25

#endif // __CASPSR_DEF_H

