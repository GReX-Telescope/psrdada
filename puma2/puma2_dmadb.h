/* dada, ipc stuff */

#include "dada_hdu.h"
#include "dada_def.h"

#include "ipcio.h"
#include "multilog.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>

/* pic and EDT stuff */
#include "pic.h"
#include "edtinc.h"

/* EDT recommends datarate/20 for bufsize */
#define bufsize 4000000
