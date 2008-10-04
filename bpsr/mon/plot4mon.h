/***************************************************************************/
/*                                                                         */
/* include file for plot4mon                                               */
/*                                                                         */
/* Ver 1.0        AP 30 Sept 2008                                          */
/*                                                                         */
/* Ver 2.0        RB 04 Sept 2008
                  default plot device: png 
                  no longer need to set large stack sizes (use of malloc)  */
/*                                                                         */
/***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cpgplot.h"
#define BPSRMON_VERSION 1.0 
//#define STD_DEVICE "/ps"
#define STD_DEVICE "/png"
//#define MAXNUMPOINTS 10000000 /* max number of points to be plotted  */
#define MAX_PERIOD 5.0 /* max period to look at in FFT in sec */
#define MIN_PERIOD 0.001 /* min period to look at in FFT in sec */


void get_commandline(int , char *argv[], char *, char *, char *, char *, int *);
void display_help(char *);
void compute_extremes(float *, long, float *, float *);
void compute_margin(long, float, float, float *);
void read_params(char *, int *, float *, int *, float *, int *, int *, char *);
void read_stream(int , char *, float *, long *);
void work_on_data(char *, float *, float *, long, long *, float, float, int, char *, int);
void create_xaxis(char *,int,long,long, int, float, float *); 
void set_array_dim(long, int, int *, int *);
void create_outputname( char *, char *, char *, int, char *);
void create_labels(char *, int, char *, char *, char *);
void do_powerspec(float *, float *, long, long *, float);
void realft(float *, unsigned long, int);
void whiten(float *, long, long);
float selectmed(unsigned long, unsigned long, float *);
int check_file_exists(char *);
int strings_compare(char *, char *); 
int plot_stream_1D(float *, float *, float *, long, 
                char *, char*, char *, char *);
int plot_stream_2D(float *, int, int, float, int,
                char *, char*, char *, char *);
