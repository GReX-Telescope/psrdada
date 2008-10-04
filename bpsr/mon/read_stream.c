/***************************************************************************/
/*                                                                         */
/* function read_stream                                                    */
/*                                                                         */
/* it reads from a stream                                                  */
/*                                                                         */
/***************************************************************************/

#include "plot4mon.h"

void read_stream(char inpfile[],float *readstream,long *totvaluesread)
{
int i,nread;
long totnread;
long blksz=1;   // to be adjusted later, if required
FILE *pn2file; 

/* it opens the input binary files, testing for the successful completion */
  if ((pn2file = fopen(inpfile, "rb")) == NULL ) 
  { 
     printf(" Error opening %s for reading  \n",inpfile); 
     exit(-1); 
  }
  printf(" Reading the data of the file %s... \n",inpfile);   

/* "i" runs on the location of the values in the array readstream */
  totnread=0;
  for (i=0;i<=MAXNUMPOINTS+10;i++) 
  {      
     nread = fread(&readstream[blksz*i],1,blksz*sizeof(float),pn2file);
     totnread=totnread+(long)nread;
     if (feof(pn2file)) break;
  } 
  if (totnread > MAXNUMPOINTS*sizeof(float)) 
  {
     printf(" Error! Overflow in the array of plotted values \n");
     printf("        %ld values to be plotted wrt a max of %d values \n",
             totnread/sizeof(float), MAXNUMPOINTS);
     printf("        Redefine MAXNUMPOINTS in the code include file \n");
     exit(-1);
  }
  fclose(pn2file);
  *totvaluesread=totnread/sizeof(float);  // tot number of read values
  printf(" Read %ld %d-bytes long values from %s \n",
          *totvaluesread,sizeof(float),inpfile);  
}
