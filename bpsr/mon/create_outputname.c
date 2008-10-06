/***************************************************************************/
/*                                                                         */
/* function create_outputname                                              */
/*                                                                         */
/* it creates the name for the outputfile according to the selected device */
/*                                                                         */
/* RB 04 Oct 08: appropriate extentions to the output png files           */
/*                                                                         */
/***************************************************************************/

#include "plot4mon.h"

void create_outputname( char inpfile[], char inpdev[], char outputfile[],
			int plotnum, char add_work[])
{
  int dot_position;
  char extens[80];
  char *pn2extens;
  char newfile[80];
  char newinp[80];
  char ext[10];

  if (plotnum==0)
    {
      if (!( (strings_compare(inpdev,"/xs")) || (strings_compare(inpdev,"/XS"))
        ||(strings_compare(inpdev,"/xw")) || (strings_compare(inpdev,"/XW")) ))
       {
         if (strstr(inpfile, "bps") != NULL) strcpy(ext,".bps");
         else if (strstr(inpfile, "bp") != NULL) strcpy(ext,".bp");
         else if (strstr(inpfile, "ts") != NULL) strcpy(ext,".ts");
         else strcpy(ext,"");

         strcpy(newinp,inpfile);
         dot_position=strcspn(inpfile,".");
         newinp[dot_position]='\0';
         strcpy(newfile,newinp);
	 strcpy(extens,inpdev);
	 while((pn2extens=strpbrk(extens,"/"))!=NULL) *pn2extens='.';
	 strcat(newfile,ext);
	 strcat(newfile,extens);
	 printf(" The output file will be %s \n",newfile);
	 strcat(newfile,inpdev);
	 strcat(outputfile,newfile);
       }
      else
       {
	 strcpy(outputfile,inpdev);
	printf(" The output file will be directed to %s screen \n",outputfile);
       }

    }
  else if (plotnum==1) 
    {
      if (!( (strings_compare(inpdev,"/xs")) || (strings_compare(inpdev,"/XS"))
        ||(strings_compare(inpdev,"/xw")) || (strings_compare(inpdev,"/XW")) ))
       {
         strcpy(newinp,inpfile);
         dot_position=strcspn(inpfile,".");
         newinp[dot_position]='\0';
	 strcpy(newfile,newinp);
	 strcat(newfile,".");
	 strcat(newfile,add_work);
	 strcpy(extens,inpdev);
	 while((pn2extens=strpbrk(extens,"/"))!=NULL) *pn2extens='.';
	 strcat(newfile,extens);
	 printf(" The output file will be %s \n",newfile);
	 strcat(newfile,inpdev);
	 strcpy(outputfile,newfile);
       }
      else
       {
	 strcpy(newfile,"2");
	 strcat(newfile,inpdev);
	 strcpy(outputfile,newfile);
	printf(" The output file will be directed to %s screen \n",outputfile);
       }
    }
}
