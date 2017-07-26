
// gcc mopsr_plot_cands_fgbg.c -I/home/dada/linux_64/pgplot -L/home/dada/linux_64/pgplot -l cpgplot -l pgplot -L/usr/local/lib -L/usr/lib/gcc/x86_64-redhat-linux/4.4.7 -L/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../lib64 -L/lib/../lib64 -L/usr/lib/../lib64 -L/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../.. -lstdc++ -lgfortranbegin -lgfortran -lm -lX11 -lX11 -lpng -o mopsr_plot_cands_fgbg

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cpgplot.h>

void help() {

  printf("Usage: >plot_cands -f <coincidenced cands file> -time1 <time1> -time2 <time2> -nbeams <nbeams> -nbeams_cut <max number of coincident beams> -snr_cut <snr cut> -max_filter_width <max filter width> -dev <> -scale <pixel_scale> -catted_raw_output -verbose\n");
  exit(1);
  
}
void set_resolution (int width_pixels, int height_pixels);
void get_scale (int from, int to, float * width, float * height);
int wid_col(int wid);

int wid_col(int wid) {

  if (wid==0) return 3;
  else if (wid==1) return 9;
  else if (wid==2) return 5;
  else if (wid==3) return 11;
  else if (wid==4) return 4;
  else if (wid==5) return 7;
  else if (wid==6) return 8;
  else if (wid==7) return 6;
  else if (wid==8) return 2; 
  else if (wid==9) return 2; 
  else return 14;

}

int main(int argc, char *argv[]) {

  FILE *fin;
  int i, j;
  char fnam[200], line[200];
  char pg_dev[100];
  char lgnd[100];
  sprintf(pg_dev,"/xs");

  int nbeams = 1;
  int nbeams_cut = 10;
  float t_cand1 = 0.;
  float t_cand2 = 100.;
  float snr_cut = 6.0;
  int max_filter_width = 9;
  int cro = 0; // catted_raw_output
  int verbose = 0;
  float dm_rfi = 1.5;
  float dm_gal = 100.;
  float height_scale = 30;

  for (i=1;i<argc;i++) {

    if (strcmp(argv[i],"-nbeams")==0) {
      nbeams = atoi(argv[++i]);
    }
    if (strcmp(argv[i],"-f")==0) {
      strcpy(fnam,argv[++i]);
    }
    if (strcmp(argv[i],"-max_filter_width")==0) {
      max_filter_width = atoi(argv[++i]);
    }
    if (strcmp(argv[i],"-snr_cut")==0) {
      snr_cut = atof(argv[++i]);
    }
    if (strcmp(argv[i],"-time1")==0) {
      t_cand1 = atof(argv[++i]);
    }
    if (strcmp(argv[i],"-time2")==0) {
      t_cand2 = atof(argv[++i]);
    }
    if (strcmp(argv[i],"-nbeams_cut")==0) {
      nbeams_cut = atoi(argv[++i]);
    }
    if (strcmp(argv[i],"-scale")==0) {
      height_scale = atoi(argv[++i]);
    }
    if (strcmp(argv[i],"-h")==0) {
      help();
    }
    if (strcmp(argv[i],"-dev")==0) {
      strcpy(pg_dev,argv[i+1]);
    }
    if ((strcmp(argv[i],"-v")==0) || (strcmp(argv[i],"-verbose")==0)) {
      verbose = 1;
    }
    if (strcmp(argv[i],"-v")==0) {
      verbose = 1;
    }
    if (strcmp(argv[i],"-catted_raw_output")==0) {
      cro = 1;
    }
  }

  if (verbose)
  {
    fprintf (stderr, "nbeams=%d\n", nbeams);
    fprintf (stderr, "max_filter_width=%d\n", max_filter_width);
    fprintf (stderr, "snr_cut=%f\n", snr_cut);
    fprintf (stderr, "verbose=%d\n", verbose);
  }

  //double tsamp = 6.5536e-4;
  //float rc, gc, bc;

  //double snr, t, dm, samp;
  float snr, t, dm, voffset;
  int fil, dmt, memb, stsamp, midsamp, samp;
  int nlines, lp;

  int numbeams, prim_beam, beam;
  //double max_snr;
  float max_snr;

  cpgbeg(0, pg_dev, 1, 1);

  cpgbbuf();

  set_resolution (1024, 768);

  cpgsch(1.0);
  cpgsfs(2);
  cpgscf(2);
  cpgslw(1);

  //  char *cpgch;
  //  cpgch = (char *)malloc(sizeof(char)*100);

  if (verbose) printf("Using pgplot device %s\n",pg_dev);

  // colour bar

  cpgsvp(0.95,1.,0.15,0.90);
  cpgswin(0.,1.,-0.5,8.5);
  cpgbox("",0.,0.,"BSTN",0.,0.);
  cpglab("","Filter width [log\\d2\\u(dt)]","");

  float ppx[4], ppy[4];
  ppx[0]=0.;
  ppx[1]=1.;
  ppx[2]=1.;
  ppx[3]=0.;
  ppy[0]=-0.5;
  ppy[1]=-0.5;
  ppy[2]=0.5;
  ppy[3]=0.5;
  for (i=0;i<=9;i++) {

    cpgsci(wid_col(i));

    cpgsfs(1);
    cpgpoly(4,ppx,ppy);
    cpgsfs(2);
    //printf("%d\n",i);
    for (j=0;j<4;j++) ppy[j] += 1.;

  }

  cpgsci(1);
  cpgbox("",0.,0.,"BSTN",0.,0.);

  // legend
  cpgsvp(0.1,0.82,0.92,0.98);
  cpgswin(0,1,0,1);
  
  ppx[0]=0.1;
  ppy[0]=0.5;
  cpgpt(1,ppx,ppy,5);
  sprintf(lgnd,"DM < %0.1f",dm_rfi);
  cpgsch(0.9);
  cpgptxt(ppx[0]+0.02,ppy[0]-0.1,0.,0.,lgnd);
  cpgsch(1.4);

  ppx[0]=0.36;
  cpgpt(1,ppx,ppy,23);
  sprintf(lgnd,"%0.1f < DM < %0.1f",dm_rfi,dm_gal);
  cpgsch(0.9);
  cpgptxt(ppx[0]+0.02,ppy[0]-0.1,0.,0.,lgnd);
  cpgsch(1.4);

  ppx[0]=0.76;
  cpgpt(1,ppx,ppy,12);
  sprintf(lgnd,"DM > %0.1f",dm_gal);
  cpgsch(0.9);
  cpgptxt(ppx[0]+0.02,ppy[0]-0.1,0.,0.,lgnd);
  cpgsch(1.4);


  // main plot

  cpgsvp(0.1,0.82,0.15,0.9);
  cpgswin(0,nbeams+1,t_cand1,t_cand2);

  // beam lines
  /*ppy[0]=t_cand1;
  ppy[1]=t_cand2;
  cpgsci(15);
  for (i=1;i<=nbeams;i++) {
    ppx[0]=i*1.;
    ppx[1]=i*1.;
    cpgsls(4);
    cpgslw(0.3);
    cpgline(2,ppx,ppy);
    cpgslw(2);
    cpgsls(1);
    }*/

  if (!(fin=fopen(fnam,"r"))) {
    printf("Can't open file\n");
    printf("%s\n",fnam);
    exit(1);
  }

  cpgslw(1.);

  float char_height;

  // find number of lines
  nlines=0;
  while (fgets(line,100,fin)!=NULL) nlines++;
  fclose(fin);
  fin=fopen(fnam,"r");
  if (verbose) printf("Have %d lines\n",nlines);

  if (t_cand2 - t_cand1 > 1800.0f) 
    height_scale *= (1800./(t_cand2-t_cand1));

  cpgscr (16, .4, .4, .4);

  float cand;
  int nread;
  int round;

  for (round=0;round<2;round++){

    if (verbose) fprintf (stderr, "round=%d\n", round);

    fin=fopen(fnam,"r");

    for (lp=0;lp<nlines;lp++) {
        
      // read a line from the input candidates file
      if (!cro) 
        nread = fscanf(fin,"%f\t%d\t%f\t%d\t%d\t%f\t%d\t%d\t%d\t%d\t%d\t%f\t%d\n",&snr,&samp,&t,&fil,&dmt,&dm,&memb,&stsamp,&midsamp,&numbeams,&prim_beam,&max_snr,&beam);
      else {
        nread = fscanf(fin,"%f\t%d\t%f\t%d\t%d\t%f\t%d\t%d\t%d\t%d\n",&snr,&samp,&t,&fil,&dmt,&dm,&memb,&stsamp,&midsamp,&prim_beam);
        if (nread != 10)
          fprintf (stderr, "misread\n");
        numbeams=1;
      }
      
      // changed by CF 17/03/17
      //ppx[0]=prim_beam+1;
      ppx[0]=prim_beam;

      ppy[0]=t*1.;

      //fprintf (stderr, "t=%f [%f - %f] numbeams %d<=%d, snr %f>=%f fil %d<=%d\n",
      //         t, t_cand1, t_cand2, numbeams, nbeams_cut, snr, snr_cut, fil, max_filter_width);

      int col, draw_cand;

      if ((t>=t_cand1) && (t<=t_cand2) && (snr>=snr_cut) && (fil<=max_filter_width) && prim_beam > 1) 
      {
        if (snr>6.0)
        {
          if (snr>20.)
            char_height = height_scale; 
          else 
            char_height = height_scale * powf(snr/20.,2.3);

          cpgsch(char_height);
          draw_cand = 0;

          // set hidden events to dark grey
          if ((round == 0) && (numbeams >= nbeams_cut))
          {
            col = 16;
            draw_cand = 1;
          }

          if ((round == 1) && (numbeams < nbeams_cut))
          {
            col = wid_col(fil);
            draw_cand = 1;
          }

          if (draw_cand)
          {
            cpgsci (col);
            if (dm<=dm_rfi) cpgpt(1,ppx,ppy,5);
            if (dm>dm_rfi && dm<=dm_gal) cpgpt(1,ppx,ppy,23);
            if (dm>dm_gal) cpgpt(1,ppx,ppy,12);
            if (verbose) fprintf(stderr, "plotted %f %f %f %d %d [%f]\n",snr, t, dm, fil, prim_beam, char_height);
          }

/*
          int draw_symb;
          draw_symb = 1;
          if (round == 0) cpgsci (16);
          if (round == 1 && numbeams >= nbeams_cut) draw_symb = 0;
          if (draw_symb){
            if (dm<=dm_rfi) cpgpt(1,ppx,ppy,5);
            if (dm>dm_rfi && dm<=dm_gal) cpgpt(1,ppx,ppy,23);
            if (dm>dm_gal) cpgpt(1,ppx,ppy,12);
            if (verbose) fprintf(stderr, "plotted %f %f %f %d %d [%f]\n",snr, t, dm, fil, prim_beam, char_height);
          }
          */
        }
      }
    }
    
    fclose(fin);
  }
    
  // pgplot interactive
  /*float x1, y1;
  while (strcmp(cpgch,"q")!=0) {
    cpgcurs(&x1,&y1,&cpgch[0]);
    }*/

  // draw box 
  cpgsch(1.4);
  cpgsci(1);
  cpgbox("BCTN",0.,0.,"BCSTN",0.,0.);
  cpglab ("Beam number", "Time (s)", "");

  // draw dashed line for central beam
  float mid_beam = (float) (nbeams/2) + 1;
  float xpts[2] = { mid_beam, mid_beam };
  float ypts[2] = { t_cand1, t_cand2 };

  cpgsls(2);
  cpgsci (15);
  cpgline (2, xpts, ypts);
  cpgsls(2);
  cpgsci (1);

  cpgebuf();
  cpgend();

  //  free(cpgch);

  return 0;
}
void get_scale (int from, int to, float * width, float * height)
{
  float j = 0;
  float fx, fy;
  cpgqvsz (from, &j, &fx, &j, &fy);

  float tx, ty;
  cpgqvsz (to, &j, &tx, &j, &ty);

  *width = tx / fx;
  *height = ty / fy;
}

void set_resolution (int width_pixels, int height_pixels)
{
  float width_scale, height_scale;
  width_pixels--;
  height_pixels--;

  get_scale (3, 1, &width_scale, &height_scale);

  float width_inches = width_pixels * width_scale;
  float aspect_ratio = height_pixels * height_scale / width_inches;

  cpgpap( width_inches, aspect_ratio );

  float x1, x2, y1, y2;
  cpgqvsz (1, &x1, &x2, &y1, &y2);
}

