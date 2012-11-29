#ifndef __DADA_GENERATOR_H
#define __DADA_GENERATOR_H

/*
 * Functions for generating 2/4/8 bit signals
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

double rand_normal(double mean, double stddev);
char * char_to_binary_string(char value);
char * unsigned_to_binary_string(unsigned int value);
void fill_gaussian_chars(char *data, int size, int nbits, int gain);
void fill_cal_chars(char *data, int size, int nbits, int gain);
void fill_gaussian_unsigned(unsigned int *data, int size, int nbits, int gain);
void fill_cal_unsigned(unsigned int *data, int size, int nbits, int gain);
char * int8_to_binary_string(int8_t value);
char * int16_to_binary_string(int16_t value);
char * int_to_binary_string(int value);

#ifdef __cplusplus
}
#endif

#endif /* __DADA_GENERATOR_H */
