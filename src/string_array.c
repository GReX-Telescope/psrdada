#include "string_array.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ctype.h>

string_array* string_array_create ()
{
  string_array* list = (string_array*) malloc (sizeof(string_array));
  list->size = 0;
  list->strs = NULL;
  return list;
}

void string_array_free (string_array* list)
{
   unsigned i;
   for (i=0; i < list->size; i++)
      free(list->strs[i]);
   free (list->strs);
   free (list);
}

/*! This function will insert the string "entry" into the array of
   string_array referenced by "list".  It will be placed into the array
   at the index "pos" (starting at zero).  If a string already
   exists at this position, all entries will be bumped toward the end of
   the array.  If pos > list->size_entries, -1 will be returned.
 */
int string_array_insert (string_array* list, char* entry, unsigned pos)
{
  char** temp;
  unsigned val;

  if (pos > list->size)
    return -1;

  temp = (char**) realloc (list->strs, (list->size+1)*sizeof(char**));
  if (temp == NULL) {
    perror ("string_array_insert: realloc failed!\n");
    return -1;
  }
  list->strs = temp;
  for (val = list->size; val > pos; val--)
    list->strs[val] = list->strs[val-1];
  list->strs[pos] = strdup (entry);
  list->size ++;

  return 0;
}

int string_array_append (string_array* list, char* entry)
{
  return (string_array_insert (list, entry, list->size));
}

int string_array_remove (string_array* list, unsigned pos)
{
  unsigned val;
  char ** temp;

  if (list->size == 0)
    return -1;

  if (pos >= list->size)
    return -1;

  free (list->strs[pos]);
  list->size --;

  for (val=pos; val<list->size; val++)
    list->strs[val] = list->strs[val+1];

  temp = (char**) realloc (list->strs, (list->size)*sizeof(char**));
  if (list->size == 0) {
    list->strs = NULL;
    return 0;
  }
  if (temp == NULL) {
    perror ("string_array_remove: realloc failed!\n");
    return -1;
  }
  list->strs = temp;

  return 0;
}

int string_array_switch (string_array* list, unsigned pos1, unsigned pos2)
/* This function switches the two string_array in the array "list" indexed
   by "pos1" and "pos2".  */
{
  char* temp;
  temp = list->strs[pos1];
  list->strs[pos2] = list->strs[pos1];
  list->strs[pos1] = temp;
  return 0;
}

char* string_array_search (string_array* list, char* match)
/* this function returns the string in "list" containing
   the string "match".  If the string does not exist in the array,
   NULL is returned.  */
{
  unsigned i;
  for (i=0; i < list->size; i++)
    if (strcmp(list->strs[i], match) == 0)
      return list->strs[i];
  return 0;
}


