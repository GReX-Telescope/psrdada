#ifndef STRING_ARRAY_H
#define STRING_ARRAY_H

typedef struct {
  unsigned size;
  char** strs;
} string_array_t;

/*! Create a new array of strings */
string_array_t* string_array_create ();

/*! Destroy an exisitng array of strings */
void string_array_destroy (string_array_t* list);

/*! Return the requested string */
char* string_array_get (string_array_t* list, unsigned pos);

/*! Return the size of the array */
unsigned string_array_size (string_array_t* list);

/*! Load an array of strings from file */
int string_array_load (string_array_t* list, const char* filename);

/*! Insert a new string into the specified position */
int string_array_insert (string_array_t* list, const char* str, unsigned pos);

/*! Append a new string to the end of the array */
int string_array_append (string_array_t* list, const char* entry);

/*! Remove a string from the array */
int string_array_remove (string_array_t* list, unsigned pos);

/*! Switch two strings in the array */
int string_array_switch (string_array_t* list, unsigned pos1, unsigned pos2);

/*! Return the matching string (null if not found) */
char* string_array_search (string_array_t* list, char* match);

#endif
