
#include <sys/types.h>
#include <sys/socket.h>

#ifdef MSG_NOSIGNAL
#define DADA_MSG_FLAGS MSG_NOSIGNAL | MSG_WAITALL
#else
#define DADA_MSG_FLAGS MSG_WAITALL
#endif
