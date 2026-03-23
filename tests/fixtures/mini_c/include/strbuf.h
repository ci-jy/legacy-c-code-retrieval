#ifndef STRBUF_H
#define STRBUF_H

#include <stddef.h>

typedef struct StrBuf {
    char *data;
    size_t len;
    size_t cap;
} StrBuf;

/* Return the number of bytes currently stored in the buffer. */
static inline size_t sb_len(const StrBuf *sb)
{
    return sb->len;
}

int sb_append(StrBuf *sb, const char *s);
void sb_free(StrBuf *sb);

#endif
