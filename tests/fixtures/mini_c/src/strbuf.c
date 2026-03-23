#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "strbuf.h"

static void log_msg(const char *m)
{
    fprintf(stderr, "strbuf: %s\n", m);
}

/* Grow the buffer so that it can hold at least need more bytes. */
static int sb_grow(StrBuf *sb, size_t need)
{
    size_t newcap = sb->cap ? sb->cap : 16;
    while (newcap < sb->len + need + 1)
        newcap *= 2;
    char *p = realloc(sb->data, newcap);
    if (p == NULL) {
        log_msg("out of memory");
        return -1;
    }
    sb->data = p;
    sb->cap = newcap;
    return 0;
}

/* Append a NUL-terminated string to the end of the string buffer. */
int sb_append(StrBuf *sb, const char *s)
{
    size_t n = strlen(s);
    if (sb_grow(sb, n) != 0)
        return -1;
    memcpy(sb->data + sb_len(sb), s, n + 1);
    sb->len += n;
    return 0;
}

/*
 * Release the memory owned by a string buffer
 * and reset it to empty.
 */
void sb_free(StrBuf *sb)
{
    free(sb->data);
    sb->data = NULL;
    sb->len = sb->cap = 0;
}
