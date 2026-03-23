#ifndef HASH_H
#define HASH_H

#include <stddef.h>

typedef struct Entry {
    char *key;
    void *value;
    struct Entry *next;
} Entry;

typedef struct Table {
    Entry **buckets;
    size_t nbuckets;
    size_t count;
    void (*on_resize)(struct Table *t);
} Table;

unsigned long hash_bytes(const char *s);
void *table_get(Table *t, const char *key);
int table_put(Table *t, const char *key, void *value);
void table_free(Table *t);

#endif
