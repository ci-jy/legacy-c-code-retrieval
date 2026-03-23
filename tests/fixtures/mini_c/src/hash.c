#include <stdlib.h>
#include <string.h>
#include "hash.h"

static int table_resize(Table *t);

/* Compute the FNV-1a hash of a byte string. */
unsigned long hash_bytes(const char *s)
{
    unsigned long h = 2166136261UL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 16777619UL;
    }
    return h;
}

/* Look up a key in the hash table and return its stored value, or NULL when the key is missing. */
void *table_get(Table *t, const char *key)
{
    if (t->nbuckets == 0)
        return NULL;
    Entry *e = t->buckets[hash_bytes(key) % t->nbuckets];
    for (; e != NULL; e = e->next)
        if (strcmp(e->key, key) == 0)
            return e->value;
    return NULL;
}

/* Insert or replace the value stored under a key, resizing the table when it becomes too full. */
int table_put(Table *t, const char *key, void *value)
{
    if (t->count + 1 > t->nbuckets * 3 / 4 && table_resize(t) != 0)
        return -1;
    size_t i = hash_bytes(key) % t->nbuckets;
    Entry *e = malloc(sizeof *e);
    if (e == NULL)
        return -1;
    e->key = strdup(key);
    e->value = value;
    e->next = t->buckets[i];
    t->buckets[i] = e;
    t->count++;
    return 0;
}

// Double the number of buckets
// and rehash every entry.
static int table_resize(Table *t)
{
    size_t n = t->nbuckets ? t->nbuckets * 2 : 8;
    Entry **nb = calloc(n, sizeof *nb);
    if (nb == NULL)
        return -1;
    for (size_t i = 0; i < t->nbuckets; i++) {
        Entry *e = t->buckets[i];
        while (e != NULL) {
            Entry *next = e->next;
            size_t j = hash_bytes(e->key) % n;
            e->next = nb[j];
            nb[j] = e;
            e = next;
        }
    }
    free(t->buckets);
    t->buckets = nb;
    t->nbuckets = n;
    if (t->on_resize)
        t->on_resize(t);
    return 0;
}

/* Recursively free a chain of hash table entries. */
static void free_chain(Entry *e)
{
    if (e == NULL)
        return;
    free_chain(e->next);
    free(e->key);
    free(e);
}

void table_free(Table *t)
{
    for (size_t i = 0; i < t->nbuckets; i++)
        free_chain(t->buckets[i]);
    free(t->buckets);
}
