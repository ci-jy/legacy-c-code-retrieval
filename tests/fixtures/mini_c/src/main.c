#include <stdio.h>
#include <string.h>
#include "hash.h"
#include "strbuf.h"

static void log_msg(const char *m)
{
    fputs(m, stderr);
}

/* Parse one "key=value" line and store it in the table. */
static int parse_line(Table *t, char *line)
{
    char *eq = strchr(line, '=');
    if (eq == NULL)
        return -1;
    *eq = '\0';
    return table_put(t, line, strdup(eq + 1));
}

/* Read configuration lines from a file into the table, skipping comments and blank lines. */
int load_config(Table *t, const char *path)
{
    char line[256];
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        log_msg("cannot open config");
        return -1;
    }
    while (fgets(line, sizeof line, fp) != NULL) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        line[strcspn(line, "\n")] = '\0';
        parse_line(t, line);
    }
    fclose(fp);
    return 0;
}

#ifdef DEBUG
/* Write every configuration entry to standard error. */
static void dump_config(Table *t)
{
    StrBuf sb = {0};
    for (size_t i = 0; i < t->nbuckets; i++)
        for (Entry *e = t->buckets[i]; e != NULL; e = e->next) {
            sb_append(&sb, e->key);
            sb_append(&sb, "\n");
        }
    log_msg(sb.data);
    sb_free(&sb);
}
#endif

/* Entry point: load the config file named on the command line and print one value. */
int main(int argc, char **argv)
{
    Table t = {0};
    if (argc < 3 || load_config(&t, argv[1]) != 0)
        return 1;
#ifdef DEBUG
    dump_config(&t);
#endif
    const char *v = table_get(&t, argv[2]);
    printf("%s\n", v ? v : "(missing)");
    table_free(&t);
    return 0;
}
