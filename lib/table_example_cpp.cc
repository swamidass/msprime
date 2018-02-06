#include "tables.h"

#include <iostream>
#include <stdlib.h>

using namespace std;


static void
handle_error(string msg, int err)
{
    cout << "Error:" << msg << ":" << msp_strerror(err) << endl;
    exit(1);
}

int
main(int argc, char **argv)
{
    int j, ret;
    node_table_t nodes;
    edge_table_t edges;
    migration_table_t migrations;
    site_table_t sites;
    mutation_table_t mutations;
    simplifier_t simplifier;
    node_id_t samples[] = {0};

    ret = node_table_alloc(&nodes, 0, 0);
    if (ret != 0) {
        handle_error("alloc_nodes", ret);
    }
    ret = edge_table_alloc(&edges, 0);
    if (ret != 0) {
        handle_error("alloc_edges", ret);
    }
    /* Even though we're not going to use them, we still have to allocate
     * migration, site and mutations tables because the simplifier class
     * expects them. This is annoying and will be fixed at some point */
    ret = migration_table_alloc(&migrations, 0);
    if (ret != 0) {
        handle_error("alloc_migrations", ret);
    }
    ret = site_table_alloc(&sites, 0, 0, 0);
    if (ret != 0) {
        handle_error("alloc_sites", ret);
    }
    ret = mutation_table_alloc(&mutations, 0, 0, 0);
    if (ret != 0) {
        handle_error("alloc_mutations", ret);
    }

    /* Create a simple chain of nodes, with 0 as the only sample. */
    for (j = 0; j < 10; j++) {
        ret = node_table_add_row(&nodes, j == 0, j, 0, NULL, 0);
        if (ret < 0) {
            handle_error("add_node", ret);
        }
        if (j > 0) {
            ret = edge_table_add_row(&edges, 0, 1, j, j - 1);
            if (ret < 0) {
                handle_error("add_edge", ret);
            }
        }
    }
    /* Useful debugging feature */
    node_table_print_state(&nodes, stdout);
    edge_table_print_state(&edges, stdout);

    ret = simplifier_alloc(&simplifier, 1.0, samples, 1,
            &nodes, &edges, &migrations, &sites, &mutations, 0, 0);
    if (ret < 0) {
        handle_error("simplifier_alloc", ret);
    }
    ret = simplifier_run(&simplifier, NULL);
    if (ret < 0) {
        handle_error("simplifier_run", ret);
    }

    /* After simplify, we only have 1 node left and no edges */
    node_table_print_state(&nodes, stdout);
    edge_table_print_state(&edges, stdout);

    /* Clean up. This should usually also be done in the error handling case,
     * but since this is a simple standalone program and we can exit on
     * error. */
    simplifier_free(&simplifier);
    node_table_free(&nodes);
    edge_table_free(&edges);
    migration_table_free(&migrations);
    site_table_free(&sites);
    mutation_table_free(&mutations);
    return 0;
}
