/*
** Copyright (C) 2017 University of Oxford
**
** This file is part of msprime.
**
** msprime is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** msprime is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with msprime.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "err.h"
#include "object_heap.h"
#include "msprime.h"

#define NO_LIKELIHOOD (-1)

typedef struct {
    node_id_t node;
    double likelihood;
} likelihood_t;

static int
cmp_likelihood(const void *a, const void *b) {
    const likelihood_t *ia = (const likelihood_t *) a;
    const likelihood_t *ib = (const likelihood_t *) b;
    return (ia->node > ib->node) - (ia->node < ib->node);
}

static void
haplotype_matcher_check_state(haplotype_matcher_t *self)
{
    size_t j;
    avl_node_t *a, *b;
    likelihood_t search, *lk;
    node_id_t u;

    /* make sure the parent array we're maintaining is correct */
    for (j = 0; j < self->num_nodes; j++) {
        assert(self->parent[j] == self->tree.parent[j]);
    }
    /* Check the properties of the likelihood map */
    for (a = self->likelihood.head; a != NULL; a = a->next) {
        lk = (likelihood_t *) a->item;
        /* Traverse up to root and make sure we don't see any other L values
         * on the way. */
        u = self->parent[lk->node];
        while (u != MSP_NULL_NODE) {
            search.node = u;
            b = avl_search(&self->likelihood, &search);
            assert(b == NULL);
            u = self->parent[u];
        }
    }

    assert(avl_count(&self->likelihood) ==
            object_heap_get_num_allocated(&self->avl_node_heap));
}

void
haplotype_matcher_print_state(haplotype_matcher_t *self, FILE *out)
{
    avl_node_t *a;
    likelihood_t *lk;

    fprintf(out, "tree_sequence = %p\n", (void *) self->tree_sequence);
    fprintf(out, "likelihood = (%d)\n", (int) avl_count(&self->likelihood));
    for (a = self->likelihood.head; a != NULL; a = a->next) {
        lk = (likelihood_t *) a->item;
        fprintf(out, "%d\t->%f\n", lk->node, lk->likelihood);
    }
    fprintf(out, "tree = \n");
    fprintf(out, "\tindex = %d\n", (int) self->tree.index);

    object_heap_print_state(&self->avl_node_heap, out);
}

int WARN_UNUSED
haplotype_matcher_alloc(haplotype_matcher_t *self, tree_sequence_t *tree_sequence,
        double recombination_rate)
{
    int ret = MSP_ERR_GENERIC;
    size_t avl_node_block_size = 8192; /* TODO make this a parameter? */

    memset(self, 0, sizeof(haplotype_matcher_t));
    self->tree_sequence = tree_sequence;
    self->recombination_rate = recombination_rate;
    self->num_sites = tree_sequence_get_num_sites(tree_sequence);
    self->num_nodes = tree_sequence_get_num_nodes(tree_sequence);
    self->recombination_dest = malloc(self->num_sites * sizeof(site_id_t));
    self->parent = malloc(self->num_nodes * sizeof(node_id_t));
    self->traceback = malloc(self->num_sites * sizeof(node_list_t *));
    if (self->recombination_dest == NULL || self->parent == NULL
            || self->traceback == NULL) {
        ret = MSP_ERR_NO_MEMORY;
        goto out;
    }
    /* The AVL node heap stores the avl node and the likelihood_t payload in
     * adjacent memory */
    ret = object_heap_init(&self->avl_node_heap,
            sizeof(avl_node_t) + sizeof(likelihood_t), avl_node_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    avl_init_tree(&self->likelihood, cmp_likelihood, NULL);
    ret = sparse_tree_alloc(&self->tree, self->tree_sequence, 0);
    if (ret != 0) {
        goto out;
    }
    ret = tree_diff_iterator_alloc(&self->diff_iterator, tree_sequence);
    if (ret != 0) {
        goto out;
    }
    ret = 0;
out:
    return ret;
}

int
haplotype_matcher_free(haplotype_matcher_t *self)
{
    msp_safe_free(self->recombination_dest);
    msp_safe_free(self->parent);
    msp_safe_free(self->traceback);
    object_heap_free(&self->avl_node_heap);
    sparse_tree_free(&self->tree);
    tree_diff_iterator_free(&self->diff_iterator);
    return 0;
}


static inline void
haplotype_matcher_free_avl_node(haplotype_matcher_t *self, avl_node_t *node)
{
    object_heap_free_object(&self->avl_node_heap, node);
}

static inline avl_node_t * WARN_UNUSED
haplotype_matcher_alloc_avl_node(haplotype_matcher_t *self, node_id_t node,
        double likelihood)
{
    avl_node_t *ret = NULL;
    likelihood_t *payload;

    if (object_heap_empty(&self->avl_node_heap)) {
        if (object_heap_expand(&self->avl_node_heap) != 0) {
            goto out;
        }
    }
    ret = (avl_node_t *) object_heap_alloc_object(&self->avl_node_heap);
    if (ret == NULL) {
        goto out;
    }
    /* We store the double value after the node */
    payload = (likelihood_t *) (ret + 1);
    payload->node = node;
    payload->likelihood = likelihood;
    avl_init_node(ret, payload);
out:
    return ret;
}

static int
haplotype_matcher_insert_likelihood(haplotype_matcher_t *self, node_id_t node,
        double likelihood)
{
    int ret = 0;
    avl_node_t *avl_node;

    avl_node = haplotype_matcher_alloc_avl_node(self, node, likelihood);
    if (avl_node == NULL) {
        ret = MSP_ERR_NO_MEMORY;
        goto out;
    }
    avl_node = avl_insert_node(&self->likelihood, avl_node);
    assert(avl_node != NULL);
out:
    return ret;
}

/* Returns the likelihood associated with the specified node or NO_LIKELIHOOD
 * if it does not exist.
 */
static double
haplotype_matcher_get_likelihood(haplotype_matcher_t *self, node_id_t node)
{
    double ret = NO_LIKELIHOOD;
    likelihood_t search;
    avl_node_t *avl_node;

    search.node = node;
    avl_node = avl_search(&self->likelihood, &search);
    if (avl_node != NULL) {
        ret = ((likelihood_t *) avl_node->item)->likelihood;
    }
    return ret;
}

static int
haplotype_matcher_delete_likelihood(haplotype_matcher_t *self, node_id_t node)
{
    likelihood_t search;
    avl_node_t *avl_node;

    search.node = node;
    avl_node = avl_search(&self->likelihood, &search);
    assert(avl_node != NULL);
    avl_unlink_node(&self->likelihood, avl_node);
    haplotype_matcher_free_avl_node(self, avl_node);
    return 0;
}

static int
haplotype_matcher_reset(haplotype_matcher_t *self, node_id_t *samples,
        size_t num_samples)
{
    size_t j;
    int ret;

    memset(self->parent, 0xff, self->num_nodes * sizeof(node_id_t));
    memset(self->recombination_dest, 0xff, self->num_sites * sizeof(node_id_t));
    memset(self->traceback, 0, self->num_sites * sizeof(node_list_t *));
    avl_clear_tree(&self->likelihood);
    for (j = 0; j < num_samples; j++) {
        ret = haplotype_matcher_insert_likelihood(self, samples[j], 1.0);
        if (ret != 0) {
            goto out;
        }
    }
    ret = sparse_tree_first(&self->tree);
    /* 1 is the expected condition here, as it indicates a tree is available */
    if (ret == 1) {
        ret = 0;
    }
out:
    return ret;
}

static int WARN_UNUSED
haplotype_matcher_update_tree_state(haplotype_matcher_t *self,
        node_record_t *records_out, node_record_t *records_in)
{
    int ret = 0;
    node_record_t *record;
    node_id_t parent, u, v, w, child, top;
    double x;
    double L_children[2];
    size_t num_L_children;
    list_len_t k;

    /* printf("RECORDS OUT\n"); */
    for (record = records_out; record != NULL; record = record->next) {
        for (k = 0; k < record->num_children; k++) {
            self->parent[record->children[k]] = MSP_NULL_NODE;
        }
        parent = record->node;
        x = haplotype_matcher_get_likelihood(self, parent);
        /* NOTE: we could save a search here by returning the AVL node in search */
        if (x == NO_LIKELIHOOD) {
            /* The children are now the roots of disconnected subtrees, and
             * need to be assigned L values. We set these by traversing up
             * the tree until we find the L value and then set this to the
             * children. */
            u = parent;
            while (u != MSP_NULL_NODE
                    && (x = haplotype_matcher_get_likelihood(self, u)) == NO_LIKELIHOOD) {
                u = self->parent[u];
            }
            if (u != MSP_NULL_NODE) {
                for (k = 0; k < record->num_children; k++) {
                    ret = haplotype_matcher_insert_likelihood(self, record->children[k], x);
                    if (ret != 0) {
                        goto out;
                    }
                }
            }
        } else {
            /* If we remove a node and it has an L value, then this L value is
             * mapped to its children. */
            haplotype_matcher_delete_likelihood(self, parent);
            for (k = 0; k < record->num_children; k++) {
                ret = haplotype_matcher_insert_likelihood(self, record->children[k], x);
                if (ret != 0) {
                    goto out;
                }
            }
        }
    }

    /* printf("AFTER OUT\n"); */
    /* haplotype_matcher_print_state(self, stdout); */

    /* printf("RECORDS IN\n"); */
    for (record = records_in; record != NULL; record = record->next) {
        parent = record->node;
        for (k = 0; k < record->num_children; k++) {
            self->parent[record->children[k]] = parent;
        }
        /* Short cut to avoid mallocs here. Easy enough to fix later. */
        assert(record->num_children == 2);
        num_L_children = 0;
        for (k = 0; k < record->num_children; k++) {
            child = record->children[k];
            x = haplotype_matcher_get_likelihood(self, child);
            if (x != NO_LIKELIHOOD) {
                L_children[num_L_children] = x;
                num_L_children++;
            }
        }
        /* Again, taking shortcuts for binary tree sequences here. fix. */
        if (num_L_children == 2 && L_children[0] == L_children[1]) {
            /* Coalesce the L values for the children into the parent */
            haplotype_matcher_delete_likelihood(self, record->children[0]);
            haplotype_matcher_delete_likelihood(self, record->children[1]);
            ret = haplotype_matcher_insert_likelihood(self, parent, L_children[0]);
            if (ret != 0) {
                goto out;
            }
        }
        if (num_L_children > 0) {
            /* Check for conflicts with L values higher in the tree */
            u = self->parent[parent];
            while (u != MSP_NULL_NODE
                    && (x = haplotype_matcher_get_likelihood(self, u)) == NO_LIKELIHOOD) {
                u = self->parent[u];
            }
            if (u != MSP_NULL_NODE) {
                haplotype_matcher_delete_likelihood(self, u);
                top = u;
                u = parent;
                /* Set the L value for the siblings of u as we traverse to top */
                while (u != top) {
                    v = self->parent[u];
                    for (k = 0; k < self->tree.num_children[v]; k++) {
                        w = self->tree.children[v][k];
                        if (w != u) {
                            ret = haplotype_matcher_insert_likelihood(self, w, x);
                            if (ret != 0) {
                                goto out;
                            }
                        }
                    }
                    u = v;
                }
            }
        }
    }
out:
    return ret;
}

static int WARN_UNUSED
haplotype_matcher_upate_site_state(haplotype_matcher_t *self, site_t *site, char state)
{
    int ret = 0;
    node_id_t mutation_node = site->mutations[0].node;

    assert(site->mutations_length == 1);
    assert(site->ancestral_state[0] == '0');
    printf("Updating for site %d, node = %d\n", site->id, mutation_node);

    return ret;
}

int WARN_UNUSED
haplotype_matcher_run(haplotype_matcher_t *self, char *haplotype,
        node_id_t *samples, size_t num_samples, node_id_t *path)
{
    int ret = 0;
    bool done = false;
    double length, left;
    node_record_t *records_out, *records_in;
    site_t *sites;
    list_len_t j, num_sites;

    ret = haplotype_matcher_reset(self, samples, num_samples);
    if (ret != 0) {
        goto out;
    }
    /* sparse_tree_first has been called by reset */
    /* This is used as a sanity check that our tree and the diff iterator
     * are in sync */
    left = 0;
    while (!done) {
        /* haplotype_matcher_print_state(self, stdout); */
        assert(left == self->tree.left);
        ret = tree_diff_iterator_next(&self->diff_iterator, &length, &records_out,
                &records_in);
        assert(ret == 1);
        left += length;
        ret = haplotype_matcher_update_tree_state(self, records_out, records_in);
        if (ret != 0) {
            goto out;
        }
        /* haplotype_matcher_print_state(self, stdout); */
        haplotype_matcher_check_state(self);

        ret = sparse_tree_get_sites(&self->tree, &sites, &num_sites);
        if (ret != 0) {
            goto out;
        }
        for (j = 0; j < num_sites; j++) {
            ret = haplotype_matcher_upate_site_state(self, &sites[j],
                    haplotype[sites[j].id]);
            if (ret != 0) {
                goto out;
            }
        }
        ret = sparse_tree_next(&self->tree);
        if (ret < 0) {
            goto out;
        }
        done = ret == 0;
    }
out:
    return ret;
}
