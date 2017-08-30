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
#include "block_allocator.h"
#include "msprime.h"

#define NULL_LIKELIHOOD (-1)
#define INVALID_PATH (-2)

static int
cmp_node_id(const void *a, const void *b) {
    const node_id_t *ia = (const node_id_t *) a;
    const node_id_t *ib = (const node_id_t *) b;
    return (*ia > *ib) - (*ia < *ib);
}

static void
haplotype_matcher_check_state(haplotype_matcher_t *self)
{
    size_t num_likelihoods;
    avl_node_t *a;
    node_id_t u;
    double x;

    /* Check the properties of the likelihood map */
    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        assert(self->likelihood[u] != NULL_LIKELIHOOD);
        x = self->likelihood[u];
        u = self->parent[u];
        if (u != MSP_NULL_NODE) {
            /* Traverse up to the next L value, and ensure it's not equal to x */
            while (self->likelihood[u] == NULL_LIKELIHOOD) {
                u = self->parent[u];
                assert(u != MSP_NULL_NODE);
            }
            assert(self->likelihood[u] != x);
        }
    }
    /* Make sure that there are no other non null likelihoods in the array */
    num_likelihoods = 0;
    for (u = 0; u < (node_id_t) self->num_nodes; u++) {
        if (self->likelihood[u] != NULL_LIKELIHOOD) {
            num_likelihoods++;
        }
    }
    assert(num_likelihoods == avl_count(&self->likelihood_nodes));
    assert(avl_count(&self->likelihood_nodes) ==
            object_heap_get_num_allocated(&self->avl_node_heap));
}

void
haplotype_matcher_print_state(haplotype_matcher_t *self, FILE *out)
{
    avl_node_t *a;
    likelihood_list_t *l;
    size_t j;
    node_id_t u;
    bool initialised = ((int) self->tree.index) != -1;

    fprintf(out, "recombination_rate = %g\n", self->recombination_rate);
    fprintf(out, "tree_sequence = %p\n", (void *) self->tree_sequence);
    fprintf(out, "likelihood = (%d)\n", (int) avl_count(&self->likelihood_nodes));
    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        fprintf(out, "%d\t->%g\n", u, self->likelihood[u]);
    }
    fprintf(out, "traceback\n");
    for (j = 0; j < self->num_sites; j++) {
        if (self->traceback[j] != NULL) {
            fprintf(out, "\t%d\t", (int) j);
            for (l = self->traceback[j]; l != NULL; l = l->next) {
                fprintf(out, "(%d, %f)", l->node, l->likelihood);
            }
            fprintf(out, "\n");
        }
    }
    fprintf(out, "tree = \n");
    fprintf(out, "\tindex = %d\n", (int) self->tree.index);
    object_heap_print_state(&self->avl_node_heap, out);
    block_allocator_print_state(&self->likelihood_list_allocator, out);

    if (initialised) {
        haplotype_matcher_check_state(self);
    }
}

int WARN_UNUSED
haplotype_matcher_alloc(haplotype_matcher_t *self, tree_sequence_t *tree_sequence,
        double recombination_rate)
{
    int ret = MSP_ERR_GENERIC;
    size_t avl_node_block_size = 8192; /* TODO make this a parameter? */
    size_t likelihood_list_block_size = 8192; /* TODO make this a parameter? */
    size_t j;
    site_t site;

    memset(self, 0, sizeof(haplotype_matcher_t));
    self->tree_sequence = tree_sequence;
    self->recombination_rate = recombination_rate;
    self->num_sites = tree_sequence_get_num_sites(tree_sequence);
    self->num_nodes = tree_sequence_get_num_nodes(tree_sequence);
    self->likelihood = malloc(self->num_nodes * sizeof(double));
    self->parent = malloc(self->num_nodes * sizeof(node_id_t));
    self->traceback = malloc(self->num_sites * sizeof(likelihood_list_t *));
    self->site_position = malloc(self->num_sites * sizeof(double));
    if (self->parent == NULL || self->likelihood == NULL
            || self->traceback == NULL || self->site_position == NULL) {
        ret = MSP_ERR_NO_MEMORY;
        goto out;
    }
    /* We only support tree sequences where all nodes are samples for now */
    if (tree_sequence_get_num_nodes(tree_sequence)
            != tree_sequence_get_sample_size(tree_sequence)) {
        ret = MSP_ERR_BAD_PARAM_VALUE;
        goto out;

    }
    /* The AVL node heap stores the avl node and the node_id_t payload in
     * adjacent memory. */
    ret = object_heap_init(&self->avl_node_heap,
            sizeof(avl_node_t) + sizeof(node_id_t), avl_node_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    avl_init_tree(&self->likelihood_nodes, cmp_node_id, NULL);
    ret = block_allocator_alloc(&self->likelihood_list_allocator,
            likelihood_list_block_size);
    if (ret != 0) {
        goto out;
    }
    ret = sparse_tree_alloc(&self->tree, self->tree_sequence, 0);
    if (ret != 0) {
        goto out;
    }
    ret = tree_diff_iterator_alloc(&self->diff_iterator, tree_sequence);
    if (ret != 0) {
        goto out;
    }
    /* Allocate the site positions for convenience during traceback */
    for (j = 0; j < self->num_sites; j++) {
        ret = tree_sequence_get_site(tree_sequence, (site_id_t) j, &site);
        assert(ret == 0);
        self->site_position[j] = site.position;
    }

    ret = 0;
out:
    return ret;
}

int
haplotype_matcher_free(haplotype_matcher_t *self)
{
    msp_safe_free(self->parent);
    msp_safe_free(self->likelihood);
    msp_safe_free(self->traceback);
    msp_safe_free(self->site_position);
    object_heap_free(&self->avl_node_heap);
    block_allocator_free(&self->likelihood_list_allocator);
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
haplotype_matcher_alloc_avl_node(haplotype_matcher_t *self, node_id_t node)
{
    avl_node_t *ret = NULL;
    node_id_t *payload;

    if (object_heap_empty(&self->avl_node_heap)) {
        if (object_heap_expand(&self->avl_node_heap) != 0) {
            goto out;
        }
    }
    ret = (avl_node_t *) object_heap_alloc_object(&self->avl_node_heap);
    if (ret == NULL) {
        goto out;
    }
    /* We store the node_id_t value after the avl_node */
    payload = (node_id_t *) (ret + 1);
    *payload = node;
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

    assert(likelihood >= 0);
    avl_node = haplotype_matcher_alloc_avl_node(self, node);
    if (avl_node == NULL) {
        ret = MSP_ERR_NO_MEMORY;
        goto out;
    }
    avl_node = avl_insert_node(&self->likelihood_nodes, avl_node);
    assert(self->likelihood[node] == NULL_LIKELIHOOD);
    assert(avl_node != NULL);
    self->likelihood[node] = likelihood;
out:
    return ret;
}

static int
haplotype_matcher_delete_likelihood(haplotype_matcher_t *self, node_id_t node)
{
    avl_node_t *avl_node;

    avl_node = avl_search(&self->likelihood_nodes, &node);
    assert(self->likelihood[node] != NULL_LIKELIHOOD);
    assert(avl_node != NULL);
    avl_unlink_node(&self->likelihood_nodes, avl_node);
    haplotype_matcher_free_avl_node(self, avl_node);
    self->likelihood[node] = NULL_LIKELIHOOD;
    return 0;
}

static int
haplotype_matcher_reset(haplotype_matcher_t *self, node_id_t *samples,
        size_t num_samples)
{
    size_t j;
    int ret;
    avl_node_t *a, *tmp;

    memset(self->parent, 0xff, self->num_nodes * sizeof(node_id_t));
    memset(self->traceback, 0, self->num_sites * sizeof(likelihood_list_t *));
    for (j = 0; j < self->num_nodes; j++) {
        self->likelihood[j] = NULL_LIKELIHOOD;
    }
    /* Set the recombination_dest to -1 and tracebacks to NULL. */
    memset(self->traceback, 0, self->num_sites * sizeof(likelihood_list_t *));
    ret = block_allocator_reset(&self->likelihood_list_allocator);
    if (ret != 0) {
        goto out;
    }
    a = self->likelihood_nodes.head;
    while (a != NULL) {
        tmp = a->next;
        haplotype_matcher_free_avl_node(self, a);
        a = tmp;
    }
    avl_clear_tree(&self->likelihood_nodes);

    /* Set the samples */
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
    tree_diff_iterator_reset(&self->diff_iterator);
    self->total_traceback_size = 0;
out:
    return ret;
}

static int WARN_UNUSED
haplotype_matcher_update_tree_state(haplotype_matcher_t *self,
        node_record_t *records_out, node_record_t *records_in)
{
    int ret = 0;
    node_record_t *record;
    node_id_t parent, u, v;
    double x;
    list_len_t k;

    /* printf("RECORDS OUT\n"); */
    for (record = records_out; record != NULL; record = record->next) {
        parent = record->node;
        u = parent;
        while (self->likelihood[u] == NULL_LIKELIHOOD) {
            u = self->parent[u];
            assert(u != MSP_NULL_NODE);
        }
        x = self->likelihood[u];
        for (k = 0; k < record->num_children; k++) {
            v = record->children[k];
            if (self->likelihood[v] == NULL_LIKELIHOOD) {
                ret = haplotype_matcher_insert_likelihood(self, v, x);
                if (ret != 0) {
                    goto out;
                }
            }
            self->parent[v] = MSP_NULL_NODE;
        }
    }
    /* printf("AFTER OUT\n"); */
    /* haplotype_matcher_print_state(self, stdout); */

    /* printf("RECORDS IN\n"); */
    for (record = records_in; record != NULL; record = record->next) {
        parent = record->node;
        u = parent;
        while (self->likelihood[u] == NULL_LIKELIHOOD) {
            u = self->parent[u];
            assert(u != MSP_NULL_NODE);
        }
        x = self->likelihood[u];
        for (k = 0; k < record->num_children; k++) {
            v = record->children[k];
            if (self->likelihood[v] == x) {
                haplotype_matcher_delete_likelihood(self, v);
            }
            self->parent[v] = parent;
        }
    }
out:
    return ret;
}

/* Update the likelihoods to account for a mutation at the specified node. We do not
 * change the values of the likelihoods here, just shift around the values associated
 * with nodes so that we easily update the actual values in the next step.
 */
static int WARN_UNUSED
haplotype_matcher_update_site_likelihood_nodes(haplotype_matcher_t *self,
        node_id_t mutation_node)
{
    int ret = 0;
    node_id_t u;

    if (self->likelihood[mutation_node] == NULL_LIKELIHOOD) {
        u = mutation_node;
        while (self->likelihood[u] == NULL_LIKELIHOOD) {
            u = self->parent[u];
            assert(u != MSP_NULL_NODE);
        }
        ret = haplotype_matcher_insert_likelihood(self, mutation_node,
                self->likelihood[u]);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

static int WARN_UNUSED
haplotype_matcher_update_site_likelihood_values(haplotype_matcher_t *self, site_t *site,
        char state, size_t num_samples)
{
    int ret = 0;
    double n = (double) num_samples;
    double r = 1 - exp(-self->recombination_rate / n);
    double recomb_proba = r / n;
    double no_recomb_proba = 1 - r + r / n;
    double *L = self->likelihood;
    double x, y, max_L, emission;
    bool is_descendant;
    node_id_t mutation_node = site->mutations[0].node;
    node_id_t u;
    avl_node_t *a;

    max_L = -1;
    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        x = L[u] * no_recomb_proba;
        assert(x >= 0);
        if (x > recomb_proba) {
            y = x;
        } else {
            y = recomb_proba;
        }
        is_descendant = sparse_tree_is_descendant(&self->tree, u, mutation_node);
        if (state == '1') {
            emission = (double) is_descendant;
        } else {
            emission = (double) (! is_descendant);
        }
        L[u] = y * emission;
        if (L[u] > max_L) {
            max_L = L[u];
        }
        /* printf("x = %f, y = %f, emission = %f\n", x, y, emission); */
        /* printf("state = %d %d: likelihood = %f\n", state, '1', L[u]); */
    }
    assert(max_L > 0);
    /* Normalise */
    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        L[u] /= max_L;
    }
    return ret;
}

static int WARN_UNUSED
haplotype_matcher_coalesce_likelihoods(haplotype_matcher_t *self)
{
    int ret = 0;
    avl_node_t *a, *tmp;
    node_id_t u, v;

    a = self->likelihood_nodes.head;
    while (a != NULL) {
        tmp = a->next;
        u = *((node_id_t *) a->item);
        if (self->parent[u] != MSP_NULL_NODE) {
            /* If we can find an equal L value higher in the tree, delete
             * this one.
             */
            v = self->parent[u];
            while (self->likelihood[v] == NULL_LIKELIHOOD) {
                v = self->parent[v];
                assert(v != MSP_NULL_NODE);
            }
            if (self->likelihood[u] == self->likelihood[v]) {
                /* Delete this likelihood value */
                avl_unlink_node(&self->likelihood_nodes, a);
                haplotype_matcher_free_avl_node(self, a);
                self->likelihood[u] = NULL_LIKELIHOOD;
            }
        }
        a = tmp;
    }
    return ret;
}

/* Store the current state of the likelihood tree in the traceback.
 */
static int WARN_UNUSED
haplotype_matcher_store_traceback(haplotype_matcher_t *self, site_id_t site_id)
{
    int ret = 0;
    avl_node_t *a;
    node_id_t u;
    likelihood_list_t *list_node;

    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        list_node = block_allocator_get(&self->likelihood_list_allocator,
                sizeof(likelihood_list_t));
        if (list_node == NULL) {
            ret = MSP_ERR_NO_MEMORY;
            goto out;
        }
        list_node->node = u;
        list_node->likelihood = self->likelihood[u];
        list_node->next = self->traceback[site_id];
        self->traceback[site_id] = list_node;
    }
    self->total_traceback_size += avl_count(&self->likelihood_nodes);
out:
    return ret;
}

static int WARN_UNUSED
haplotype_matcher_update_site_state(haplotype_matcher_t *self, site_t *site,
        char state, size_t num_samples)
{
    int ret = 0;
    node_id_t mutation_node;

    assert(site->ancestral_state[0] == '0');
    if (site->mutations_length == 0) {
        ret = haplotype_matcher_store_traceback(self, site->id);
        if (ret != 0) {
            goto out;
        }
        /* TODO throw an error or something */
        assert(state == '0');
    } else {
        assert(site->mutations_length == 1);
        mutation_node = site->mutations[0].node;
        /* printf("Updating for site %d, node = %d state = %c\n", site->id, */
        /*         mutation_node, state); */
        ret = haplotype_matcher_update_site_likelihood_nodes(self, mutation_node);
        if (ret != 0) {
            goto out;
        }
        ret = haplotype_matcher_store_traceback(self, site->id);
        if (ret != 0) {
            goto out;
        }
        ret = haplotype_matcher_update_site_likelihood_values(self, site, state,
                num_samples);
        if (ret != 0) {
            goto out;
        }
        ret = haplotype_matcher_coalesce_likelihoods(self);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

static void
haplotype_matcher_position_tree(haplotype_matcher_t *self, double x)
{
    int ret;
    sparse_tree_t *tree = &self->tree;

    assert(x < tree->right);

    /* printf("x = %f, left = %f\n", x, tree->left); */
    while (x < tree->left) {
        ret = sparse_tree_prev(tree);
        assert(ret == 1);
    }
    assert(tree->left <= x && x < tree->right);

    /* printf("tree positioned %d (%f, %f) for site %f\n", (int) tree->index, tree->left, */
            /* tree->right, x); */
}

static node_id_t
haplotype_matcher_choose_sample(haplotype_matcher_t *self, site_id_t site)
{
    likelihood_list_t *z;

    z = self->traceback[site];
    assert(z != NULL);
    while (z->likelihood != 1.0) {
        z = z->next;
        assert(z != NULL);
    }
    return z->node;
}

static double
haplotype_matcher_get_likelihood(haplotype_matcher_t *self, site_id_t site, node_id_t u)
{
    double ret = -1;
    double *L = self->likelihood;
    node_id_t v;
    likelihood_list_t *z;

    assert(self->tree.left <= self->site_position[site]);
    assert(self->site_position[site] < self->tree.right);

    /* Set the likelihood values so that we can find them quickly */
    for (z = self->traceback[site]; z != NULL; z = z->next) {
        L[z->node] = z->likelihood;
    }

    /* Traverse up the tree until we find an L value for u */
    while (L[u] == NULL_LIKELIHOOD) {
        ret = sparse_tree_get_parent(&self->tree, u, &v);
        assert(ret == 0);
        u = v;
        assert(u != MSP_NULL_NODE);
    }
    ret = L[u];

    /* Reset L */
    for (z = self->traceback[site]; z != NULL; z = z->next) {
        L[z->node] = -1;
    }
    return ret;
}

static int WARN_UNUSED
haplotype_matcher_traceback(haplotype_matcher_t *self, node_id_t *path)
{
    int ret = 0;
    site_id_t l = (site_id_t) self->num_sites - 1;
    double *position = self->site_position;
    double x;
    avl_node_t *a, *tmp;
    node_id_t u;

    ret = sparse_tree_last(&self->tree);
    assert(ret == 1);
    /* haplotype_matcher_print_state(self, stdout); */
    /* printf("TRACEBACK\n"); */

    haplotype_matcher_position_tree(self, position[l]);
    /* Choose the initial sample and reset the likelihoods so that we can
     * reused the buffer during traceback. Free the avl nodes too so that
     * we can check the consistency of the arrays during/after traceback */

    /* NOTE!! All of this assumes that all nodes are samples. This will have
     * to be made quite a bit more sophisticated when we have samples
     * distributed around the tree in awkward places.
     */
    a = self->likelihood_nodes.head;
    path[l] = -1;
    while (a != NULL) {
        tmp = a->next;
        u = *((node_id_t *) a->item);
        if (self->likelihood[u] == 1.0) {
            path[l] = u;
        }
        self->likelihood[u] = NULL_LIKELIHOOD;
        haplotype_matcher_free_avl_node(self, a);
        a = tmp;
    }
    avl_clear_tree(&self->likelihood_nodes);
    assert(path[l] != -1);

    while (l > 0) {
        u = path[l];
        /* printf("LOOP: l = %d, u = %d\n",(int) l, u); */
        haplotype_matcher_position_tree(self, position[l]);
        x = haplotype_matcher_get_likelihood(self, l, u);
        if (x != 1.0) {
            u = haplotype_matcher_choose_sample(self, l);
        }
        l--;
        path[l] = u;
    }
    ret = 0;
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
    node_id_t *S;

    /* If we provide an input of 0 samples this is interpreted as all samples. */
    S = samples;
    if (num_samples == 0) {
        num_samples = tree_sequence_get_sample_size(self->tree_sequence);
        ret = tree_sequence_get_samples(self->tree_sequence, &S);
        if (ret != 0) {
            goto out;
        }
    }
    ret = haplotype_matcher_reset(self, S, num_samples);
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
            ret = haplotype_matcher_update_site_state(self, &sites[j],
                    haplotype[sites[j].id], num_samples);
            if (ret != 0) {
                goto out;
            }
            /* haplotype_matcher_print_state(self, stdout); */
            haplotype_matcher_check_state(self);
        }
        ret = sparse_tree_next(&self->tree);
        if (ret < 0) {
            goto out;
        }
        done = ret == 0;
    }
    ret = haplotype_matcher_traceback(self, path);
out:
    return ret;
}

double
haplotype_matcher_get_mean_traceback_size(haplotype_matcher_t *self)
{
    return ((double) self->total_traceback_size) / ((double) self->num_sites);
}
