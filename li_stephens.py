"""
Implemenation of the Li and Stephens algorithm on a tree sequence.
sequence.
"""
import random
import sys
import time

import six
import numpy as np

import msprime
import _msprime

def random_breakpoint():
    return min(1.0, max(0.0, 2 * random.random() - 0.5))


def wf_sim(N, ngens, survival=0.0, mutation_rate=0.0, seed=1):
    """
    SIMPLE simulation of a bisexual, haploid Wright-Fisher population of size N
    for ngens generations, in which each individual survives with probability
    survival and only those who die are replaced.  The chromosome is 1.0
    Morgans long, and the mutation rate is in units of mutations/Morgan/generation.
    """
    random.seed(seed)
    nodes = msprime.NodeTable()
    edgesets = msprime.EdgesetTable()
    for _ in range(N):
        nodes.add_row(time=ngens)

    pop = list(range(N))
    for t in range(ngens - 1, -1, -1):
        dead = [random.random() > survival for k in pop]
        # sample these first so that all parents are from the previous gen
        new_parents = [
            (random.choice(pop), random.choice(pop)) for k in range(sum(dead))]
        k = 0
        for j in range(N):
            if dead[j]:
                # this is: offspring ID, lparent, rparent, breakpoint
                offspring = nodes.num_rows
                nodes.add_row(time=t)
                lparent, rparent = new_parents[k]
                k += 1
                bp = random_breakpoint()
                pop[j] = offspring
                if bp > 0.0:
                    edgesets.add_row(
                        left=0.0, right=bp, parent=lparent, children=(offspring,))
                if bp < 1.0:
                    edgesets.add_row(
                        left=bp, right=1.0, parent=rparent, children=(offspring,))
    flags = [
        (msprime.NODE_IS_SAMPLE if u in pop else 0) for u in range(nodes.num_rows)]
    nodes.set_columns(time=nodes.time, flags=flags)
    msprime.sort_tables(nodes, edgesets)
    msprime.simplify_tables(pop, nodes, edgesets)
    sites = msprime.SiteTable()
    mutations = msprime.MutationTable()
    rng = msprime.RandomGenerator(seed)
    mutgen = msprime.MutationGenerator(rng, mutation_rate)
    mutgen.generate(nodes, edgesets, sites, mutations)
    return msprime.load_tables(
        nodes=nodes, edgesets=edgesets, sites=sites, mutations=mutations)




if sys.version_info[0] < 3:
    raise Exception("Python 3 you idiot!")

def best_path(h, H, recombination_rate):
    n, m = H.shape
    r = 1 - np.exp(-recombination_rate / n)
    recomb_proba = r / n
    no_recomb_proba = 1 - r + r / n

    L = np.ones(n)
    # This returns the full likelihoods to help develop the tree likelihood methods.
    L_save = np.zeros((m, n))
    T = [set() for _ in range(m)]
    T_dest = np.zeros(m, dtype=int)

    for l in range(m):
        L_save[l] = L
        L_next = np.zeros(n)
        for j in range(n):
            x = L[j] * no_recomb_proba
            y = recomb_proba
            if x > y:
                z = x
            else:
                z = y
                T[l].add(j)
            # if H[j, l] == h[l]:
            #     emission_p = 1
            # else:
            #     # Hack: to get a nonzero likelihood when we can't find a match
            #     # stick in some tiny value.
            #     emission_p = 1e-200
            emission_p = int(H[j, l] == h[l])
            L_next[j] = z * emission_p
        # Find the max and renormalise
        L = L_next
        j = np.argmax(L)
        T_dest[l] = j
        L /= L[j]
    P = np.zeros(m, dtype=int)
    P[m - 1] = T_dest[m - 1]
    mismatches = []
    for l in range(m - 1, 0, -1):
        j = P[l]
        if H[j, l] != h[l]:
            mismatches.append(l)
        if j in T[l]:
            assert l != 0
            j = T_dest[l - 1]
        P[l - 1] = j
    if H[j, 0] != h[0]:
        mismatches.append(0)
    return P, mismatches, L_save


def get_likelihood(u, L, tree):
    """
    Gets the likelihood of the specified node in the specified likelihood
    encoding tree.
    """
    v = u
    while v not in L:
        v = tree.parent(v)
    return L[v]

def get_likelihood_at_site(ts, site_id, u, L):
    trees = ts.trees()
    tree = next(trees)
    position = list(ts.sites())[site_id].position
    while tree.interval[1] < position:
        tree = next(trees)
    left, right = tree.interval
    assert left <= position < right
    return get_likelihood(u, L, tree)


def best_path_ts(h, ts, recombination_rate):
    n, m = ts.sample_size, ts.num_sites
    r = 1 - np.exp(-recombination_rate / n)
    recomb_proba = r / n
    no_recomb_proba = 1 - r + r / n

    sites = list(ts.sites())
    L = {u: 1.0 for u in ts.samples()}
    all_L = [None for _ in range(m)]
    tree_index = [None for _ in range(m)]

    for tree, diff in zip(ts.trees(sample_counts=True), ts.diffs()):
        # print("new tree:, index = ", tree.index, "L = ", L)
        _, records_out, records_in = diff
        for parent, children, _ in records_out:
            # print("OUT:", parent, children)
            u = parent
            while u not in L:
                u = tree.parent(u)
            x = L[u]
            # Delete L value for parent if it is a non-sample root.
            # TODO just dealing with all internal samples for now.
            # if tree.parent(parent) == msprime.NULL_NODE and not tree.is_sample(parent):
            #     del L[parent]
            for c in children:
                if c not in L:
                    # TODO need to reason about non-sample leaves here
                    # at some point.
                    L[c] = x

        for parent, children, _ in records_in:
            u = parent
            while u not in L:
                u = tree.parent(u)
            x = L[u]
            # print("IN:", parent, children)
            for c in children:
                if c in L:
                    if L[c] == x:
                        del L[c]

        # print("New L = ", L)

        for site in tree.sites():
            l = site.index
            tree_index[l] = tree.index
            if len(site.mutations) == 0:
                all_L[l] = dict(L)
                continue

            mutation_node = site.mutations[0].node
            state = h[l]
            # Insert an new L-value for the mutation node if needed.
            if mutation_node not in L:
                u = mutation_node
                while u not in L:
                    u = tree.parent(u)
                L[mutation_node] = L[u]
            all_L[l] = dict(L)

            # Update the likelihoods for this site.
            max_L = -1
            for v in L.keys():
                x = L[v] * no_recomb_proba
                assert x >= 0
                y = recomb_proba
                if x > y:
                    z = x
                else:
                    z = y
                if state == 1:
                    emission_p = int(is_descendent(tree, v, mutation_node))
                else:
                    emission_p = int(not is_descendent(tree, v, mutation_node))
                L[v] = z * emission_p
                if L[v] > max_L:
                    max_L = L[v]
            assert max_L > 0

            # Normalise
            for v in L.keys():
                L[v] /= max_L
            # Compress
            # TODO we probably don't need the second dict here and can just take
            # a copy of the keys.
            L_next = {}
            for u in L.keys():
                if u != tree.root:
                    # Traverse upwards until we find another L value
                    v = tree.parent(u)
                    while v not in L:
                        v = tree.parent(v)
                    if L[u] != L[v]:
                        L_next[u] = L[u]
                else:
                    L_next[u] = L[u]
            L = L_next
            # print("Site: ", l, "mutation_node = ", mutation_node, "done:", L)

    # print("last L = ", L)
    u = [k for k, v in L.items() if v == 1.0][0]
    p = np.zeros(m, dtype=int)
    trees = ts.trees()
    tree = next(trees)
    while tree.index != tree_index[l]:
        tree = next(trees)
    assert tree.is_sample(u)
    p[l] = u
    for l in range(m - 1, 0, -1):
        # print("tracing back at ", l, "u = ", p[l], "tree_index = ", tree_index[l], all_L[l])
        if tree.index != tree_index[l]:
            # print("Repositioning tree")
            trees = ts.trees()
            tree = next(trees)
            while tree.index != tree_index[l]:
                tree = next(trees)
        u = p[l]
        # print("Getting likelihood for ", u)
        x = get_likelihood(u, all_L[l], tree)
        if x != 1.0:
            u = [k for k, v in all_L[l].items() if v == 1.0][0]
            # print("\tSwitch to node", u)
            assert tree.is_sample(u)
        p[l - 1] = u
    return p


def is_descendent(tree, u, v):
    """
    Returns True if the specified node u is a descendent of node v. That is,
    v is on the path to root from u.
    """
    # print("IS_DESCENDENT(", u, v, ")")
    while u != v and u != msprime.NULL_NODE:
        # print("\t", u)
        u = tree.parent(u)
    # print("END, ", u, v)
    return u == v

def is_descendent_at_site(ts, site_id, u, v):
    trees = ts.trees()
    tree = next(trees)
    position = list(ts.sites())[site_id].position
    while tree.interval[1] < position:
        tree = next(trees)
    left, right = tree.interval
    assert left <= position < right
    return is_descendent(tree, u, v)



class HaplotypeMatcher(object):

    def __init__(self, tree_sequence, recombination_rate, samples=None):
        self.tree_sequence = tree_sequence
        self.tree = None
        if samples is None:
            samples = list(self.tree_sequence.samples())
        self.samples = samples
        self.num_sites = tree_sequence.num_sites
        self.num_nodes = tree_sequence.num_nodes
        self.recombination_rate = recombination_rate
        # Map of tree nodes to likelihoods. We maintain the property that the
        # nodes in this map are non-overlapping; that is, for any u in the map,
        # there is no v that is an ancestor of u.
        self.likelihood = np.zeros(self.num_nodes) - 1
        # This is the set of nodes that are currently set in the likelihood map.
        self.likelihood_nodes = set()
        # We keep a local copy of the parent array to allow us maintain the
        # likelihood map between tree transitions.
        self.parent = np.zeros(self.num_nodes, dtype=int) - 1
        # For each locus, store a set of nodes at which we must recombine during
        # traceback.
        self.traceback = [[] for _ in range(self.num_sites)]
        # If we recombine during traceback, this is the node we recombine to.
        self.recombination_dest = np.zeros(self.num_sites, dtype=int) - 1
        # This is the buffer used to propagate L values up the tree.
        self.compression_buffer = np.zeros(self.num_nodes, dtype=int) - 1

    def reset(self, samples):
        self.likelihood_nodes = set()
        self.likelihood[:] = -1
        for u in self.samples:
            self.likelihood_nodes.add(u)
            self.likelihood[u] = 0.0
        for u in samples:
            self.likelihood[u] = 1.0
        self.parent[:] = -1
        self.traceback = [[] for _ in range(self.num_sites)]
        self.recombination_dest[:] = -1

    def print_state(self, traceback=True):
        print("HaplotypeMatcher state")
        print("likelihood:")
        for u in sorted(self.likelihood_nodes):
            print("\t", u, "->", self.likelihood[u])
        print("tree = ", repr(self.tree))
        if self.tree is not None:
            print("\tindex = ", self.tree.index)
            print("\tnum_sites = ", len(list(self.tree.sites())))
            print("\tp = ", self.tree.parent_dict)
        if traceback:
            print("Traceback:")
            for l in range(self.num_sites):
                print("\t", l, "\t", self.recombination_dest[l], "\t", self.traceback[l])

    def check_sample_coverage(self, nodes):
        """
        Ensures that all the samples from the specified tree are covered by the
        set of nodes.
        """
        samples = set()
        for u in nodes:
            # TODO assuming here that all nodes are samples.
            # subtree_samples = set(self.tree.nodes(u))
            subtree_samples = set(self.tree.leaves(u))
            assert len(subtree_samples & samples) == 0
            samples |= subtree_samples
        print("nodes = ", nodes)
        print("samples = ", samples)
        assert samples == set(self.tree_sequence.samples())

    def check_partial_tree_consistency(self):
        """
        Given the partially udpated tree in the parent array, ensure that the
        L values still have the correct properties.
        """
        P = self.parent
        # Build the children array
        C = [[] for _ in P]
        for u, v in enumerate(P):
            if v != -1:
                C[v].append(u)
        # for u in range(P.shape[0]):
        #     if len(C[u]) > 0:
        #         print(u, "->", C[u])
        for u in self.likelihood_nodes:
            # traverse down from here. We should not meet any other L values.
            stack = list(C[u])
            while len(stack) > 0:
                v = stack.pop()
                stack.extend(C[v])
                assert v not in self.likelihood_nodes

    def check_state(self):
        # print("AFTER IN:", L_tree)
        ts = self.tree_sequence
        P_dict = {
            u: self.parent[u] for u in range(ts.num_nodes) if self.parent[u] != -1}
        assert self.tree.parent_dict == P_dict
        self.check_sample_coverage(self.likelihood_nodes)
        # Check that the likelihood nodes are correctly mapped.
        for u in self.likelihood_nodes:
            assert self.likelihood[u] != -1
        for u in range(self.num_nodes):
            if self.likelihood[u] != -1:
                assert u in self.likelihood_nodes
        # print("DONE")
        assert np.all(self.compression_buffer == -1)

    def update_tree_state(self, diff):
        """
        Update the state of the likelihood map to reflect the new tree. We use
        the diffs to efficiently migrate the likelihoods from nodes in the previous
        tree to the new tree.
        """
        _, records_out, records_in = diff
        for parent, children, _ in records_out:
            # print("OUT:", parent, children)
            for c in children:
                self.parent[c] = msprime.NULL_NODE
            x = self.likelihood[parent]
            if x != -1:
                # If we remove a node and it has an L value, then this L value is
                # mapped to its children.
                self.likelihood[parent] = -1
                self.likelihood_nodes.remove(parent)
                for c in children:
                    assert self.likelihood[c] == -1
                    assert c not in self.likelihood_nodes
                    self.likelihood[c] = x
                    self.likelihood_nodes.add(c)
            else:
                # The children are now the roots of disconnected subtrees, and
                # need to be assigned L values. We set these by traversing up
                # the tree until we find the L value and then set this to the
                # children.
                u = parent
                while u != -1 and self.likelihood[u] == -1:
                    u = self.parent[u]
                # TODO It's not completely clear to me what's happening in the
                # case where u is -1. The logic of this section can be clarified
                # here I think as we should be setting values for the children
                # in all cases where they do not have L values already.
                if u != -1:
                    x = self.likelihood[u]
                    for c in children:
                        assert c not in self.likelihood_nodes
                        assert self.likelihood[c] == -1
                        self.likelihood[c] = x
                        self.likelihood_nodes.add(c)

        self.check_partial_tree_consistency()

        # TODO we are not correctly coalescing all equal valued L values among
        # children here. Definitely need another pass at this algorithm to
        # make it more elegant and catch all the corner cases.
        for parent, children, _ in records_in:
            for c in children:
                self.parent[c] = parent
            # Coalesce the L values for children if possible.
            # TODO this is ugly and inefficient. Need a simpler approach.
            L_children = []
            for c in children:
                if self.likelihood[c] != -1:
                    L_children.append(self.likelihood[c])
            if len(L_children) == len(children) and len(set(L_children)) == 1:
                # TODO I don't understand how we can have a likelihood value
                # for this parent already, but it does happen. Definitely need
                # improve this algorithm.
                if self.likelihood[parent] == -1:
                    self.likelihood[parent] = self.likelihood[children[0]]
                    self.likelihood_nodes.add(parent)
                else:
                    assert self.likelihood[parent] == self.likelihood[children[0]]
                for c in children:
                    self.likelihood_nodes.remove(c)
                    self.likelihood[c] = -1
            if len(L_children) > 0:
                # Need to check for conflicts with L values higher in the tree.
                u = self.parent[parent]
                while u != msprime.NULL_NODE and self.likelihood[u] == -1:
                    u = self.parent[u]
                if u != msprime.NULL_NODE:
                    top = u
                    x = self.likelihood[u]
                    self.likelihood[u] = -1
                    self.likelihood_nodes.remove(u)
                    u = parent
                    while u != top:
                        v = self.parent[u]
                        for w in self.tree.children(v):
                            if w != u:
                                assert w not in self.likelihood_nodes
                                assert self.likelihood[w] == -1
                                self.likelihood[w] = x
                                self.likelihood_nodes.add(w)
                        u = v

    def add_recombination_node(self, site_id, u):
        """
        Adds a recombination node for the specified site.
        """
        self.traceback[site_id].append(u)

    def choose_recombination_destination(self, site_id):
        """
        Given the state of the likelihoods, choose the destination for
        haplotypes recombining onto this site.
        """
        # Find a node with L == 1 and register as the recombinant haplotype root.
        found = False
        for u in self.likelihood_nodes:
            value = self.likelihood[u]
            if value == 1.0:
                self.recombination_dest[site_id] = u
                found = True
                break
        assert found

    def coalesce_equal(self):
        """
        Coalesce L values into the minimal representation by propagating
        values up the tree and finding the roots of the subtrees sharing
        the same L value.
        """
        # TODO need to make these algorithms such that they self repair
        # the V map and so that we automatically null out elements of the
        # likelihood array that are remove. At the moment we incur two
        # O(n) operations, which is poor.
        tree = self.tree
        # Coalesce equal values
        V = self.compression_buffer
        S = set(self.likelihood_nodes)
        # Take all the L values an propagate them up the tree.
        for u in S:
            x = self.likelihood[u]
            self.likelihood_nodes.remove(u)
            self.likelihood[u] = -1
            while u != msprime.NULL_NODE and V[u] == -1:
                V[u] = x
                u = self.parent[u]
            if u != msprime.NULL_NODE and V[u] != x:
                # Mark the path up to root as invalid
                while u!= msprime.NULL_NODE:
                    V[u] = -2
                    u = self.parent[u]
        assert len(self.likelihood_nodes) == 0
        for u in S:
            x = V[u]
            last_u = u
            while u != msprime.NULL_NODE and V[u] != -2:
                last_u = u
                u = self.parent[u]
            if x != -2 and self.likelihood[last_u] == -1:
                self.likelihood[last_u] = x
                self.likelihood_nodes.add(last_u)
        # Reset V
        for u in S:
            while u != msprime.NULL_NODE and V[u] != -1:
                V[u] = -1
                u = self.parent[u]



    def update_site(self, site, state):
        """
        Updates the algorithm state for the specified site given the specified
        input state.
        """
        assert len(site.mutations) == 1
        assert site.ancestral_state == '0'
        mutation_node = site.mutations[0].node
        n = len(self.samples)
        r = 1 - np.exp(-self.recombination_rate / n)
        recomb_proba = r / n
        no_recomb_proba = 1 - r + r / n

        L = self.likelihood
        S = self.likelihood_nodes
        tree = self.tree
        S_nodes = list(S)
        # Update L to add nodes for the mutation node, splitting and removing
        # existing L nodes as necessary.
        for node in S_nodes:
            value = L[node]
            if is_descendent(tree, mutation_node, node):
                L[node] = -1
                S.remove(node)
                L[mutation_node] = value
                S.add(mutation_node)
                # Traverse upwards until we reach old L node, adding values
                # for the siblings off the path.
                u = mutation_node
                while u != node:
                    v = tree.parent(u)
                    for w in tree.children(v):
                        if w != u:
                            L[w] = value
                            S.add(w)
                    u = v
        # Update the likelihoods for this site.
        max_L = -1
        for v in self.likelihood_nodes:
            x = L[v] * no_recomb_proba
            assert x >= 0
            y = recomb_proba
            if x > y:
                z = x
            else:
                z = y
                self.add_recombination_node(site.index, v)
            if state == 1:
                emission_p = int(is_descendent(tree, v, mutation_node))
            else:
                emission_p = int(not is_descendent(tree, v, mutation_node))
            L[v] = z * emission_p
            if L[v] > max_L:
                max_L = L[v]
        assert max_L > 0

        # Normalise
        for v in self.likelihood_nodes:
            L[v] /= max_L
        # print("BEFORE COAL")
        # self.print_state(traceback=False)
        self.coalesce_equal()
        # print("AFTER COAL")
        # self.print_state(traceback=False)
        self.choose_recombination_destination(site.index)


    def map_sample(self, site_id, node):
        """
        Maps the specified node for the specified site to a sample node.
        """
        trees = self.tree_sequence.trees()
        tree = next(trees)
        position = list(self.tree_sequence.sites())[site_id].position
        while tree.interval[1] < position:
            tree = next(trees)
        left, right = tree.interval
        assert left <= position < right
        u = node
        while not tree.is_leaf(u):
            u = tree.children(u)[0]
        node = self.tree_sequence.node(u)
        assert node.is_sample
        return u

    def run_traceback(self, p):
        m = self.num_sites
        p[m - 1] = self.map_sample(m - 1, self.recombination_dest[m - 1])

        for l in range(m - 1, 0, -1):
            u = p[l]
            for v in self.traceback[l]:
                if is_descendent_at_site(self.tree_sequence, l, u, v):
                    assert l != 0
                    # print("RECOMBINING at ", l, ":", j, u, T_dest_tree[l - 1])
                    u = self.map_sample(l - 1, self.recombination_dest[l - 1])
                    break
            p[l - 1] = u
        return p

    def run(self, samples, haplotype, path):
        self.reset(samples)
        ts = self.tree_sequence
        self.print_state()
        for tree, diff in zip(ts.trees(), ts.diffs()):
            self.tree = tree
            self.update_tree_state(diff)
            self.tree.draw("t{}.svg".format(self.tree.index),
                    width=800, height=800, mutation_locations=False)
            self.print_state()
            self.check_state()
            # self.check_state()
            for site in tree.sites():
                print("Update site", site)
                if len(site.mutations) > 0:
                    self.update_site(site, haplotype[site.index])
                else:
                    assert haplotype[site.index] == 0
                self.print_state()
                self.check_state()
        # self.print_state()
        self.run_traceback(path)


def random_mosaic(H):
    n, m = H.shape
    h = np.zeros(m, dtype=np.int8)
    for l in range(m):
        h[l] = H[random.randint(0, n - 1), l]
    return h

def copy_process_dev(n, L, seed):
    random.seed(seed)
    ts = msprime.simulate(
        n, length=L, mutation_rate=1, recombination_rate=1, random_seed=seed)
    m = ts.num_sites
    print("n = {} m = {}".format(n, m))

    H = np.zeros((n, m), dtype=np.int8)
    for v in ts.variants():
        H[:, v.index] = v.genotypes

    matcher = HaplotypeMatcher(ts, recombination_rate=1e-8)
    # matcher = _msprime.HaplotypeMatcher(ts._ll_tree_sequence,
    #         recombination_rate=1e-8, mutation_rate=1e-8)
    p = np.zeros(m, dtype=np.int32)

    # print(H)
    for j in range(n):
        # h = random_mosaic(H) + ord('0')
        # print(h)
        # h = np.hstack([H[0,:10], H[1,10:]])
        # print()
        # print(h)
        h = H[j]
        H_minus = np.vstack([H[:j], H[j + 1:]])
        # print(H_minus)
        p, mismatches, L = best_path(h, H_minus, 1e-8)
        print("mismatches = ", mismatches)
        # p = best_path_ts(h, ts, 1e-8)
        # before = time.clock()
        matcher.run([k for k in range(n) if k != j], h, p)
        # duration = time.clock() - before
        # print("Done in {} seconds".format(duration))

        # hp = H[p, np.arange(m)] + ord('0')
        print(p)
        hp = H_minus[p, np.arange(m)]
        for l in mismatches:
            assert hp[l] != h[l]
            hp[l] = h[l]
        print(h)
        print(hp)
        # print("p = ", p)
        # print()
        # print(h - ord('0'))
        # print(hp - ord('0'))
        assert np.array_equal(h, hp)


def compress_likelihoods(tree, L):
    # N = L.shape[0]
    # samples = list(tree.samples())
    # print("Input = ")
    # for u in tree.samples():
    #     print("\t", u, "->", L[u])
    # print("samples = ", samples)
    # print("mapping", L, "to ", tree)
    Q = {}
    # for each sample, traverse upwards to the root.
    for u in tree.nodes(order="postorder"):
        Q[u] = L[u]
        for v in tree.children(u):
            if Q[v] == Q[u]:
                del Q[v]
    # print("Output")
    # for u in tree.samples():
    #     if u in Q:
    #         print("\t", u, "->", Q[u])
    return Q


def ancestral_sample_match_dev(n, seed):

    ts = msprime.load("ancestors_example-2.hdf5")
    # ts = msprime.load("ancestors_example.hdf5")
    # for t in ts.trees():
    #     t.draw("t{}.svg".format(t.index),
    #             width=800, height=800, mutation_locations=False)

    n = ts.sample_size
    m = ts.num_sites
    print("n = {} m = {}".format(n, m))
    H = np.zeros((n, m), dtype=np.int8)
    for v in ts.variants():
        H[:, v.index] = v.genotypes
    # print(H)

    # matcher = HaplotypeMatcher(ts, recombination_rate=1e-8)

    # # h = random_mosaic(H) + ord('0')
    matcher = _msprime.HaplotypeMatcher(ts._ll_tree_sequence, recombination_rate=1e-8)
    for j in range(10000):
        # print(j)
        random.seed(j)
        h = random_mosaic(H)
        # print()
        # print("h = ", h)

        # print("".join(str(x) for x in h))
        # print()


        # # # h = np.hstack([H[0,:10], H[1,10:]])
        # # # print()
        # # # print(h)
        # p, mismatches, L = best_path(h, H, 1e-8)
        # hp = H[p, np.arange(m)]
        # # print(p)
        # # print(h)
        # # print(hp)
        # assert np.array_equal(h, hp)
        # p = best_path_ts(h, ts, 1e-8)

        # hp = H[p, np.arange(m)]
        # # print(p)
        # # print(h)
        # # print(hp)
        # # print()
        # # print(np.where(h != hp))
        # assert np.array_equal(h, hp)

        p = np.zeros(m, dtype=np.int32)
        matcher.run(h + ord('0'), p)

        hp = H[p, np.arange(m)]

        # print(p)
        # print(h)
        # print(hp)

        assert np.array_equal(h, hp)

        # print()
        # print(np.where(h != hp))



    # for tree in ts.trees():
    #     for site in tree.sites():
    #         print("site", site.index, ": u = ", site.mutations[0].node,
    #                 "state = ", h[site.index],
    #                 compress_likelihoods(tree, L[site.index]))
    #         compress_likelihoods(tree, L[site.index])
    # matcher.run(h, p)

def main():
    np.set_printoptions(linewidth=2000)
    np.set_printoptions(threshold=20000)
    # for j in range(1, 10000):
    #     print(j)
    #     copy_process_dev(100, 2000, j)
    # copy_process_dev(10, 4, 4)
    # for n in [10, 100, 1000, 10**4, 10**5]:
    #     copy_process_dev(n, 10000, 4)
    # for seed in range(1, 10000):
    #     print(seed)
    #     ancestral_sample_match_dev(500, seed)
    ancestral_sample_match_dev(5, 1)



if __name__ == "__main__":
    main()
