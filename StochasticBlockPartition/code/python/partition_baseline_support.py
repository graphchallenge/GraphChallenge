""" This library of supporting functions are written to perform graph partitioning according to the following reference

    References
    ----------
        .. [1] Peixoto, Tiago P. 'Entropy of stochastic blockmodel ensembles.'
               Physical Review E 85, no. 5 (2012): 056122.
        .. [2] Peixoto, Tiago P. 'Parsimonious module inference in large networks.'
               Physical review letters 110, no. 14 (2013): 148701.
        .. [3] Karrer, Brian, and Mark EJ Newman. 'Stochastic blockmodels and community structure in networks.'
               Physical Review E 83, no. 1 (2011): 016107.
"""

import pandas as pd
import numpy as np
from scipy import sparse as sparse
import scipy.misc as misc
from munkres import Munkres # for correctness evaluation
use_graph_tool_options = False # for visualiziing graph partitions (optional)
if use_graph_tool_options:
    import graph_tool.all as gt

from partition import Partition, PartitionTriplet
# from graph import Graph


class EdgeCountUpdates(object):
    """Holds the updates to the current interblock edge counts given a proposed block or node move.

    Since a block move affects only the rows and columns for the original and proposed blocks, only four rows and
    columns need to be stored for the edge count matrix updates.
    """

    def __init__(self, block_row: np.array, proposal_row: np.array, block_col: np.array,
                 proposal_col: np.array) -> None:
        """Creates a new EdgeCountUpdates object.

            Parameters
            ---------
            block_row : np.array [int]
                    the updates for the row of the current block
            proposal_row : np.array [int]
                    the updates for the row of the proposed block
            block_col : np.array [int]
                    the updates for the column of the current block
            proposal_col : np.array [int]
                    the updates for the column of the proposed block
        """
        self.block_row = block_row
        self.proposal_row = proposal_row
        self.block_col = block_col
        self.proposal_col = proposal_col
    # End of __init__()
# End of EdgeCountUpdates()


def propose_new_partition(r, neighbors_out, neighbors_in, b, partition: Partition, agg_move, use_sparse):
    """Propose a new block assignment for the current node or block

        Parameters
        ----------
        r : int
                    current block assignment for the node under consideration
        neighbors_out : ndarray (int) of two columns
                    out neighbors array where the first column is the node indices and the second column is the edge weight
        neighbors_in : ndarray (int) of two columns
                    in neighbors array where the first column is the node indices and the second column is the edge weight
        b : ndarray (int)
                    array of block assignment for each node
        partition : Partition
                    the current partitioning results
        agg_move : bool
                    whether the proposal is a block move
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        s : int
                    proposed block assignment for the node under consideration
        k_out : int
                    the out degree of the node
        k_in : int
                    the in degree of the node
        k : int
                    the total degree of the node

        Notes
        -----
        - d_u: degree of block u

        Randomly select a neighbor of the current node, and obtain its block assignment u. With probability \frac{B}{d_u + B}, randomly propose
        a block. Otherwise, randomly selects a neighbor to block u and propose its block assignment. For block (agglomerative) moves,
        avoid proposing the current block.
    """
    neighbors = np.concatenate((neighbors_out, neighbors_in))
    k_out = sum(neighbors_out[:,1])
    k_in = sum(neighbors_in[:,1])
    k = k_out + k_in
    if k==0: # this node has no neighbor, simply propose a block randomly
        s = np.random.randint(partition.num_blocks)
        return s, k_out, k_in, k
    rand_neighbor = np.random.choice(neighbors[:,0], p=neighbors[:,1]/float(k))
    u = b[rand_neighbor]
    # propose a new block randomly
    if np.random.uniform() <= partition.num_blocks/float(partition.block_degrees[u]+partition.num_blocks):  # chance inversely prop. to block_degree
        if agg_move:  # force proposal to be different from current block
            candidates = set(range(partition.num_blocks))
            candidates.discard(r)
            s = np.random.choice(list(candidates))
        else:
            s = np.random.randint(partition.num_blocks)
    else:  # propose by random draw from neighbors of block partition[rand_neighbor]
        if use_sparse:
            multinomial_prob = (partition.interblock_edge_count[u, :].toarray().transpose() + partition.interblock_edge_count[:, u].toarray()) / float(partition.block_degrees[u])
        else:
            multinomial_prob = (partition.interblock_edge_count[u, :].transpose() + partition.interblock_edge_count[:, u]) / float(partition.block_degrees[u])
        if agg_move:  # force proposal to be different from current block
            multinomial_prob[r] = 0
            if multinomial_prob.sum() == 0:  # the current block has no neighbors. randomly propose a different block
                candidates = set(range(partition.num_blocks))
                candidates.discard(r)
                s = np.random.choice(list(candidates))
                return s, k_out, k_in, k
            else:
                multinomial_prob = multinomial_prob / multinomial_prob.sum()
        candidates = multinomial_prob.nonzero()[0]
        s = candidates[np.flatnonzero(np.random.multinomial(1, multinomial_prob[candidates].ravel()))[0]]
    return s, k_out, k_in, k


def compute_new_rows_cols_interblock_edge_count_matrix(M, r, s, b_out, count_out, b_in, count_in, count_self,
                                                       agg_move, use_sparse):
    """Compute the two new rows and cols of the edge count matrix under the proposal for the current node or block

        Parameters
        ----------
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        r : int
                    current block assignment for the node under consideration
        s : int
                    proposed block assignment for the node under consideration
        b_out : ndarray (int)
                    blocks of the out neighbors
        count_out : ndarray (int)
                    edge counts to the out neighbor blocks
        b_in : ndarray (int)
                    blocks of the in neighbors
        count_in : ndarray (int)
                    edge counts to the in neighbor blocks
        count_self : int
                    edge counts to self
        agg_move : bool
                    whether the proposal is a block move
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        M_r_row : ndarray or sparse matrix (int)
                    the current block row of the new edge count matrix under proposal
        M_s_row : ndarray or sparse matrix (int)
                    the proposed block row of the new edge count matrix under proposal
        M_r_col : ndarray or sparse matrix (int)
                    the current block col of the new edge count matrix under proposal
        M_s_col : ndarray or sparse matrix (int)
                    the proposed block col of the new edge count matrix under proposal

        Notes
        -----
        The updates only involve changing the entries to and from the neighboring blocks
    """

    B = M.shape[0]
    if agg_move:  # the r row and column are simply empty after this merge move
        if use_sparse:
            M_r_row = sparse.lil_matrix(M[r, :].shape, dtype=int)
            M_r_col = sparse.lil_matrix(M[:, r].shape, dtype=int)
        else:
            M_r_row = np.zeros((1, B), dtype=int)
            M_r_col = np.zeros((B, 1), dtype=int)
    else:
        if use_sparse:
            M_r_row = M[r, :].copy()
            M_r_col = M[:, r].copy()
        else:
            M_r_row = M[r, :].copy().reshape(1, B)
            M_r_col = M[:, r].copy().reshape(B, 1)
        M_r_row[0, b_out] -= count_out
        M_r_row[0, r] -= np.sum(count_in[np.where(b_in == r)])
        M_r_row[0, s] += np.sum(count_in[np.where(b_in == r)])
        M_r_col[b_in, 0] -= count_in.reshape(M_r_col[b_in, 0].shape)
        M_r_col[r, 0] -= np.sum(count_out[np.where(b_out == r)])
        M_r_col[s, 0] += np.sum(count_out[np.where(b_out == r)])
    if use_sparse:
        M_s_row = M[s, :].copy()
        M_s_col = M[:, s].copy()
    else:
        M_s_row = M[s, :].copy().reshape(1, B)
        M_s_col = M[:, s].copy().reshape(B, 1)
    M_s_row[0, b_out] += count_out
    M_s_row[0, r] -= np.sum(count_in[np.where(b_in == s)])
    M_s_row[0, s] += np.sum(count_in[np.where(b_in == s)])
    M_s_row[0, r] -= count_self
    M_s_row[0, s] += count_self
    M_s_col[b_in, 0] += count_in.reshape(M_s_col[b_in, 0].shape)
    M_s_col[r, 0] -= np.sum(count_out[np.where(b_out == s)])
    M_s_col[s, 0] += np.sum(count_out[np.where(b_out == s)])
    M_s_col[r, 0] -= count_self
    M_s_col[s, 0] += count_self

    return EdgeCountUpdates(M_r_row, M_s_row, M_r_col, M_s_col)
# End of compute_new_rows_cols_interblock_edge_count_matrix()


def compute_new_block_degrees(r, s, partition: Partition, k_out, k_in, k):
    """Compute the new block degrees under the proposal for the current node or block

        Parameters
        ----------
        r : int
                    current block assignment for the node under consideration
        s : int
                    proposed block assignment for the node under consideration
        d_out : ndarray (int)
                    the current out degree of each block
        d_in : ndarray (int)
                    the current in degree of each block
        d : ndarray (int)
                    the current total degree of each block
        k_out : int
                    the out degree of the node
        k_in : int
                    the in degree of the node
        k : int
                    the total degree of the node

        Returns
        -------
        d_out_new : ndarray (int)
                    the new out degree of each block under proposal
        d_in_new : ndarray (int)
                    the new in degree of each block under proposal
        d_new : ndarray (int)
                    the new total degree of each block under proposal

        Notes
        -----
        The updates only involve changing the degrees of the current and proposed block"""
    new = []
    for old, degree in zip([partition.block_degrees_out, partition.block_degrees_in, partition.block_degrees], [k_out, k_in, k]):
        new_d = old.copy()
        new_d[r] -= degree
        new_d[s] += degree
        new.append(new_d)
    return new


def compute_Hastings_correction(b_out, count_out, b_in, count_in, s, M, M_r_row, M_r_col, B, d, d_new, use_sparse):
    """Compute the Hastings correction for the proposed block from the current block

        Parameters
        ----------
        b_out : ndarray (int)
                    blocks of the out neighbors
        count_out : ndarray (int)
                    edge counts to the out neighbor blocks
        b_in : ndarray (int)
                    blocks of the in neighbors
        count_in : ndarray (int)
                    edge counts to the in neighbor blocks
        s : int
                    proposed block assignment for the node under consideration
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        M_r_row : ndarray or sparse matrix (int)
                    the current block row of the new edge count matrix under proposal
        M_r_col : ndarray or sparse matrix (int)
                    the current block col of the new edge count matrix under proposal
        B : int
                    total number of blocks
        d : ndarray (int)
                    total number of edges to and from each block
        d_new : ndarray (int)
                    new block degrees under the proposal
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        Hastings_correction : float
                    term that corrects for the transition asymmetry between the current block and the proposed block

        Notes
        -----
        - p_{i, s \rightarrow r} : for node i, probability of proposing block r if its current block is s
        - p_{i, r \rightarrow s} : for node i, probability of proposing block s if its current block is r
        - r : current block for node i
        - s : proposed block for node i
        - M^-: current edge count matrix between the blocks
        - M^+: new edge count matrix under the proposal
        - d^-_t: current degree of block t
        - d^+_t: new degree of block t under the proposal
        - \mathbf{b}_{\mathcal{N}_i}: the neighboring blocks to node i
        - k_i: the degree of node i
        - k_{i,t} : the degree of node i to block t (i.e. number of edges to and from block t)
        - B : the number of blocks

        The Hastings correction is:

        \frac{p_{i, s \rightarrow r}}{p_{i, r \rightarrow s}}

        where

        p_{i, r \rightarrow s} = \sum_{t \in \{\mathbf{b}_{\mathcal{N}_i}^-\}} \left[ {\frac{k_{i,t}}{k_i} \frac{M_{ts}^- + M_{st}^- + 1}{d^-_t+B}}\right]

        p_{i, s \rightarrow r} = \sum_{t \in \{\mathbf{b}_{\mathcal{N}_i}^-\}} \left[ {\frac{k_{i,t}}{k_i} \frac{M_{tr}^+ + M_{rt}^+ +1}{d_t^++B}}\right]

        summed over all the neighboring blocks t"""

    t, idx = np.unique(np.append(b_out, b_in), return_inverse=True)  # find all the neighboring blocks
    count = np.bincount(idx, weights=np.append(count_out, count_in)).astype(int)  # count edges to neighboring blocks
    if use_sparse:
        M_t_s = M[t, s].toarray().ravel()
        M_s_t = M[s, t].toarray().ravel()
        M_r_row = M_r_row[0, t].toarray().ravel()
        M_r_col = M_r_col[t, 0].toarray().ravel()
    else:
        M_t_s = M[t, s].ravel()
        M_s_t = M[s, t].ravel()
        M_r_row = M_r_row[0, t].ravel()
        M_r_col = M_r_col[t, 0].ravel()
        
    p_forward = np.sum(count*(M_t_s + M_s_t + 1) / (d[t] + float(B)))
    p_backward = np.sum(count*(M_r_row + M_r_col + 1) / (d_new[t] + float(B)))
    return p_backward / p_forward


def compute_delta_entropy(r, s, partition: Partition, edge_count_updates: EdgeCountUpdates, d_out_new, d_in_new, use_sparse):
    """Compute change in entropy under the proposal. Reduced entropy means the proposed block is better than the current block.

        Parameters
        ----------
        r : int
                    current block assignment for the node under consideration
        s : int
                    proposed block assignment for the node under consideration
        partition : Partition
                    the current partitioning results
        edge_count_updates : EdgeCountUpdates
                    the updates to the current partition's edge count
        d_out_new : ndarray (int)
                    the new out degree of each block under proposal
        d_in_new : ndarray (int)
                    the new in degree of each block under proposal
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        delta_entropy : float
                    entropy under the proposal minus the current entropy

        Notes
        -----
        - M^-: current edge count matrix between the blocks
        - M^+: new edge count matrix under the proposal
        - d^-_{t, in}: current in degree of block t
        - d^-_{t, out}: current out degree of block t
        - d^+_{t, in}: new in degree of block t under the proposal
        - d^+_{t, out}: new out degree of block t under the proposal
        
        The difference in entropy is computed as:
        
        \dot{S} = \sum_{t_1, t_2} {\left[ -M_{t_1 t_2}^+ \text{ln}\left(\frac{M_{t_1 t_2}^+}{d_{t_1, out}^+ d_{t_2, in}^+}\right) + M_{t_1 t_2}^- \text{ln}\left(\frac{M_{t_1 t_2}^-}{d_{t_1, out}^- d_{t_2, in}^-}\right)\right]}
        
        where the sum runs over all entries $(t_1, t_2)$ in rows and cols $r$ and $s$ of the edge count matrix
    """
    if use_sparse: # computation in the sparse matrix is slow so convert to numpy arrays since operations are on only two rows and cols
        M_r_row = edge_count_updates.block_row.toarray()
        M_s_row = edge_count_updates.proposal_row.toarray()
        M_r_col = edge_count_updates.block_col.toarray()
        M_s_col = edge_count_updates.proposal_col.toarray()
        M_r_t1 = partition.interblock_edge_count[r, :].toarray()
        M_s_t1 = partition.interblock_edge_count[s, :].toarray()
        M_t2_r = partition.interblock_edge_count[:, r].toarray()
        M_t2_s = partition.interblock_edge_count[:, s].toarray()
    else:
        M_r_row = edge_count_updates.block_row
        M_s_row = edge_count_updates.proposal_row
        M_r_col = edge_count_updates.block_col
        M_s_col = edge_count_updates.proposal_col
        M_r_t1 = partition.interblock_edge_count[r, :]
        M_s_t1 = partition.interblock_edge_count[s, :]
        M_t2_r = partition.interblock_edge_count[:, r]
        M_t2_s = partition.interblock_edge_count[:, s]

    # remove r and s from the cols to avoid double counting
    idx = list(range(len(d_in_new)))
    del idx[max(r, s)]
    del idx[min(r, s)]
    M_r_col = M_r_col[idx]
    M_s_col = M_s_col[idx]
    M_t2_r = M_t2_r[idx]
    M_t2_s = M_t2_s[idx]
    d_out_new_ = d_out_new[idx]
    d_out_ = partition.block_degrees_out[idx]

    # only keep non-zero entries to avoid unnecessary computation
    d_in_new_r_row = d_in_new[M_r_row.ravel().nonzero()]
    d_in_new_s_row = d_in_new[M_s_row.ravel().nonzero()]
    M_r_row = M_r_row[M_r_row.nonzero()]
    M_s_row = M_s_row[M_s_row.nonzero()]
    d_out_new_r_col = d_out_new_[M_r_col.ravel().nonzero()]
    d_out_new_s_col = d_out_new_[M_s_col.ravel().nonzero()]
    M_r_col = M_r_col[M_r_col.nonzero()]
    M_s_col = M_s_col[M_s_col.nonzero()]
    d_in_r_t1 = partition.block_degrees_in[M_r_t1.ravel().nonzero()]
    d_in_s_t1 = partition.block_degrees_in[M_s_t1.ravel().nonzero()]
    M_r_t1= M_r_t1[M_r_t1.nonzero()]
    M_s_t1 = M_s_t1[M_s_t1.nonzero()]
    d_out_r_col = d_out_[M_t2_r.ravel().nonzero()]
    d_out_s_col = d_out_[M_t2_s.ravel().nonzero()]
    M_t2_r = M_t2_r[M_t2_r.nonzero()]
    M_t2_s = M_t2_s[M_t2_s.nonzero()]

    # sum over the two changed rows and cols
    delta_entropy = 0
    delta_entropy -= np.sum(M_r_row * np.log(M_r_row.astype(float) / d_in_new_r_row / d_out_new[r]))
    delta_entropy -= np.sum(M_s_row * np.log(M_s_row.astype(float) / d_in_new_s_row / d_out_new[s]))
    delta_entropy -= np.sum(M_r_col * np.log(M_r_col.astype(float) / d_out_new_r_col / d_in_new[r]))
    delta_entropy -= np.sum(M_s_col * np.log(M_s_col.astype(float) / d_out_new_s_col / d_in_new[s]))
    delta_entropy += np.sum(M_r_t1 * np.log(M_r_t1.astype(float) / d_in_r_t1 / partition.block_degrees_out[r]))
    delta_entropy += np.sum(M_s_t1 * np.log(M_s_t1.astype(float) / d_in_s_t1 / partition.block_degrees_out[s]))
    delta_entropy += np.sum(M_t2_r * np.log(M_t2_r.astype(float) / d_out_r_col / partition.block_degrees_in[r]))
    delta_entropy += np.sum(M_t2_s * np.log(M_t2_s.astype(float) / d_out_s_col / partition.block_degrees_in[s]))
    return delta_entropy


def carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block, partition: Partition) -> Partition:
    """Execute the best merge (agglomerative) moves to reduce a set number of blocks

        Parameters
        ----------
        delta_entropy_for_each_block : ndarray (float)
                    the delta entropy for merging each block
        best_merge_for_each_block : ndarray (int)
                    the best block to merge with for each block
        partition : Partition
                    the current partitioning results

        Returns
        -------
        partition : Partition
                    the modified partition, with the merges carried out
    """
    bestMerges = delta_entropy_for_each_block.argsort()
    block_map = np.arange(partition.num_blocks)
    num_merge = 0
    counter = 0
    while num_merge < partition.num_blocks_to_merge:
        mergeFrom = bestMerges[counter]
        mergeTo = block_map[best_merge_for_each_block[bestMerges[counter]]]
        counter += 1
        if mergeTo != mergeFrom:
            block_map[np.where(block_map == mergeFrom)] = mergeTo
            partition.block_assignment[np.where(partition.block_assignment == mergeFrom)] = mergeTo
            num_merge += 1
    remaining_blocks = np.unique(partition.block_assignment)
    mapping = -np.ones(partition.num_blocks, dtype=int)
    mapping[remaining_blocks] = np.arange(len(remaining_blocks))
    partition.block_assignment = mapping[partition.block_assignment]
    partition.num_blocks -= partition.num_blocks_to_merge
    return partition
# End of carry_out_best_merges()


def update_partition(partition: Partition, ni, r, s, edge_count_updates: EdgeCountUpdates, d_out_new, d_in_new, d_new,
    use_sparse: bool) -> Partition:
    """Move the current node to the proposed block and update the edge counts

        Parameters
        ----------
        partition : Partition
                    the current partitioning results
        ni : int
                    current node index
        r : int
                    current block assignment for the node under consideration
        s : int
                    proposed block assignment for the node under consideration
        M_r_row : ndarray or sparse matrix (int)
                    the current block row of the new edge count matrix under proposal
        M_s_row : ndarray or sparse matrix (int)
                    the proposed block row of the new edge count matrix under proposal
        M_r_col : ndarray or sparse matrix (int)
                    the current block col of the new edge count matrix under proposal
        M_s_col : ndarray or sparse matrix (int)
                    the proposed block col of the new edge count matrix under proposal
        d_out_new : ndarray (int)
                    the new out degree of each block under proposal
        d_in_new : ndarray (int)
                    the new in degree of each block under proposal
        d_new : ndarray (int)
                    the new total degree of each block under proposal
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        partition : Partition
                    the updated partitioning results
    """
    partition.block_assignment[ni] = s
    partition.interblock_edge_count[r, :] = edge_count_updates.block_row
    partition.interblock_edge_count[s, :] = edge_count_updates.proposal_row
    if use_sparse:
        partition.interblock_edge_count[:, r] = edge_count_updates.block_col
        partition.interblock_edge_count[:, s] = edge_count_updates.proposal_col
    else:
        partition.interblock_edge_count[:, r] = edge_count_updates.block_col.reshape(partition.interblock_edge_count[:, r].shape)
        partition.interblock_edge_count[:, s] = edge_count_updates.proposal_col.reshape(partition.interblock_edge_count[:, s].shape)
    partition.block_degrees_out = d_out_new
    partition.block_degrees_in = d_in_new
    partition.block_degrees = d_new
    return partition
# End of update_partition()


def compute_overall_entropy(partition: Partition, N, E, use_sparse) -> float:
    """Compute the overall entropy, including the model entropy as well as the data entropy, on the current partition.
       The best partition with an optimal number of blocks will minimize this entropy.

        Parameters
        ----------
        partition : Partition
                    the current partitioning results
        N : int
                    number of nodes in the graph
        E : int
                    number of edges in the graph
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        S : float
                    the overall entropy of the current partition

        Notes
        -----
        - M: current edge count matrix
        - d_{t, out}: current out degree of block t
        - d_{t, in}: current in degree of block t
        - B: number of blocks
        - C: some constant invariant to the partition
        
        The overall entropy of the partition is computed as:
        
        S = E\;h\left(\frac{B^2}{E}\right) + N \ln(B) - \sum_{t_1, t_2} {M_{t_1 t_2} \ln\left(\frac{M_{t_1 t_2}}{d_{t_1, out} d_{t_2, in}}\right)} + C
        
        where the function h(x)=(1+x)\ln(1+x) - x\ln(x) and the sum runs over all entries (t_1, t_2) in the edge count matrix
    """
    nonzeros = partition.interblock_edge_count.nonzero()  # all non-zero entries
    edge_count_entries = partition.interblock_edge_count[nonzeros[0], nonzeros[1]]
    if use_sparse:
        edge_count_entries = edge_count_entries.toarray()

    entries = edge_count_entries * np.log(edge_count_entries / (partition.block_degrees_out[nonzeros[0]] * partition.block_degrees_in[nonzeros[1]]).astype(float))
    data_S = -np.sum(entries)
    model_S_term = partition.num_blocks**2 / float(E)
    model_S = E * (1 + model_S_term) * np.log(1 + model_S_term) - model_S_term * np.log(model_S_term) + N*np.log(partition.num_blocks)
    S = model_S + data_S
    return S


def prepare_for_partition_on_next_num_blocks(partition: Partition, partition_triplet: PartitionTriplet, B_rate):
    """Checks to see whether the current partition has the optimal number of blocks. If not, the next number of blocks
       to try is determined and the intermediate variables prepared.

        Parameters
        ----------
        partition : Partition
                the most recent partitioning results
        partition_triplet : Partition
                the triplet of the three best partitioning results for Fibonacci search

        Returns:
        ----------
        partition : Partition
                the partitioning results to use for the next iteration of the algorithm
        partition_triplet : Partition
                the updated triplet of the three best partitioning results for Fibonacci search

        Old Parameters
        ----------
        b : ndarray (int)
                    current array of block assignment for each node
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        d : ndarray (int)
                    the current total degree of each block
        d_out : ndarray (int)
                    the current out degree of each block
        d_in : ndarray (int)
                    the current in degree of each block
        B : int
                    the number of blocks in the current partition
        old_b : list of length 3
                    holds the best three partitions so far
        old_M : list of length 3
                    holds the edge count matrices for the best three partitions so far
        old_d : list of length 3
                    holds the block degrees for the best three partitions so far
        old_d_out : list of length 3
                    holds the out block degrees for the best three partitions so far
        old_d_in : list of length 3
                    holds the in block degrees for the best three partitions so far
        old_S : list of length 3
                    holds the overall entropy for the best three partitions so far
        old_B : list of length 3
                    holds the number of blocks for the best three partitions so far
        B_rate : float
                    the ratio on the number of blocks to reduce before the golden ratio bracket is established

        Old Returns
        -------
        b : ndarray (int)
                starting array of block assignment on each node for the next number of blocks to try
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    starting edge count matrix for the next number of blocks to try
        d : ndarray (int)
                    the starting total degree of each block for the next number of blocks to try
        d_out : ndarray (int)
                    the starting out degree of each block for the next number of blocks to try
        d_in : ndarray (int)
                    the starting in degree of each block for the next number of blocks to try
        B : int
                    the starting number of blocks before the next block merge
        B_to_merge : int
                    number of blocks to merge next
        old_b : list of length 3
                    holds the best three partitions including the current partition
        old_M : list of length 3
                    holds the edge count matrices for the best three partitions including the current partition
        old_d : list of length 3
                    holds the block degrees for the best three partitions including the current partition
        old_d_out : list of length 3
                    holds the out block degrees for the best three partitions including the current partition
        old_d_in : list of length 3
                    holds the in block degrees for the best three partitions including the current partition
        old_S : list of length 3
                    holds the overall entropy for the best three partitions including the current partition
        old_B : list of length 3
                    holds the number of blocks for the best three partitions including the current partition
        optimal_B_found : bool
                    flag for whether the optimal block has been found

        Notes
        -----
        The holders for the best three partitions so far and their statistics will be stored in the order of the number
        of blocks, starting from the highest to the lowest. The middle entry is always the best so far. The number of
        blocks is reduced by a fixed rate until the golden ratio bracket (three best partitions with the middle one
        being the best) is established. Once the golden ratio bracket is established, perform golden ratio search until
        the bracket is narrowed to consecutive number of blocks where the middle one is identified as the optimal
        number of blocks.
    """

    optimal_B_found = False
    B_to_merge = 0

    partition_triplet.update(partition)

    # find the next number of blocks to try using golden ratio bisection
    if partition_triplet.overall_entropy[2] == np.Inf:  # if the three points in the golden ratio bracket has not yet been established
        partition.num_blocks_to_merge = int(partition.num_blocks*B_rate)
        if (partition.num_blocks_to_merge == 0): # not enough number of blocks to merge so done
            optimal_B_found = True
        partition.block_assignment = partition_triplet.block_assignment[1].copy()
        partition.interblock_edge_count = partition_triplet.interblock_edge_count[1].copy()
        partition.block_degrees = partition_triplet.block_degrees[1].copy()
        partition.block_degrees_out = partition_triplet.block_degrees_out[1].copy()
        partition.block_degrees_in = partition_triplet.block_degrees_in[1].copy()
    else:  # golden ratio search bracket established
        if partition_triplet.num_blocks[0] - partition_triplet.num_blocks[2] == 2:  # we have found the partition with the optimal number of blocks
            optimal_B_found = True
            partition.num_blocks = partition_triplet.num_blocks[1]
            partition.block_assignment = partition_triplet.block_assignment[1]
        else:  # not done yet, find the next number of block to try according to the golden ratio search
            if (partition_triplet.num_blocks[0]-partition_triplet.num_blocks[1]) >= (partition_triplet.num_blocks[1]-partition_triplet.num_blocks[2]):  # the higher segment in the bracket is bigger
                index = 0
            else:  # the lower segment in the bracket is bigger
                index = 1
            next_B_to_try = partition_triplet.num_blocks[index + 1] + np.round((partition_triplet.num_blocks[index] - partition_triplet.num_blocks[index + 1]) * 0.618).astype(int)
            partition.num_blocks_to_merge = partition_triplet.num_blocks[index] - next_B_to_try
            partition.num_blocks = partition_triplet.num_blocks[index]
            partition.block_assignment = partition_triplet.block_assignment[index].copy()
            partition.interblock_edge_count = partition_triplet.interblock_edge_count[index].copy()
            partition.block_degrees = partition_triplet.block_degrees[index].copy()
            partition.block_degrees_out = partition_triplet.block_degrees_out[index].copy()
            partition.block_degrees_in = partition_triplet.block_degrees_in[index].copy()

    partition_triplet.optimal_num_blocks_found = optimal_B_found
    return partition, partition_triplet


def plot_graph_with_partition(out_neighbors, b, graph_object=None, pos=None):
    """Plot the graph with force directed layout and color/shape each node according to its block assignment

        Parameters
        ----------
        out_neighbors : list of ndarray; list length is N, the number of nodes
                    each element of the list is a ndarray of out neighbors, where the first column is the node indices
                    and the second column the corresponding edge weights
        b : ndarray (int)
                    array of block assignment for each node
        graph_object : graph tool object, optional
                    if a graph object already exists, use it to plot the graph
        pos : ndarray (float) shape = (#nodes, 2), optional
                    if node positions are given, plot the graph using them

        Returns
        -------
        graph_object : graph tool object
                    the graph tool object containing the graph and the node position info"""

    if len(out_neighbors) <= 5000:
        if graph_object is None:
            graph_object = gt.Graph()
            edge_list = [(i, j) for i in range(len(out_neighbors)) if len(out_neighbors[i]) > 0 for j in
                         out_neighbors[i][:, 0]]
            graph_object.add_edge_list(edge_list)
            if pos is None:
                graph_object.vp['pos'] = gt.sfdp_layout(graph_object)
            else:
                graph_object.vp['pos'] = graph_object.new_vertex_property("vector<float>")
                for v in graph_object.vertices():
                    graph_object.vp['pos'][v] = pos[graph_object.vertex_index[v], :]
        block_membership = graph_object.new_vertex_property("int")
        vertex_shape = graph_object.new_vertex_property("int")
        block_membership.a = b[0:len(out_neighbors)]
        vertex_shape.a = np.mod(block_membership.a, 10)
        gt.graph_draw(graph_object, inline=True, output_size=(400, 400), pos=graph_object.vp['pos'],
                      vertex_shape=vertex_shape,
                      vertex_fill_color=block_membership, edge_pen_width=0.1, edge_marker_size=1, vertex_size=7)
    else:
        print('That\'s a big graph!')
    return graph_object


def evaluate_partition(true_b, alg_b):
    """Evaluate the output partition against the truth partition and report the correctness metrics.
       Compare the partitions using only the nodes that have known truth block assignment.

        Parameters
        ----------
        true_b : ndarray (int)
                array of truth block assignment for each node. If the truth block is not known for a node, -1 is used
                to indicate unknown blocks.
        alg_b : ndarray (int)
                array of output block assignment for each node. The length of this array corresponds to the number of
                nodes observed and processed so far."""

    blocks_b1 = true_b
    blocks_b1_set = set(true_b)
    blocks_b1_set.discard(-1)  # -1 is the label for 'unknown'
    B_b1 = len(blocks_b1_set)

    blocks_b2 = alg_b
    B_b2 = max(blocks_b2) + 1

    print('\nPartition Correctness Evaluation\n')
    print('Number of nodes: {}'.format(len(alg_b)))
    print('Number of partitions in truth partition: {}'.format(B_b1))
    print('Number of partitions in alg. partition: {}'.format(B_b2))

    # populate the confusion matrix between the two partitions
    contingency_table = np.zeros((B_b1, B_b2))
    for i in range(len(alg_b)):  # evaluation based on nodes observed so far
        if true_b[i] != -1:  # do not include nodes without truth in the evaluation
            contingency_table[blocks_b1[i], blocks_b2[i]] += 1
    N = contingency_table.sum()

    # associate the labels between two partitions using linear assignment
    assignment = Munkres()  # use the Hungarian algorithm / Kuhn-Munkres algorithm
    if B_b1 > B_b2:  # transpose matrix for linear assignment (this implementation assumes #col >= #row)
        contingency_table = contingency_table.transpose()
    indexes = assignment.compute(-contingency_table)
    total = 0
    contingency_table_before_assignment = np.array(contingency_table)
    for row, column in indexes:
        contingency_table[:, row] = contingency_table_before_assignment[:, column]
        total += contingency_table[row, row]
    # fill in the un-associated columns
    unassociated_col = set(range(contingency_table.shape[1])) - set(np.array(indexes)[:, 1])
    counter = 0;
    for column in unassociated_col:
        contingency_table[:, contingency_table.shape[0] + counter] = contingency_table_before_assignment[:, column]
        counter += 1
    if B_b1 > B_b2:  # transpose back
        contingency_table = contingency_table.transpose()
    print('Contingency Table: \n{}'.format(contingency_table))
    joint_prob = contingency_table / sum(
        sum(contingency_table))  # joint probability of the two partitions is just the normalized contingency table
    accuracy = sum(joint_prob.diagonal())
    print('Accuracy (with optimal partition matching): {}'.format(accuracy))
    print('\n')

    # Compute pair-counting-based metrics
    def nchoose2(a):
        return misc.comb(a, 2)

    num_pairs = nchoose2(N)
    colsum = np.sum(contingency_table, axis=0)
    rowsum = np.sum(contingency_table, axis=1)
    # compute counts of agreements and disagreement (4 types) and the regular rand index
    sum_table_squared = sum(sum(contingency_table ** 2))
    sum_colsum_squared = sum(colsum ** 2)
    sum_rowsum_squared = sum(rowsum ** 2)
    count_in_each_b1 = np.sum(contingency_table, axis=1)
    count_in_each_b2 = np.sum(contingency_table, axis=0)
    num_same_in_b1 = sum(count_in_each_b1 * (count_in_each_b1 - 1)) / 2
    num_same_in_b2 = sum(count_in_each_b2 * (count_in_each_b2 - 1)) / 2
    num_agreement_same = 0.5 * sum(sum(contingency_table * (contingency_table - 1)));
    num_agreement_diff = 0.5 * (N ** 2 + sum_table_squared - sum_colsum_squared - sum_rowsum_squared);
    num_agreement = num_agreement_same + num_agreement_diff
    rand_index = num_agreement / num_pairs

    vectorized_nchoose2 = np.vectorize(nchoose2)
    sum_table_choose_2 = sum(sum(vectorized_nchoose2(contingency_table)))
    sum_colsum_choose_2 = sum(vectorized_nchoose2(colsum))
    sum_rowsum_choose_2 = sum(vectorized_nchoose2(rowsum))
    adjusted_rand_index = (sum_table_choose_2 - sum_rowsum_choose_2 * sum_colsum_choose_2 / num_pairs) / (
        0.5 * (sum_rowsum_choose_2 + sum_colsum_choose_2) - sum_rowsum_choose_2 * sum_colsum_choose_2 / num_pairs)
    print('Rand Index: {}'.format(rand_index))
    print('Adjusted Rand Index: {}'.format(adjusted_rand_index))
    print('Pairwise Recall: {}'.format(num_agreement_same / (num_same_in_b1)))
    print('Pairwise Precision: {}'.format(num_agreement_same / (num_same_in_b2)))
    print('\n')

    # compute the information theoretic metrics
    marginal_prob_b2 = np.sum(joint_prob, 0)
    marginal_prob_b1 = np.sum(joint_prob, 1)
    idx1 = np.nonzero(marginal_prob_b1)
    idx2 = np.nonzero(marginal_prob_b2)
    conditional_prob_b2_b1 = np.zeros(joint_prob.shape)
    conditional_prob_b1_b2 = np.zeros(joint_prob.shape)
    conditional_prob_b2_b1[idx1, :] = joint_prob[idx1, :] / marginal_prob_b1[idx1, None]
    conditional_prob_b1_b2[:, idx2] = joint_prob[:, idx2] / marginal_prob_b2[None, idx2]
    # compute entropy of the non-partition2 and the partition2 version
    H_b2 = -np.sum(marginal_prob_b2[idx2] * np.log(marginal_prob_b2[idx2]))
    H_b1 = -np.sum(marginal_prob_b1[idx1] * np.log(marginal_prob_b1[idx1]))

    # compute the conditional entropies
    idx = np.nonzero(joint_prob)
    H_b2_b1 = -np.sum(np.sum(joint_prob[idx] * np.log(conditional_prob_b2_b1[idx])))
    H_b1_b2 = -np.sum(np.sum(joint_prob[idx] * np.log(conditional_prob_b1_b2[idx])))
    # compute the mutual information (symmetric)
    marginal_prod = np.dot(marginal_prob_b1[:, None], np.transpose(marginal_prob_b2[:, None]))
    MI_b1_b2 = np.sum(np.sum(joint_prob[idx] * np.log(joint_prob[idx] / marginal_prod[idx])))

    if H_b1 > 0:
        fraction_missed_info = H_b1_b2 / H_b1
    else:
        fraction_missed_info = 0
    if H_b2 > 0:
        fraction_err_info = H_b2_b1 / H_b2
    else:
        fraction_err_info = 0
    print('Entropy of truth partition: {}'.format(abs(H_b1)))
    print('Entropy of alg. partition: {}'.format(abs(H_b2)))
    print('Conditional entropy of truth partition given alg. partition: {}'.format(abs(H_b1_b2)))
    print('Conditional entropy of alg. partition given truth partition: {}'.format(abs(H_b2_b1)))
    print('Mututal informationion between truth partition and alg. partition: {}'.format(abs(MI_b1_b2)))
    print('Fraction of missed information: {}'.format(abs(fraction_missed_info)))
    print('Fraction of erroneous information: {}'.format(abs(fraction_err_info)))
