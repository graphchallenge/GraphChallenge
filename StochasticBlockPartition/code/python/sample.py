"""Helper functions for performing different kinds of sampling.
"""

from typing import List, Dict, Tuple
from copy import copy

import numpy as np


class Sample():
    """Stores the variables needed to create a subgraph.
    """

    # def __init__(self, sample_num: int, new_out_neighbors: List[np.ndarray], new_in_neighbors: List[np.ndarray],
    #     num_edges: int, true_block_assignment: np.ndarray, mapping: Dict[int,int],
    #     true_blocks_mapping: Dict[int,int]) -> None:
    def __init__(self, sample_idx: np.ndarray, old_out_neighbors: List[np.ndarray], old_in_neighbors: List[np.ndarray],
        old_true_block_assignment: np.ndarray) -> None:
        """Creates a new Sample object.
        """
        self.vertex_mapping = dict([(v, k) for k,v in enumerate(sample_idx)])
        self.out_neighbors = list()  # type: List[np.ndarray]
        self.in_neighbors = list()  # type: List[np.ndarray]
        self.num_edges = 0
        for index in sample_idx:
            out_neighbors = old_out_neighbors[index]
            out_mask = np.isin(out_neighbors[:,0], sample_idx, assume_unique=False)
            sampled_out_neighbors = out_neighbors[out_mask]
            for out_neighbor in sampled_out_neighbors:
                out_neighbor[0] = self.vertex_mapping[out_neighbor[0]]
            self.out_neighbors.append(sampled_out_neighbors)
            in_neighbors = old_in_neighbors[index]
            in_mask = np.isin(in_neighbors[:,0], sample_idx, assume_unique=False)
            sampled_in_neighbors = in_neighbors[in_mask]
            for in_neighbor in sampled_in_neighbors:
                in_neighbor[0] = self.vertex_mapping[in_neighbor[0]]
            self.in_neighbors.append(sampled_in_neighbors)
            self.num_edges += np.sum(out_mask) + np.sum(in_mask)
        true_block_assignment = old_true_block_assignment[sample_idx]
        true_blocks = list(set(true_block_assignment))
        self.true_blocks_mapping = dict([(v, k) for k,v in enumerate(true_blocks)])
        self.true_block_assignment = np.asarray([self.true_blocks_mapping[b] for b in true_block_assignment])
        self.sample_num = len(sample_idx)
    # End of __init__()

    @staticmethod
    def create_sample(num_vertices: int, old_out_neighbors: List[np.ndarray],
        old_in_neighbors: List[np.ndarray], old_true_block_assignment: np.ndarray,
        args: 'argparse.Namespace') -> 'Sample':
        """Performs sampling according to the sample type in args.
        """
        if args.sample_type == "uniform_random":
            return Sample.uniform_random_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                old_true_block_assignment, args)
        elif args.sample_type == "random_walk":
            return Sample.random_walk_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                             old_true_block_assignment, args)
        elif args.sample_type == "random_jump":
            return Sample.random_jump_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                old_true_block_assignment, args)
        elif args.sample_type == "degree_weighted":
            return Sample.degree_weighted_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                 old_true_block_assignment, args)
        elif args.sample_type == "random_node_neighbor":
            return Sample.random_node_neighbor_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                      old_true_block_assignment, args)
        elif args.sample_type == "forest_fire":
            return Sample.forest_fire_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                             old_true_block_assignment, args)
        elif args.sample_type == "expansion_snowball":
            return Sample.expansion_snowball_sample(num_vertices, old_out_neighbors, old_in_neighbors,
                                                    old_true_block_assignment, args)
        else:
            raise NotImplementedError("Sample type: {} is not implemented!".format(args.sample_type))
    # End of create_sample()

    @staticmethod
    def uniform_random_sample(num_vertices: int, old_out_neighbors: List[np.ndarray],
        old_in_neighbors: List[np.ndarray], old_true_block_assignment: np.ndarray,
        args: 'argparse.Namespace') -> 'Sample':
        """Uniform random sampling.
        """
        sample_num = int(num_vertices * (args.sample_size / 100))
        print("Sampling {} vertices from graph".format(sample_num))
        sample_idx = np.random.choice(num_vertices, sample_num, replace=False)
        return Sample(sample_idx, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of uniform_random_sampling()

    @staticmethod
    def random_walk_sample(num_vertices: int, old_out_neighbors: List[np.ndarray], old_in_neighbors: List[np.ndarray],
        old_true_block_assignment: np.ndarray, args: 'argparse.Namespace') -> 'Sample':
        """Random walk sampling.
        """
        sample_num = int(num_vertices * (args.sample_size / 100))
        print("Sampling {} vertices from graph".format(sample_num))
        sampled_marker = [False] * num_vertices
        index_set = list()  # type: List[int]
        num_tries = 0
        start = np.random.randint(sample_num)  # start with a random vertex
        vertex = start

        while len(index_set) < sample_num:
            num_tries += 1
            if not sampled_marker[vertex]:
                index_set.append(vertex)
                sampled_marker[vertex] = True
            if num_tries % sample_num == 0:  # If the number of tries is large, restart from new random vertex
                start = np.random.randint(sample_num)
                vertex = start
                num_tries = 0
            elif np.random.random() < 0.15:  # With a probability of 0.15, restart at original node
                vertex = start
            elif len(old_out_neighbors[vertex]) > 0:  # If the vertex has out neighbors, go to one of them
                vertex = np.random.choice(old_out_neighbors[vertex][:,0])
            else:  # Otherwise, restart from the original vertex
                if len(old_out_neighbors[start]) == 0:  # if original vertex has no out neighbors, change it
                    start = np.random.randint(sample_num)
                vertex = start
            
        sample_idx = np.asarray(index_set)
        return Sample(sample_idx, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of Random_walk_sampling()

    @staticmethod
    def random_jump_sample(num_vertices: int, old_out_neighbors: List[np.ndarray], old_in_neighbors: List[np.ndarray],
        old_true_block_assignment: np.ndarray, args: 'argparse.Namespace') -> 'Sample':
        """Random jump sampling.
        """
        sample_num = int(num_vertices * (args.sample_size / 100))
        print("Sampling {} vertices from graph".format(sample_num))
        sampled_marker = [False] * num_vertices
        index_set = list()  # type: List[int]
        num_tries = 0
        start = np.random.randint(sample_num)  # start with a random vertex
        vertex = start

        while len(index_set) < sample_num:
            num_tries += 1
            if not sampled_marker[vertex]:
                index_set.append(vertex)
                sampled_marker[vertex] = True
            # If the number of tries is large, or with a probability of 0.15, start from new random vertex
            if num_tries % sample_num == 0 or np.random.random() < 0.15:
                start = np.random.randint(sample_num)
                vertex = start
                num_tries = 0
            elif len(old_out_neighbors[vertex]) > 0:  # If the vertex has out neighbors, go to one of them
                vertex = np.random.choice(old_out_neighbors[vertex][:,0])
            else:  # Otherwise, restart from the original vertex
                if len(old_out_neighbors[start]) == 0:  # if original vertex has no out neighbors, change it
                    start = np.random.randint(sample_num)
                vertex = start
            
        sample_idx = np.asarray(index_set)
        return Sample(sample_idx, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of random_jump_sample()

    @staticmethod
    def degree_weighted_sample(num_vertices: int, old_out_neighbors: List[np.ndarray],
        old_in_neighbors: List[np.ndarray], old_true_block_assignment: np.ndarray,
        args: 'argparse.Namespace') -> 'Sample':
        """Degree-weighted sampling, where the probability of picking a vertex is proportional to its degree.
        """
        sample_num = int(num_vertices * (args.sample_size / 100))
        print("Sampling {} vertices from graph".format(sample_num))
        vertex_degrees = np.add([len(neighbors) for neighbors in old_out_neighbors], 
                                [len(neighbors) for neighbors in old_in_neighbors])
        sample_idx = np.random.choice(num_vertices, sample_num, replace=False, p=vertex_degrees/np.sum(vertex_degrees))
        return Sample(sample_idx, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of Random_walk_sampling()

    @staticmethod
    def random_node_neighbor_sample(num_vertices: int, old_out_neighbors: List[np.ndarray],
        old_in_neighbors: List[np.ndarray], old_true_block_assignment: np.ndarray,
        args: 'argparse.Namespace') -> 'Sample':
        """Random node neighbor sampling, where whenever a single node is sampled, all its out neighbors are sampled
        as well.
        """
        sample_num = int(num_vertices * (args.sample_size / 100))
        print("Sampling {} vertices from graph".format(sample_num))
        random_samples = np.random.choice(num_vertices, sample_num, replace=False)
        sampled_marker = [False] * num_vertices
        index_set = list()  # type: List[int]
        for vertex in random_samples:
            if not sampled_marker[vertex]:
                index_set.append(vertex)
                sampled_marker[vertex] = True
            for neighbor in old_out_neighbors[vertex]:
                if not sampled_marker[neighbor[0]]:
                    index_set.append(neighbor[0])
                    sampled_marker[neighbor[0]] = True
            if len(index_set) >= sample_num:
                break
        sample_idx = np.asarray(index_set[:sample_num])
        return Sample(sample_idx, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of random_node_neighbor_sample()

    @staticmethod
    def forest_fire_sample(num_vertices: int, old_out_neighbors: List[np.ndarray], old_in_neighbors: List[np.ndarray],
        old_true_block_assignment: np.ndarray, args: 'argparse.Namespace') -> 'Sample':
        """Forest-fire sampling with forward probability = 0.7.
        """
        sample_num = int(num_vertices * (args.sample_size / 100))
        print("Sampling {} vertices from graph".format(sample_num))
        sampled_marker = [False] * num_vertices
        burnt_marker = [False] * num_vertices
        current_fire_front = [np.random.randint(num_vertices)]
        next_fire_front = list()  # type: List[int]
        index_set = list()  # type: List[int]
        while len(index_set) < sample_num:
            for vertex in current_fire_front:
                # add vertex to index set
                if not sampled_marker[vertex]:
                    sampled_marker[vertex] = True
                    burnt_marker[vertex] = True
                    index_set.append(vertex)
                # select edges to burn
                num_to_choose = np.random.geometric(0.7)
                out_neighbors = old_out_neighbors[vertex]
                if len(out_neighbors) < 1:  # If there are no outgoing neighbors
                    continue
                if len(out_neighbors) <= num_to_choose:
                    num_to_choose = len(out_neighbors)
                mask = np.zeros(len(out_neighbors))
                indexes = np.random.choice(np.arange(len(out_neighbors)), num_to_choose, replace=False)
                mask[indexes] = 1
                for index, value in enumerate(mask):
                    neighbor = out_neighbors[index][0]
                    if value == 1:  # if chosen, add to next frontier
                        if not burnt_marker[neighbor]:
                            next_fire_front.append(neighbor)
                    burnt_marker[neighbor] = True  # mark all neighbors as visited
            if np.sum(burnt_marker) == num_vertices:  # all samples are burnt, restart
                burnt_marker = [False] * num_vertices
                current_fire_front = [np.random.randint(num_vertices)]
                next_fire_front = list()
                continue
            if len(next_fire_front) == 0:  # if fire is burnt-out
                current_fire_front = [np.random.randint(num_vertices)]
            else:
                current_fire_front = copy(next_fire_front)
                next_fire_front = list()
        sample_idx = np.asarray(index_set[:sample_num])
        return Sample(sample_idx, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of forest_fire_sample()

    @staticmethod
    def expansion_snowball_sample(num_vertices: int, old_out_neighbors: List[np.ndarray],
        old_in_neighbors: List[np.ndarray], old_true_block_assignment: np.ndarray,
        args: 'argparse.Namespace') -> 'Sample':
        """Expansion snowball sampling. At every iterations, picks a node adjacent to the current sample that
        contributes the most new neighbors.
        """
        sample_num = int(num_vertices * (args.sample_size / 100))
        print("Sampling {} vertices from graph".format(sample_num))
        start = np.random.randint(num_vertices)
        index_flag = [False] * num_vertices
        index_flag[start] = True
        index_set = [start]

        neighbors = list(old_out_neighbors[start][:,0])
        # Set up the initial contributions counts and flag currently neighboring vertices
        neighbors_flag = [False] * num_vertices
        contribution = [0] * num_vertices
        for neighbor in old_out_neighbors[start][:,0]:
            neighbors_flag[neighbor] = True
            new_neighbors = 0
            for _neighbor in old_out_neighbors[neighbor][:,0]:
                if not (index_flag[_neighbor] or neighbors_flag[_neighbor]): new_neighbors += 1
            contribution[neighbor] += new_neighbors
        while len(index_set) < sample_num:
            if len(neighbors) == 0:
                vertex = np.random.randint(num_vertices)
                if index_flag[vertex]:
                    continue
                index_set.append(vertex)
                for neighbor in old_out_neighbors[start][:,0]:
                    if not neighbors_flag[neighbor]:
                        Sample._add_neighbor(neighbor, contribution, index_flag, neighbors_flag,
                                             old_out_neighbors[neighbor][:,0], old_in_neighbors[neighbor][:,0],
                                             neighbors)
                continue
            vertex = np.argmax(contribution)
            index_set.append(vertex)
            index_flag[vertex] = True
            neighbors.remove(vertex)
            contribution[vertex] = 0
            for neighbor in old_in_neighbors[vertex][:,0]:
                if not neighbors_flag[neighbor]:
                    Sample._add_neighbor(neighbor, contribution, index_flag, neighbors_flag,
                                         old_out_neighbors[neighbor][:,0], old_in_neighbors[neighbor][:,0], neighbors)
        sample_idx = np.asarray(index_set)
        return Sample(sample_idx, old_out_neighbors, old_in_neighbors, old_true_block_assignment)
    # End of expansion_snowball_sample()

    @staticmethod
    def _add_neighbor(vertex: int, contribution: List[int], index_flag: List[bool], neighbor_flag: List[bool],
        out_neighbors: np.ndarray, in_neighbors: np.ndarray, neighbors: List[int]) -> Tuple[List[int], List[bool]]:
        """Updates the expansion contribution for neighbors of a single vertex.
        """
        neighbors.append(vertex)
        neighbor_flag[vertex] = True
        if contribution[vertex] == 0:
            Sample._calculate_contribution(vertex, contribution, index_flag, neighbor_flag, out_neighbors, in_neighbors)
        # # Compute contribution of this vertex
        # for out_neighbor in out_neighbors:
        #     if not (index_flag[out_neighbor] or neighbor_flag[out_neighbor]):
        #         contribution[vertex] += 1
        # # Decrease contribution of all neighbors with out links to this vertex
        # for in_neighbor in in_neighbors:
        #     if contribution[in_neighbor] > 0:
        #         contribution[in_neighbor] -= 1
        return contribution, neighbor_flag
    # End of _add_neighbor()

    @staticmethod
    def _calculate_contribution(vertex: int, contribution: List[int], index_flag: List[bool], neighbor_flag: List[bool],
        out_neighbors: np.ndarray, in_neighbors: np.ndarray):
        # Compute contribution of this vertex
        for out_neighbor in out_neighbors:
            if not (index_flag[out_neighbor] or neighbor_flag[out_neighbor]):
                contribution[vertex] += 1
        # Decrease contribution of all neighbors with out links to this vertex
        for in_neighbor in in_neighbors:
            if contribution[in_neighbor] > 0:
                contribution[in_neighbor] -= 1
# End of Sample()
