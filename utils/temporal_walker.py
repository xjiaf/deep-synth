from bisect import bisect_left, bisect_right
import torch
import torch.nn.functional as F
from datasets import TemporalGraph


class TemporalWalker:
    def __init__(self, is_ascending=True, sample_based='weight', sample_method='softmax'):
        """
        Class for simulating random walks on temporal graphs.
        Attributes:
            is_ascending (bool): If true, the walker will generate walk in time ascending order.
            sample_based (str): Strategy to determine neighbor selection. ('weight' or 'time')
            sample_method (str): Method to sample a neighbor based on probabilities. ('uniform', 'linear', 'softmax')
        """
        self.is_ascending = is_ascending
        self.sample_based = sample_based
        self.sample_method = sample_method

    def walk(self, graph: TemporalGraph, start_node_list: list,
             walk_length: int = 5, start_time='begin') -> torch.Tensor:
        """
        Simulate random walks for all start nodes.
        Args:
            graph (TemporalGraph): Temporal graph on which the walks will be performed.
            start_nodes (List): Nodes from which walks will start.
            walk_length (int): Number of steps each walk will take.
            start_time (dict or float/int): Mapping start nodes to their start timestamp.
                        Further processed in get_start_time function

        Returns:
            tuple: Tuple containing three tensors - all_walks, all_walk_edge_weights, all_walk_edge_times.
        """
        self.graph = graph
        self.start_node_list = list(start_node_list)
        self.walk_length = walk_length
        self.start_time = self.get_start_time(start_time=start_time)
        all_walks, all_walk_edge_weights, all_walk_edge_times = [], [], []

        for start_node in self.start_node_list:
            walk, walk_edge_weights, walk_edge_times = self._single_walk(start_node)
            all_walks.append(walk)
            all_walk_edge_weights.append(walk_edge_weights)
            all_walk_edge_times.append(walk_edge_times)

        return torch.stack(all_walks), torch.stack(all_walk_edge_weights), torch.stack(all_walk_edge_times)

    def _single_walk(self, start_node):
        """
        Simulate a single random walk for a given start node.

        Args:
            start_node (int): Node from which the walk will start.

        Returns:
            tuple: Tuple containing walk, walk_edge_weights, and walk_edge_times for the given start node.
        """
        walk = []
        walk_edge_weights = []
        walk_edge_times = []
        current_node = start_node
        current_time = self.start_time[start_node]

        steps_taken = 0
        while steps_taken < self.walk_length:
            # Get neighbors of current_node for timestamps greater than current_time
            neighbors_data = self.graph.neighbor_sequence[current_node]
            if self.is_ascending:
                idx = bisect_right([t for _, t, _ in neighbors_data], current_time)  # current_time excluded
                valid_neighbors = neighbors_data[idx:]
            else:
                idx = bisect_left([t for _, t, _ in neighbors_data], current_time)
                valid_neighbors = neighbors_data[:idx]
            # If no valid neighbors found, restart the walk
            if not valid_neighbors:
                current_node = start_node
                current_time = self.start_time[start_node]
                continue

            # Compute probabilities based on the chosen strategy
            probabilities = compute_probabilities(valid_neighbors, current_time, sample_based=self.sample_based)

            # Sample a neighbor based on the probabilities
            chosen_index = sample_neighbor(probabilities, method=self.sample_method)

            # Update current_node, current_time, and the corresponding edge weight
            current_node, current_time, edge_weight = valid_neighbors[chosen_index]

            walk.append(current_node)
            walk_edge_times.append(current_time)
            walk_edge_weights.append(edge_weight)

            steps_taken += 1

        return walk, walk_edge_times, walk_edge_weights

    def get_start_time(self, start_time):
        if start_time == "begin":
            self.is_ascending = True
            return {node: (self.graph.neighbor_sequence[node][0][1] - 1) for node in self.start_node_list}
        elif start_time == "end":
            self.is_ascending = False
            return {node: (self.graph.neighbor_sequence[node][-1][1] + 1) for node in self.start_node_list}
        elif isinstance(start_time, int) or isinstance(start_time, float):
            return {node: start_time for node in self.start_node_list}
        elif isinstance(start_time, dict):
            return start_time


def compute_probabilities(valid_neighbors: list, current_time: float, sample_based='weight') -> torch.Tensor:
    """
    Compute probabilities for selecting neighbors.

    Args:
        valid_neighbors (list): List of valid neighbors.
        current_time (float): Current timestamp.
        sample_based (str, optional): Strategy for determining neighbor selection. Defaults to 'weight'.

    Returns:
        torch.Tensor: Probabilities for selecting each neighbor.
    """
    if sample_based == 'weight':
        return torch.tensor([weight for _, _, weight in valid_neighbors])
    elif sample_based == 'time':
        return torch.tensor([abs(t - current_time) for _, t, _ in valid_neighbors])
    else:
        raise ValueError(f"Sample based on '{sample_based}' not implemented")


def sample_neighbor(probabilities: torch.Tensor, method: str = 'softmax') -> int:
    """
    Sample a neighbor based on given probabilities and method.

    Args:
        probabilities (torch.Tensor): Probabilities for selecting each neighbor.
        method (str, optional): Sampling method. Defaults to 'softmax'.

    Returns:
        int: Chosen neighbor index.
    """
    if method == 'uniform':
        chosen_index = torch.randint(0, len(probabilities), (1,)).item()
    elif method == 'linear':
        # Ensure probabilities are non-negative
        normalized_probs = probabilities - probabilities.min() + 1e-9
        chosen_index = torch.multinomial(normalized_probs, 1).item()
    elif method == 'softmax':
        softmax_probs = F.softmax(probabilities, dim=0)
        chosen_index = torch.multinomial(softmax_probs, 1).item()
    else:
        raise ValueError(f"Sampling method '{method}' not implemented")
    return chosen_index
