# Adapted from: https://github.com/Gudora/Clonal-Tree.git
# Original author: @Gudora
# License: MIT License

import random

from ete3 import Tree
from pathlib import Path
import numpy as np

def build_tree(file_path, index):
    """
    Builds a tree from a parent pointer list in a specified line of a file.
    """
    with open(file_path, 'r') as file:
        parent_data = file.readlines()[index].strip().split()

    node_dict = {}

    for idx, parent_id in enumerate(parent_data, start=1):
        if idx not in node_dict:
            node_dict[idx] = Tree(name=str(idx))
        if int(parent_id) not in node_dict:
            node_dict[int(parent_id)] = Tree(name=str(parent_id))

        node_dict[int(parent_id)].add_child(node_dict[idx])
    
    return node_dict

def get_final_mutations_L(file_path, index):
    """
    Reads the final mutation states from a specified line in a file.
    """
    with open(file_path, 'r') as file:
        final_distribution = file.readlines()[index].strip().split()
    
    return [str(num) for num in final_distribution]

def sample_final_mutations(final_distribution, k):
    """
    Randomly samples a subset of final mutations.
    """
    return set(random.sample(final_distribution, k))

def find_all_nodes(tree, nodes):
    """
    Finds all nodes required to span a given set of nodes, including their ancestors.
    """
    necessary_nodes = set(nodes)
    for node_id in nodes:
        current_node = tree.search_nodes(name=str(node_id))[0]
        while current_node.up: 
            necessary_nodes.add(current_node.up.name)
            current_node = current_node.up
    return necessary_nodes

def count_non_empty_lines(file_path):
    """
    Counts the number of non-empty lines in a text file.
    """
    path = Path(file_path)
    if not path.exists():
        return 0

    with open(file_path, 'r') as f:
        return sum(1 for line in f if line.strip())
    
def get_coalescence_time(tree, samples):
    """
    Computes the coalescence time of a set of sample nodes in a tree.
    """
    samples = set(samples)
    if len(samples) <= 1: return 0
    ancestor = tree.get_common_ancestor(samples)
    max_distance = max(ancestor.get_distance(tree & sample) for sample in samples)
    return max_distance

def compute_subtree_sizes(node):
    """
    Recursively computes and stores the size of each subtree rooted at a given node.
    
    Args:
        node (TreeNode): A node in the tree (e.g., from ete3) for which subtree sizes are to be computed.

    Returns:
        (S_i, S*_i):
            S_i (int): Total size of the subtree rooted at the node, including the node itself.
            S*_i (int): Size of the subtree excluding the node itself (i.e., sum of all descendants).

    Side Effects:
        Adds the following attributes to each node:
            node.S_i (int): Size of the subtree rooted at this node (including the node).
            node.S_star_i (int): Size of the subtree excluding this node (descendants only).
            node.child_sizes (list of int): List of subtree sizes for each child.
    """
    if node.is_leaf():
        node.S_i = 1  # A leaf has size 1 (itself)
        node.S_star_i = 0  # No subtree without root
        return 1, 0  

    subtree_size = 1  # Count itself
    child_sizes = []

    for child in node.children:
        child_s, _ = compute_subtree_sizes(child)
        child_sizes.append(child_s)
        subtree_size += child_s  # Accumulate child subtree sizes
    
    S_star_i = subtree_size - 1  # Exclude root itself
    
    # Store values as attributes inside the node
    node.S_i = subtree_size
    node.S_star_i = S_star_i
    node.child_sizes = child_sizes  # Store sizes of children

    return subtree_size, S_star_i

def compute_balance_scores(node):
    """
    Computes and stores the balance score W_i for a tree node based on its child subtree sizes.
    """
    if node.is_leaf() or len(node.children) < 2:
        node.W_i = 0
        return 0
    
    p_ij = np.array(
        [child.S_i / node.S_star_i for child in node.children]
    ) # fraction of descendants contributed by each child
    
    W_i_1 = -np.sum(
        p_ij * np.log(p_ij) / np.log(len(node.children))
    ) # normalized Shannon entropy
    
    node.W_i = W_i_1
    return W_i_1

def compute_normalized_balance_index(root):
    """
    Compute the normalized tree balance index J^1.
    """
    internal_nodes = [n for n in root.traverse() if not n.is_leaf()]
    
    S_star_sum = sum(n.S_star_i for n in internal_nodes)
    weighted_sum = sum(n.S_star_i / n.S_i * n.W_i for n in internal_nodes)
    
    return weighted_sum / S_star_sum if S_star_sum > 0 else 0

def get_sackin_index(tree):
    """
    Compute the Sackin Index of tree.
    """
    leaves = tree.get_leaves()  
    total_depth = sum(leaf.get_distance(tree) for leaf in leaves)  
    num_leaves = len(leaves)  
    if num_leaves > 1:
        normalized_index = total_depth / (0.5 * num_leaves * (num_leaves + 1) - 1)
    else:
        normalized_index = 0  
    
    # # tree depth
    # tree_depth = max(leaf.get_distance(tree) for leaf in leaves) if leaves else 0
    
    # # tree width
    # levels = {}
    # for leaf in leaves:
    #     level = leaf.get_distance(tree, topology_only=True)
    #     if level not in levels:
    #         levels[level] = 0
    #     levels[level] += 1
    # tree_width = max(levels.values()) if levels else 0

    return normalized_index