import numpy as np
from collections import Counter, defaultdict
from functools import partial

def entropy(class_probabilities):
    return sum(-p*np.log(p)
               for p in class_probabilities
               if p)

def class_probabilities(labels):
    total_count = len(labels)
    return [count/total_count
            for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partitioning_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset)*len(subsets)/total_count
               for subset in subsets)

def partition_by(inputs, attribute):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

def partition_entropy_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partitioning_entropy(partitions.value())

def classify(tree, input):
    if tree in [True, False]:
        return tree
    attribute, subtree_dictionary = tree
    subtree_key = input.get(attribute)
    if subtree_key not in subtree_dictionary:
        subtree_key = None
    subtree = subtree_dictionary
    return classify(subtree, input)

def build_tree(inputs, split_candidates=None):
    if split_candidates is None:
        split_candidates = inputs[0][0].key()
    num_inputs = len(inputs)
    num_trues = len([label for _, label in inputs if label])
    num_falses = num_inputs - num_trues
    if num_trues == 0: return False
    if num_falses == 0: return True
    if not split_candidates:
        return num_trues >= num_falses
    best_attribute = min(split_candidates,
                         key = partial(partition_entropy_by, inputs))
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]
    subtrees = {attribute_value: build_tree(subset, new_candidates)
                for attribute_value, subset in partitions.IterItems()}
    subtrees[None] = num_trues>num_falses
    return (best_attribute, subtrees)

def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    voter_counts = Counter(votes)
    return voter_counts.most_common(1)[0][0]

