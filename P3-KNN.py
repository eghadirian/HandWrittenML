from collections import Counter
import numpy as np

def vector_subtract(v,w):
    return sum(v_i - w_i for v_i, w_i in zip(v, w))

def dot(v,w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    return dot(v,v)

def magnitude (v):
    return np.sqrt(sum_of_squares(v))

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v,w))

def distance (v, w):
    return magnitude(squared_distance(v, w))

def majority_vote(labels):
    votes_counts = Counter(labels)
    winner, winner_count = votes_counts.most_common(1)[0]
    num_winners = len([
        count for count in votes_counts.values()
        if count == winner_count
    ])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])

def knn_classifier(k, labeled_points, new_point):
    by_distance = sorted(labeled_points,
                         ley = lambda (point, _): distance(point, new_point))
    k_nearest_labels = [label for _, label in by_distance[:k]]
    return majority_vote(k_nearest_labels)
