def accuracy(tp, fp, tn, fn):
    correct = tp + tn
    total = tp + fp + tn + fn
    return correct/total

def precision(tp, fp, tn, fn): # measures how accurate our positive predictions are
    return tp/(tp +fp)

def recall(tp, fp, tn, fn): # measures what fraction of positives the model identifies
    return tp/(tp + fn)

def f1_score(tp, fp, tn, fn): # harmonic mean of recall ad precision. lies between them
    p = precision(tp, fp, tn, fn)
    r = recall(tp, fp, tn, fn)
    return 2*r*p/(p+r)

