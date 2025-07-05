"""
metrics.py

Functions to compute Character Error Rate (CER) and Word Error Rate (WER) for OCR evaluation.
"""

def levenshtein(s1, s2):
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def cer(pred, gt):
    """Character Error Rate: Levenshtein distance / length of ground truth."""
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return levenshtein(pred, gt) / len(gt)

def wer(pred, gt):
    """Word Error Rate: Levenshtein distance on word level / number of words in ground truth."""
    pred_words = pred.split()
    gt_words = gt.split()
    if len(gt_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    return levenshtein(pred_words, gt_words) / len(gt_words) 