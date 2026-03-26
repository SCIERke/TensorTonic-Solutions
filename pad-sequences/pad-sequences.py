import numpy as np
import pandas as pd

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    padded_seqs = []
    if not(max_len):
        max_len = max([ len(seq) for seq in seqs])

    for seq in seqs:
        curr_seq = list(seq)
        if (len(curr_seq) < max_len):
            curr_seq += [pad_value] * (max_len - len(curr_seq))
        if (len(seq)> max_len):
            curr_seq = curr_seq[:max_len]

        padded_seqs.append(curr_seq)

    final_array = np.array(padded_seqs)

    return final_array