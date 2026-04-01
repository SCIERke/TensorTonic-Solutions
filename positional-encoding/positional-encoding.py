import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    seqs = []
    for pos in range(seq_len):
        seq = []
        for idx in range(d_model):
            exponent = (2 * (idx // 2)) / d_model
            denom = np.power(base, exponent)
            
            if idx % 2 == 0:
                seq.append(np.sin(pos / denom))
            else:
                seq.append(np.cos(pos / denom))
        seqs.append(seq)

    return np.array(seqs)