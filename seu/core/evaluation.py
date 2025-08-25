"""
SEU Evaluation Module

This module contains the essential test function from the original utils.py
for evaluating entity alignment results.
"""

from typing import Union, Tuple
import numpy as np
import tensorflow as tf
import numba as nb


def test(sims: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
         mode: str = "sinkhorn", 
         batch_size: int = 1024) -> None:
    """
    Test function extracted directly from utils.py for evaluating alignment results.
    
    This is the exact implementation from the original notebook's utils.py file.
    
    Args:
        sims: Similarity matrix for Sinkhorn or tuple of (row_indices, col_indices) for Hungarian
        mode: Evaluation mode - "sinkhorn" or "hungarian"
        batch_size: Batch size for processing large similarity matrices
    """
    if mode == "sinkhorn":
        results = []
        for epoch in range(len(sims) // batch_size + 1):
            sim = sims[epoch*batch_size:(epoch+1)*batch_size]
            rank = tf.argsort(-sim, axis=-1)
            ans_rank = np.array([i for i in range(epoch * batch_size, min((epoch+1) * batch_size, len(sims)))])
            results.append(tf.where(tf.equal(tf.cast(rank, ans_rank.dtype), tf.tile(np.expand_dims(ans_rank, axis=1), [1, len(sims)]))).numpy())
        results = np.concatenate(results, axis=0)
        
        @nb.jit(nopython=True)
        def cal(results):
            hits1, hits10, mrr = 0, 0, 0
            for x in results[:, 1]:
                if x < 1:
                    hits1 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1, hits10, mrr
        
        hits1, hits10, mrr = cal(results)
        print("hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (hits1/len(sims)*100, hits10/len(sims)*100, mrr/len(sims)*100))
    else:
        c = 0
        for i, j in enumerate(sims[1]):
            if i == j:
                c += 1
        print("hits@1 : %.2f%%" % (100 * c/len(sims[0])))
