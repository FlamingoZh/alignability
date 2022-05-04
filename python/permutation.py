import numpy as np
import random

from scipy.stats import spearmanr

from sklearn.metrics.pairwise import cosine_similarity

def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

def compute_alignment_strength(z_0,z_1):
    sim_z_0=cosine_similarity(z_0)
    sim_z_1=cosine_similarity(z_1)
    return spearmanr(upper_tri_indexing(sim_z_0), upper_tri_indexing(sim_z_1))[0]

def permutation(z_0,z_1,n_sim=1000):
    alignment_strength_list=list()
    alignment_strength_list.append(compute_alignment_strength(z_0,z_1)) #true mapping system
    for i_sim in range(n_sim):
        alignment_strength_list.append(compute_alignment_strength(z_0, np.random.permutation(z_1)))
    count=0
    for ele in alignment_strength_list[1:]:
        if alignment_strength_list[0]>ele:
            count+=1
    relative_alignment_strength=count/(len(alignment_strength_list)-1)
    return relative_alignment_strength, alignment_strength_list

if __name__ == '__main__':

    print("permutation.py")

    A = np.array([
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [7, 8, 9, 10]])
    B = np.array([
        [1, 6, 100],
        [4, 5, 2],
        [7, 8, 9]])
    print(permutation(A,B))
