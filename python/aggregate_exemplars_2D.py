import pickle
import numpy as np
import pprint
import random
import sys
import argparse

from permutation import permutation
from utils.utils_funcs import get_aggregated_embeddings_2D

def simulate_exemplar_aggregation_2D(datasetname, data, aggregation_mode, n_l_exemplar_max=20, n_v_exemplar_max=20, n_sample=1, extra_info=None):

    y_mat= np.zeros((n_l_exemplar_max,n_v_exemplar_max))

    print("Start simulation...")
    for n_l_exemplar in range(1, n_l_exemplar_max + 1):
        for n_v_exemplar in range(1, n_v_exemplar_max + 1):
            sample_mean=list()
            for sample in range(1, n_sample + 1):
                print(f"n_l_exemplar {n_l_exemplar}, n_v_exemplar {n_v_exemplar}, sample {sample}")

                z_0, z_1 = get_aggregated_embeddings_2D(data, n_v_exemplar, n_l_exemplar, aggregation_mode)

                relative_alignment_strength, alignment_strength_list = permutation(z_0, z_1)
                sample_mean.append(relative_alignment_strength)

            print(f"Mean relative alignment strength: {np.round(np.mean(sample_mean), 6)}")
            y_mat[n_l_exemplar-1][n_v_exemplar-1]=np.mean(sample_mean)

    plot_data = dict(
        y_mat=y_mat,
        datasetname=datasetname,
        n_l_exemplar_max=n_l_exemplar_max,
        n_v_exemplar_max=n_v_exemplar_max,
        n_sample=n_sample,
        aggregation_mode=aggregation_mode,
        extra_info=extra_info
    )

    if extra_info:
        file_name = "_".join(["2D", datasetname, aggregation_mode, str(n_l_exemplar_max), str(n_v_exemplar_max), str(n_sample), extra_info])
    else:
        file_name = "_".join(["2D", datasetname, aggregation_mode, str(n_l_exemplar_max), str(n_v_exemplar_max), str(n_sample)])

    pickle.dump(plot_data, open('../data/dumped_plot_data/' + file_name + '.pkl', 'wb'))
    print("\n data for plotting are dumped to /data/dumped_plot_data/" + file_name + ".pkl")

if __name__ == '__main__':

    print("aggregate_exemplars.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("datasetname", help="name of the dataset")
    parser.add_argument("data", help="data to read from")
    parser.add_argument("aggregation_mode", help="aggregation mode: language, visual, language_visual")
    parser.add_argument("--n_l_exemplar_max", type=int, default=20,
                        help="maximum number of language exemplars in simulation")
    parser.add_argument("--n_v_exemplar_max", type=int, default=20,
                        help="maximum number of visual exemplars in simulation")
    parser.add_argument("--n_sample", type=int, default=1, help="number of samples in each run")
    parser.add_argument("--extra_info", default=None, help="extra information")
    args = parser.parse_args()

    data = pickle.load(open(args.data, 'rb'))
    simulate_exemplar_aggregation_2D(args.datasetname,
                                  data,
                                  args.aggregation_mode,
                                  n_l_exemplar_max=args.n_l_exemplar_max,
                                  n_v_exemplar_max=args.n_v_exemplar_max,
                                  n_sample=args.n_sample,
                                  extra_info=args.extra_info
                                  )
