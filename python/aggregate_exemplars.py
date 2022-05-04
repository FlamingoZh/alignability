import pickle
import numpy as np
import pprint
import random
import sys
import argparse

from permutation import permutation
from utils.utils_funcs import get_aggregated_embeddings

def simulate_exemplar_aggregation(datasetname, data, aggregation_mode, n_exemplar_max=20, n_sample=1, extra_info=None):

    mean_list = [0]
    max_list = [0]
    min_list = [0]
    n_exemplar_list = [0]
    y_list= [0]

    print("Start simulation...")
    for n_exemplar in range(1, n_exemplar_max + 1):
        # r_AS_list=list()
        # a_AS_list=list()
        for sample in range(1, n_sample + 1):
            print(f"n_exemplar {n_exemplar}, sample {sample}")

            z_0, z_1 = get_aggregated_embeddings(data, n_exemplar, aggregation_mode)

            relative_alignment_strength, alignment_strength_list = permutation(z_0, z_1)
            # r_AS_list.append(relative_alignment_strength)
            # a_AS_list.append(alignment_strength_list[0])
            n_exemplar_list.append(n_exemplar)
            y_list.append(relative_alignment_strength)
            print(f"Relative alignment strength: {np.round(relative_alignment_strength, 6)}")

        # n_exemplar_list.append(n_exemplar)
        # mean_list.append(np.mean(r_AS_list))
        # max_list.append(np.max(r_AS_list))
        # min_list.append(np.min(r_AS_list))
        # print(f"Relative alignment strength: Max: {np.round(np.max(r_AS_list), 6)}, Mean: {np.round(np.mean(r_AS_list), 6)} Min: {np.round(np.min(r_AS_list), 6)}")
        # print(f"Absolute alignment strength: Max: {np.round(np.max(a_AS_list), 6)}, Mean: {np.round(np.mean(a_AS_list), 6)} Min: {np.round(np.min(a_AS_list), 6)}")

    plot_data = dict(
        n_exemplar_list=n_exemplar_list,
        y_list=y_list,
        datasetname=datasetname,
        n_exemplar_max=n_exemplar_max,
        n_sample=n_sample,
        aggregation_mode=aggregation_mode,
        extra_info=extra_info
    )

    if extra_info:
        file_name = "_".join([datasetname, aggregation_mode, str(n_exemplar_max), str(n_sample),extra_info])
    else:
        file_name = "_".join([datasetname, aggregation_mode, str(n_exemplar_max), str(n_sample)])

    pickle.dump(plot_data, open('../data/dumped_plot_data/' + file_name + '.pkl', 'wb'))
    print("\n data for plotting are dumped to /data/dumped_plot_data/" + file_name + ".pkl")

if __name__ == '__main__':

    print("aggregate_exemplars.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("datasetname", help="name of the dataset")
    parser.add_argument("data", help="data to read from")
    parser.add_argument("aggregation_mode", help="aggregation mode: language, visual, language_visual")
    parser.add_argument("--n_exemplar_max", type=int, default=20, help="maximum number of exemplars in simulation")
    parser.add_argument("--n_sample", type=int, default=1, help="number of samples in each run")
    parser.add_argument("--extra_info", default=None, help="extra information")
    args = parser.parse_args()

    data = pickle.load(open(args.data, 'rb'))
    simulate_exemplar_aggregation(args.datasetname,
                                  data,
                                  args.aggregation_mode,
                                  n_exemplar_max=args.n_exemplar_max,
                                  n_sample=args.n_sample,
                                  extra_info=args.extra_info
                                  )
