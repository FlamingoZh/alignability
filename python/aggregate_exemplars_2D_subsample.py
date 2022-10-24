import pickle
import numpy as np
import pprint
import random
import sys
import argparse

from permutation import permutation
from utils.utils_funcs import get_aggregated_embeddings_2D

def sample_exemplars_from_data(input_struct,n_sample_per_visual,n_sample_per_language):
    words=input_struct['words']
    embed_dict=dict()
    for word in words:
        assert len(input_struct['embeds'][word]['visual'])>=n_sample_per_visual
        n_sample_v = min(len(input_struct['embeds'][word]['visual']), n_sample_per_visual)
        visual_temp=np.array(random.sample(input_struct['embeds'][word]['visual'],n_sample_v))

        assert len(input_struct['embeds'][word]['language'])>=n_sample_per_language
        n_sample_l = min(len(input_struct['embeds'][word]['language']), n_sample_per_language)
        language_temp=np.array(random.sample(input_struct['embeds'][word]['language'],n_sample_l))

        embed_dict[word]=dict(visual=np.squeeze(visual_temp),language=np.squeeze(language_temp))
    return dict(embeds=embed_dict,words=words)  

def simulate_exemplar_aggregation_2D(datasetname, data, aggregation_mode, n_l_exemplar_max=20, n_v_exemplar_max=20, n_sample=1, extra_info=None):
    y_mat = np.zeros((n_l_exemplar_max, n_v_exemplar_max, n_sample))
    print("Start simulation...")
    for sample in range(1, n_sample + 1):
        print(f"Sample {sample}:")

        sampled_data=sample_exemplars_from_data(data, n_v_exemplar_max, n_l_exemplar_max)

        v_exemplar_all=list()
        v_exemplar_indices=np.arange(n_v_exemplar_max)
        v_exemplar_all.append(v_exemplar_indices)
        for i in range(n_v_exemplar_max,1,-1):
            v_exemplar_indices=np.random.choice(v_exemplar_indices,len(v_exemplar_indices)-1, replace=False)
            v_exemplar_all.append(v_exemplar_indices)

        l_exemplar_all=list()
        l_exemplar_indices=np.arange(n_l_exemplar_max)
        l_exemplar_all.append(l_exemplar_indices)
        for i in range(n_l_exemplar_max,1,-1):
            l_exemplar_indices=np.random.choice(l_exemplar_indices,len(l_exemplar_indices)-1, replace=False)
            l_exemplar_all.append(l_exemplar_indices)

        for n_v_exemplar in range(n_v_exemplar_max,0,-1):
            for n_l_exemplar in range(n_l_exemplar_max,0,-1):
                print("visual_exemplar: {}, language_exemplar: {}".format(n_v_exemplar, n_l_exemplar))

                #compute alignment strength
                words=sampled_data['words']
                visual_agg=list()
                lang_agg=list()
                # aggregate embeddings
                for word in words:
                    visual_agg.append(np.mean(sampled_data['embeds'][word]['visual'][v_exemplar_all[n_v_exemplar_max-n_v_exemplar]],axis=0))
                    lang_agg.append(np.mean(sampled_data['embeds'][word]['language'][l_exemplar_all[n_l_exemplar_max-n_l_exemplar]],axis=0))
                
                z_0 = np.stack(visual_agg)
                z_1 = np.stack(lang_agg)

                relative_alignment_strength, alignment_strength_list = permutation(z_0, z_1)
                y_mat[n_v_exemplar - 1][n_l_exemplar - 1][sample - 1]=relative_alignment_strength
                print("Relative Alignment: {}".format(relative_alignment_strength))               

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
    print("\nData for plotting are dumped to /data/dumped_plot_data/" + file_name + ".pkl")

if __name__ == '__main__':

    print("aggregate_exemplars_2D_subsample.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("datasetname", help="name of the dataset")
    parser.add_argument("data", help="data to read from")
    parser.add_argument("aggregation_mode", help="aggregation mode: language, visual, visual_language, visual_language_subsample")
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
