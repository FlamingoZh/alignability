import sys
import pickle
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('../python')
# from utils.utils_funcs import aggregate_embeddings_visual_and_language
from utils.utils_funcs import get_variance, get_distinctness_from_nearest_5
from utils.utils_funcs import dump_dict_data

# aggregate both visual embeddings and language embeddings
def aggregate_embeddings_visual_and_language(input_struct,n_sample_per_visual,n_sample_per_language):
	words=input_struct['words']
	embed_dict=dict()
	for word in words:
		n_sample_v = min(len(input_struct['embeds'][word]['visual']), n_sample_per_visual)
		visual_temp=np.mean(np.array(random.sample(input_struct['embeds'][word]['visual'],n_sample_v)),axis=0)
		n_sample_l = min(len(input_struct['embeds'][word]['language']), n_sample_per_language)
		language_temp=np.mean(np.array(random.sample(input_struct['embeds'][word]['language'],n_sample_l)),axis=0)
		#embed_dict[word]=dict(visual=np.expand_dims(visual_temp,axis=0),language=language_temp)
		embed_dict[word]=dict(visual=visual_temp,language=np.squeeze(language_temp))
	return dict(embeds=embed_dict,words=words)

def dump_sim_matrix(struct,visual_sim_filename,language_sim_filename):
	words=struct["words"]
	visual_embeddings_list=list()
	langauge_embeddings_list=list()
	for word in words:
		visual_embeddings_list.append(struct["embeds"][word]["visual"])
		langauge_embeddings_list.append(struct["embeds"][word]["language"])
	z_0=np.array(visual_embeddings_list)
	z_1=np.array(langauge_embeddings_list)
	sim_z_0=cosine_similarity(z_0)
	print(sim_z_0.shape)
	sim_z_1=cosine_similarity(z_1)
	with open(visual_sim_filename,"w") as f:
		for line in sim_z_0:
			for cell in line:
				f.write(str(round(cell,4))+" ")
			f.write("\n")
	with open(language_sim_filename,"w") as f:
		for line in sim_z_1:
			for cell in line:
				f.write(str(round(cell,4))+" ")
			f.write("\n")

vg_noun_data=pickle.load(open("../data/dumped_embeddings/vg_noun_ll_swav_bert_20.pkl","rb"))
vg_verb_data=pickle.load(open("../data/dumped_embeddings/vg_verb_ll_swav_bert_20.pkl","rb"))

aggregated_vg_noun_data=aggregate_embeddings_visual_and_language(vg_noun_data,20,20)
aggregated_vg_verb_data=aggregate_embeddings_visual_and_language(vg_verb_data,20,20)

### Variance
noun_visual_variance,noun_language_variance=get_variance(vg_noun_data)
verb_visual_variance,verb_language_variance=get_variance(vg_verb_data)

dump_dict_data(noun_visual_variance,"../data/processed/variance/v_variance_noun_ll_swav_bert_20.txt")
dump_dict_data(noun_language_variance,"../data/processed/variance/l_variance_noun_ll_swav_bert_20.txt")
dump_dict_data(verb_visual_variance,"../data/processed/variance/v_variance_verb_ll_swav_bert_20.txt")
dump_dict_data(verb_language_variance,"../data/processed/variance/l_variance_verb_ll_swav_bert_20.txt")

### Distinctness
noun_visual_distinctness,noun_language_distinctness=get_distinctness_from_nearest_5(vg_noun_data)
verb_visual_distinctness,verb_language_distinctness=get_distinctness_from_nearest_5(vg_verb_data)

dump_dict_data(noun_visual_distinctness,"../data/processed/distinctness/v_distinctness_noun_ll_swav_bert_20.txt")
dump_dict_data(noun_language_distinctness,"../data/processed/distinctness/l_distinctness_noun_ll_swav_bert_20.txt")
dump_dict_data(verb_visual_distinctness,"../data/processed/distinctness/v_distinctness_verb_ll_swav_bert_20.txt")
dump_dict_data(verb_language_distinctness,"../data/processed/distinctness/l_distinctness_verb_ll_swav_bert_20.txt")

### Similarity Matrix

dump_sim_matrix(aggregated_vg_noun_data,"../data/processed/similarity_matrix/v_sim_mat_noun_ll_swav_bert_20.txt","../data/processed/similarity_matrix/l_sim_mat_noun_ll_swav_bert_20.txt")
dump_sim_matrix(aggregated_vg_verb_data,"../data/processed/similarity_matrix/v_sim_mat_verb_ll_swav_bert_20.txt","../data/processed/similarity_matrix/l_sim_mat_verb_ll_swav_bert_20.txt")
