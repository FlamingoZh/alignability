import pickle
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

# aggregate both visual embeddings and language embeddings
def aggregate_embeddings_visual_and_language(input_struct,n_sample_per_visual,n_sample_per_language):
	words=input_struct['words']
	embed_dict=dict()
	for word in words:
		visual_temp=np.mean(np.array(random.sample(input_struct['embeds'][word]['visual'],n_sample_per_visual)),axis=0)
		language_temp=np.mean(np.array(random.sample(input_struct['embeds'][word]['language'],n_sample_per_language)),axis=0)
		embed_dict[word]=dict(visual=visual_temp,language=language_temp)
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

vg_noun_data=pickle.load(open("../data/dumped_embeddings/vg_noun_concept_least20_swav_bert_20.pkl","rb"))
vg_verb_data=pickle.load(open("../data/dumped_embeddings/vg_verb_concept_least20_swav_bert_20.pkl","rb"))

aggregated_vg_noun_data=aggregate_embeddings_visual_and_language(vg_noun_data,20,20)
aggregated_vg_verb_data=aggregate_embeddings_visual_and_language(vg_verb_data,20,20)

dump_sim_matrix(aggregated_vg_noun_data,"../data/processed/v_sim_mat_noun_concept_least20_swav_bert_20.txt","../data/processed/l_sim_mat_noun_concept_least20_swav_bert_20.txt")
dump_sim_matrix(aggregated_vg_verb_data,"../data/processed/v_sim_mat_verb_concept_least20_swav_bert_20.txt","../data/processed/l_sim_mat_verb_concept_least20_swav_bert_20.txt")
