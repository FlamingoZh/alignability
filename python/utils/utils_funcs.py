import os
import sys
import numpy as np
import pickle
import pathlib
import random
from scipy.stats import spearmanr

def test():
	print("hello wolrd")

def dump_dict_data(dic,file_name):
	with open(file_name,"w") as f:
		for word in dic:
			f.write(word+" "+str(dic[word])+"\n")

def load_data_to_dict(file_name,concepts=None,header=False):
	dic=dict()
	with open(file_name,"r") as f:
		if header:
			next(f)
		if concepts:
			for row in f:
				temp=row.replace("\n","").replace('\r',"").split(" ")
				if temp[0] in concepts:
					dic[temp[0]]=float(temp[1])
		else:
			for row in f:
				temp=row.replace("\n","").replace('\r',"").split(" ")
				dic[temp[0]]=float(temp[1])			
	return dic

def get_variance(struct):
	words=struct["words"]
	visual_variance=dict()
	language_variance=dict()
	for word in words:
		visual_embeddings=struct["embeds"][word]["visual"]
		visual_variance[word]=np.mean(np.linalg.norm(visual_embeddings-np.mean(visual_embeddings,axis=0),axis=1))
		language_embeddings=struct["embeds"][word]["language"]
		language_variance[word]=np.mean(np.linalg.norm(language_embeddings-np.mean(language_embeddings,axis=0),axis=1))
	return visual_variance,language_variance

# def get_distinctness(struct):
# 	words=struct["words"]
# 	visual_centers=dict()
# 	language_centers=dict()
# 	for word in words:
# 		visual_embeddings=np.array(struct["embeds"][word]["visual"])
# 		visual_centers[word]=np.mean(visual_embeddings,axis=0)
# 		language_embeddings=np.array(struct["embeds"][word]["language"])
# 		language_centers[word]=np.mean(language_embeddings,axis=0)
# 	visual_distinctness=dict()
# 	language_distinctness=dict()
# 	for word in words:
# 		visual_distances=[np.linalg.norm(visual_centers[word]-visual_centers[temp]) for temp in words]
# 		visual_distinctness[word]=np.sum(visual_distances)/(len(visual_distances)-1)
# 		language_distances=[np.linalg.norm(language_centers[word]-language_centers[temp]) for temp in words]
# 		language_distinctness[word]=np.sum(language_distances)/(len(language_distances)-1)
# 	return visual_distinctness,language_distinctness

def get_distinctness(struct):
	words=struct["words"]
	visual_centers=dict()
	language_centers=dict()
	for word in words:
		visual_embeddings=np.array(struct["embeds"][word]["visual"])
		visual_centers[word]=np.mean(visual_embeddings,axis=0)
		language_embeddings=np.array(struct["embeds"][word]["language"])
		language_centers[word]=np.mean(language_embeddings,axis=0)
	visual_distinctness=dict()
	language_distinctness=dict()
	for word in words:
		# visual_distances=[np.linalg.norm(visual_centers[word]-visual_centers[temp]) for temp in words]
		visual_distances=list()
		for embed in struct["embeds"][word]["visual"]:
			for temp in words:
				visual_distances.append(np.linalg.norm(embed-visual_centers[temp]))
		visual_distinctness[word]=np.sum(visual_distances)/len(visual_distances)
		#language_distances=[np.linalg.norm(language_centers[word]-language_centers[temp]) for temp in words]
		language_distances=list()
		for embed in struct["embeds"][word]["language"]:
			for temp in words:
				language_distances.append(np.linalg.norm(embed-language_centers[temp]))
		language_distinctness[word]=np.sum(language_distances)/len(language_distances)
	return visual_distinctness,language_distinctness

def get_distinctness_from_nearest_5(struct):
	words=struct["words"]
	visual_centers=dict()
	language_centers=dict()
	for word in words:
		visual_embeddings=np.array(struct["embeds"][word]["visual"])
		visual_centers[word]=np.mean(visual_embeddings,axis=0)
		language_embeddings=np.array(struct["embeds"][word]["language"])
		language_centers[word]=np.mean(language_embeddings,axis=0)
	visual_distinctness=dict()
	language_distinctness=dict()
	for word in words:
		visual_distances=[np.linalg.norm(visual_centers[word]-visual_centers[temp]) for temp in words]
		visual_distinctness[word]=np.sum(sorted(visual_distances)[:6])/5
		language_distances=[np.linalg.norm(language_centers[word]-language_centers[temp]) for temp in words]
		language_distinctness[word]=np.sum(sorted(language_distances)[:6])/5
	return visual_distinctness,language_distinctness

# aggregate visual embeddings
def aggregate_embeddings_visual(input_struct,n_sample_per_category):
	words=input_struct['words']
	embed_dict=dict()
	for word in words:
		n_sample=min(len(input_struct['embeds'][word]['visual']),n_sample_per_category)
		visual_temp=np.mean(np.array(random.sample(input_struct['embeds'][word]['visual'],n_sample)),axis=0)
		language_temp=np.mean(np.array(input_struct['embeds'][word]['language']),axis=0)
		embed_dict[word]=dict(visual=visual_temp,language=language_temp)
	return dict(embeds=embed_dict,words=words)

# aggregate language embeddings
def aggregate_embeddings_language(input_struct,n_sample_per_category):
	words=input_struct['words']
	embed_dict=dict()
	for word in words:
		visual_temp=np.mean(np.array(input_struct['embeds'][word]['visual']),axis=0)
		n_sample = min(len(input_struct['embeds'][word]['language']), n_sample_per_category)
		language_temp=np.mean(np.array(random.sample(input_struct['embeds'][word]['language'],n_sample)),axis=0)
		embed_dict[word]=dict(visual=visual_temp,language=language_temp)
	return dict(embeds=embed_dict,words=words)

def get_aggregated_embeddings(data, n_sample_per_category, aggregation_mode):
	if aggregation_mode == 'visual':
		# Aggregation mode: visual only
		aggregated_data = aggregate_embeddings_visual(data, n_sample_per_category)
	elif aggregation_mode == 'language':
		# Aggregation mode: language only
		aggregated_data = aggregate_embeddings_language(data, n_sample_per_category)
	else:
		print("Error, unrecognized aggregation mode.")
		sys.exit(1)

	z_0 = np.stack([embeds['visual'] for embeds in aggregated_data['embeds'].values()])
	z_1 = np.stack([embeds['language'] for embeds in aggregated_data['embeds'].values()])
	return z_0, z_1

# aggregate both visual embeddings and language embeddings
def aggregate_embeddings_visual_and_language(input_struct,n_sample_per_visual,n_sample_per_language):
	words=input_struct['words']
	embed_dict=dict()
	for word in words:
		n_sample_v = min(len(input_struct['embeds'][word]['visual']), n_sample_per_visual)
		visual_temp=np.mean(np.array(random.sample(input_struct['embeds'][word]['visual'],n_sample_v)),axis=0)
		n_sample_l = min(len(input_struct['embeds'][word]['language']), n_sample_per_language)
		language_temp=np.mean(np.array(random.sample(input_struct['embeds'][word]['language'],n_sample_l)),axis=0)
		embed_dict[word]=dict(visual=np.squeeze(visual_temp),language=np.squeeze(language_temp))
	return dict(embeds=embed_dict,words=words)

def get_aggregated_embeddings_2D(data, n_sample_per_visual, n_sample_per_language, aggregation_mode):
	if aggregation_mode == 'visual_language':
		#Aggregation mode: visual+language
		aggregated_data = aggregate_embeddings_visual_and_language(data, n_sample_per_visual, n_sample_per_language)
	else:
		print("Error, unrecognized aggregation mode.")
		sys.exit(1)

	z_0 = np.stack([embeds['visual'] for embeds in aggregated_data['embeds'].values()])
	z_1 = np.stack([embeds['language'] for embeds in aggregated_data['embeds'].values()])
	return z_0, z_1

def call_fintuned_BERT_sentence(concepts,packed_samples,finetune_path,pos,cuda=False):
	print("Forward Propagation in BERT.")

	import torch
	import transformers
	from transformers import (
		CONFIG_MAPPING,
		MODEL_FOR_MASKED_LM_MAPPING,
		AutoConfig,
		AutoModelForMaskedLM,
		AutoTokenizer,
		DataCollatorForLanguageModeling,
		HfArgumentParser,
		Trainer,
		TrainingArguments,
		set_seed,
	)
	# silence mode
	from transformers import logging
	logging.set_verbosity_warning()
	logging.set_verbosity_error()

	# load fine-tuned base bert
	config = AutoConfig.from_pretrained(finetune_path)
	model = AutoModelForMaskedLM.from_pretrained(finetune_path, output_hidden_states=True)
	tokenizer = AutoTokenizer.from_pretrained(finetune_path)
	if cuda:
		model = model.cuda()
	model.eval()

	language_embeddings_dict = dict()

	i_concept = 0

	for concept, samples_of_the_concept in zip(concepts, packed_samples):
		i_concept += 1
		print("Computing word embedding of", concept, '(' + str(i_concept) + '/' + str(len(concepts)) + ')' + "...")
		if pos=="noun":
			word_token = "[N" + str(concepts.index(concept) + 1) + "]"
		else:
			word_token = "[V" + str(concepts.index(concept) + 1) + "]"
		word_embedding_list = list()
		sampled_contexts, corresponding_words = samples_of_the_concept
		for text, target_word in zip(sampled_contexts, corresponding_words):
			marked_text = "[CLS] " + text + " [SEP]"
			tokenized_text = tokenizer.tokenize(marked_text)
			indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

			segments_ids = [1] * len(tokenized_text)

			tokens_tensor = torch.tensor([indexed_tokens])
			segments_tensors = torch.tensor([segments_ids])

			if cuda:
				tokens_tensor = tokens_tensor.cuda()
				segments_tensors = segments_tensors.cuda()

			# forward propagation
			with torch.no_grad():
				outputs = model(tokens_tensor, segments_tensors)
				hidden_states = outputs["hidden_states"]

			token_embeddings = torch.stack(hidden_states, dim=0)
			token_embeddings = torch.squeeze(token_embeddings, dim=1)
			token_embeddings = token_embeddings.permute(1, 0, 2)

			# take the summation of the outputs of the last four layers as the contextualized embedding
			token_vecs_sum = []
			for token in token_embeddings:
				sum_vec = torch.sum(token[-4:], dim=0)
				if cuda:
					sum_vec = sum_vec.data.cpu().numpy()
				else:
					sum_vec = sum_vec.numpy()
				token_vecs_sum.append(sum_vec)
			tokenized_target_word_list = tokenizer.tokenize(word_token)
			word_embedding = list()
			for sub_word in tokenized_target_word_list:
				if sub_word not in tokenized_text:
					print(tokenized_text)
				word_index = tokenized_text.index(sub_word)
				word_embedding.append(np.array(token_vecs_sum[word_index]))
			word_embedding = np.mean(np.array(word_embedding), axis=0)
			word_embedding_list.append(word_embedding)
		language_embeddings_dict[concept] = word_embedding_list

	return language_embeddings_dict

def call_pretrained_BERT(concepts,packed_samples,cuda=False,mask=False):
	print("Forward Propagation in BERT.")

	import torch
	from transformers import BertModel, BertTokenizer

	# silence mode
	from transformers import logging
	logging.set_verbosity_warning()
	logging.set_verbosity_error()

	# load pretrained base bert; if you haven't downloaded the model yet, it will be downloaded automatically
	model_name = 'bert-base-uncased'
	tokenizer = BertTokenizer.from_pretrained(model_name)
	model = BertModel.from_pretrained(model_name, output_hidden_states = True)
	if cuda:
		model=model.cuda()
	model.eval()

	language_embeddings_dict=dict()

	i_concept=0

	for concept, samples_of_the_concept in zip(concepts,packed_samples):
		i_concept+=1
		sampled_contexts, corresponding_words = samples_of_the_concept
		print("Computing word embedding of", concept, corresponding_words[0], '('+str(i_concept)+'/'+str(len(concepts))+')'+"...")
		word_embedding_list=list()

		for text, target_word in zip(sampled_contexts,corresponding_words):
			marked_text = "[CLS] " + text + " [SEP]"
			if mask:
				marked_text=marked_text.replace(target_word,"[MASK]")
			tokenized_text = tokenizer.tokenize(marked_text)
			indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

			segments_ids = [1] * len(tokenized_text)

			tokens_tensor = torch.tensor([indexed_tokens])
			segments_tensors = torch.tensor([segments_ids])

			if cuda:
				tokens_tensor=tokens_tensor.cuda()
				segments_tensors=segments_tensors.cuda()

			# forward propagation
			with torch.no_grad():
				outputs = model(tokens_tensor, segments_tensors)
				hidden_states = outputs[2]

			token_embeddings = torch.stack(hidden_states, dim=0)
			token_embeddings = torch.squeeze(token_embeddings, dim=1)
			token_embeddings = token_embeddings.permute(1,0,2)

			# take the summation of the outputs of the last four layers as the contextualized embedding
			token_vecs_sum = []
			for token in token_embeddings:
				sum_vec = torch.squeeze(token[-1:])
				#sum_vec = torch.sum(token[-4:], dim=0)
				if cuda:
					sum_vec=sum_vec.data.cpu().numpy()
				else:
					sum_vec=sum_vec.numpy()
				token_vecs_sum.append(sum_vec)
			if mask:
				tokenized_target_word_list=tokenizer.tokenize("[MASK]")
			else:
				tokenized_target_word_list=tokenizer.tokenize(target_word)
			word_embedding=list()
			for sub_word in tokenized_target_word_list:
				word_index=tokenized_text.index(sub_word)
				word_embedding.append(np.array(token_vecs_sum[word_index]))
			word_embedding=np.mean(np.array(word_embedding),axis=0)
			word_embedding_list.append(word_embedding)
		language_embeddings_dict[concept]=word_embedding_list

	return language_embeddings_dict

def sample_sentence_from_corpus(concepts,n_sample,pos=None,corpus_name='wiki_en',window_size=15):
	# import linecache
	# please modify the below path to your corpus
	if corpus_name=="wiki_en":
		path_to_corpus='/user_data/yuchenz2/word_embedding_training/corpora/wiki_en.txt'
	elif corpus_name=="wiki_en_subset":
		path_to_corpus='/user_data/yuchenz2/word_embedding_training/corpora/wiki_en_1000.txt'
	else:
		print("Error, unrecognized corpus name.")
		sys.exit(1)
	if not os.path.exists(path_to_corpus):
		print("Cannot find the corpus file, please check the path to the corpus.")
		sys.exit(1)
	# print("counting lines in",corpus_name,"...")
	# total_lines = sum(1 for _ in open(path_to_corpus,'r'))
	# print("total lines in corpus:",total_lines)

	corpus_content=list()
	with open(path_to_corpus, 'r') as f:
		for line in f:
			corpus_content.append(line)
	print(len(corpus_content))

	packed_samples=list()
	for i, target_word in enumerate(concepts):
		print("Start sampling word", target_word, '('+str(i+1)+'/'+str(len(concepts))+')'+"...")
		i_sample=0
		sampled_contexts=list()
		corresponding_words=list()
		random.shuffle(corpus_content)
		for line_content in corpus_content:
			if i_sample>=n_sample:
				break
			line_split=line_content.split(" ")
			
			# retrive all word forms of the target word
			word_set=set()
			word_set.add(target_word)
			# if pos=="noun":
			# 	from pattern.en import pluralize
			# 	word_set.add(pluralize(target_word))
			# elif pos=="verb":
			# 	from pattern.en import lexeme
			# 	[word_set.add(item) for item in lexeme(target_word)]
			# else:
			# 	print("Error: wrong part-of-speech category.")
			# 	sys.exit(1)
			
			for word in word_set:
				if word in line_split:
					word_index=line_split.index(word)
					start_index=max(0,word_index-window_size//2)
					end_index=min(len(line_split),word_index+window_size//2)
					context=" ".join(line_split[start_index:end_index])
					#print(start_index,end_index,context)
					sampled_contexts.append(context)
					corresponding_words.append(word)
					i_sample+=1
					#print("sampled sentence:",i_image,"/",n_sample)
		if i_sample>=n_sample:
			packed_samples.append((sampled_contexts,corresponding_words))
		else:
			print("Warning, didn't find enough samples for target word", target_word, ", only find ", str(i_sample), " samples.")
			packed_samples.append((sampled_contexts, corresponding_words))
	print("Sampling Completed.")
	return packed_samples



#######################

def replace_language_embeddings(struct_name,language_dict_name):
	struct=pickle.load(open(struct_name,'rb'))
	dictionary=pickle.load(open(language_dict_name,'rb'))
	dict_keys=list(dictionary.keys())
	concepts=struct['words']

	embed_dict = dict()
	for concept in concepts:
		if concept in dict_keys:
			embed_dict[concept]=dict(
				visual=struct['embeds'][concept]['visual'],
				language=[dictionary[concept]])
		else:
			embed_dict[concept] = dict(
				visual=struct['embeds'][concept]['visual'],
				language=[random.choice(list(dictionary.values()))])
	new_struct=dict(
		embeds=embed_dict,
		words=concepts
	)
	return new_struct

def dimention_reduction_TSNE(vectors,perplexity=30):
	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2, random_state=0,perplexity=perplexity)
	Y = tsne.fit_transform(vectors)
	return Y

# generate n distinct colors
def gen_distinct_colors(num_colors):
	import colorsys
	colors=[]
	for i in np.arange(0., 360., 360. / num_colors):
		hue = i/360.
		lightness = (50 + np.random.rand() * 10)/100.
		saturation = (90 + np.random.rand() * 10)/100.
		colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
	return np.array(colors)

def search_index(word_list,word_order):
	index_list=list()
	for item in word_list:
		index_list.append(word_order.index(item))
	return index_list

def load_pretrained_Glove(filename='../pretrained_models/glove.840B.300d.txt',dumpname='../pretrained_models/glove_dict_840B.pkl'):
	if not pathlib.Path(dumpname).is_file():
		embeddings_dict = {}
		if not pathlib.Path(filename).is_file():
			print("Error in loading pretrained Glove, cannot find", filename, 'or', dumpname)
			sys.exit(1)
		with open(filename, 'r', encoding="utf-8") as f:
			ii=0
			for line in f:
				ii+=1
				values = line.split()
				#print(values)
				word = values[0]
				try:
					vector = np.asarray(values[1:], "float32")
					embeddings_dict[word] = vector
				except:
					print(values[0:5],ii)
		pickle.dump(embeddings_dict, open(dumpname, 'wb'))
	else:
		embeddings_dict=pickle.load(open(dumpname,'rb'))

	return embeddings_dict

def linear_interpolation(y, factor=10):
	"""Interpolate additional points for piece-wise linear function."""
	n_point = len(y)
	y_interp = np.array([])
	for idx in range(1, n_point):
		# start_x = idx - 1
		# end_x = idx
		start_y = y[idx - 1]
		end_y = y[idx]
		y_interp = np.hstack((
			y_interp,
			np.linspace(start_y, end_y, factor, endpoint=False)
		))
	y_interp = np.asarray(y_interp)
	return y_interp

def plot_score_vs_accuracy_flipped(ax, set_size, n_mismatch_array, rho_array, c):
	""""""
	accuracy_array = (set_size - n_mismatch_array) / set_size

	mismatch_list = np.unique(n_mismatch_array)
	mismatch_list = mismatch_list[1:]
	n_val = len(mismatch_list)

	loc = np.equal(n_mismatch_array, 0)
	rho_correct = rho_array[loc]
	rho_correct = rho_correct[0]

	score_mean = np.zeros(n_val)
	score_std = np.zeros(n_val)
	score_min = np.zeros(n_val)
	score_max = np.zeros(n_val)
	for idx_mismatch, val_mismatch in enumerate(mismatch_list):
		loc = np.equal(n_mismatch_array, val_mismatch)
		score_mean[idx_mismatch] = np.mean(rho_array[loc])
		score_std[idx_mismatch] = np.std(rho_array[loc])
		score_min[idx_mismatch] = np.min(rho_array[loc])
		score_max[idx_mismatch] = np.max(rho_array[loc])

	accuracy = (set_size - mismatch_list) / set_size

	rho, p_val = spearmanr(accuracy_array, rho_array)
	print('rho: {0:.2f} (p={1:.4f})'.format(rho, p_val))

	ax.plot(
		1-accuracy, score_mean, color=c,
		# marker='o', markersize=.5,
		linestyle='-', linewidth=1,
	)
	ax.fill_between(
		1-accuracy, score_mean - score_std, score_mean + score_std,
		facecolor=c, alpha=.3, edgecolor='none'
	)
	ax.fill_between(
		1-accuracy, score_min, score_max,
		facecolor=c, alpha=.25, edgecolor='none'
	)

	factor = 20
	score_beat_correct = linear_interpolation(score_max, factor=factor)
	accuracy_interp = linear_interpolation(accuracy, factor=factor)
	score_correct = rho_correct * np.ones(len(score_beat_correct))
	locs = np.less(score_beat_correct, score_correct)
	score_beat_correct[locs] = rho_correct
	ax.fill_between(
		1-accuracy_interp, score_correct, score_beat_correct,
		facecolor='r', alpha=.75, edgecolor='none'
	)
	print("rho_correct:",rho_correct)
	ax.scatter(
		0, rho_correct,
		s=100, marker='x',
		color=c
	)

	ax.set_xticks([0., .5, 1.])
	ax.set_xticklabels([0., .5, 1.])

if __name__ == '__main__':
	print("utils_funcs.py")
	# # test aggregate_visual_embeddings function
	# struct=dict(
	#     words=['a','b'],
	#     embeds=dict(
	#         a=dict(
	#             visual=[np.array([1,2,3]),np.array([7,8,9])],
	#             language=[np.array([0,0,0])]
	#         ),
	#         b=dict(
	#             visual=[np.array([100, 200, 300]), np.array([700, 800, 900])],
	#             language=[np.array([-1, -1, -1]),np.array([-3, -3, -3])]
	#         )
	#     )
	# )
	# import pprint
	# pprint.pprint(aggregate_visual_embeddings(struct))

	sample_sentence_from_corpus("pig",1,"noun",corpus_name='wiki_en_subset')
