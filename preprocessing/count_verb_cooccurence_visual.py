import csv
import json
import numpy as np
import re
import pickle
from collections import Counter

base_path = '/user_data/yuchenz2/raw_data_verb_alignment/vg/'
all_relationships=json.load(open(base_path+"relationships.json", "rb"))

all_nouns=set()
all_verbs=set()
for image in all_relationships:
	if not image["relationships"]:
		continue
	for relation in image['relationships']:
		if not relation['synsets'] or not relation['object']['synsets'] or not relation['subject']['synsets']:
			continue
		if (relation["synsets"][0]).split(".")[1]=="v":
			all_verbs.add(relation["synsets"][0])
			all_nouns.add(relation['object']['synsets'][0])
			all_nouns.add(relation['subject']['synsets'][0])

all_nouns=list(all_nouns)
all_verbs=list(all_verbs)
co_occur_table=np.zeros((len(all_verbs),len(all_nouns)))

all_nouns_dict=dict()
for i,w in enumerate(all_nouns):
	all_nouns_dict[w]=i
all_verbs_dict=dict()
for i,w in enumerate(all_verbs):
	all_verbs_dict[w]=i

for image in all_relationships:
	if not image["relationships"]:
		continue
	for relation in image['relationships']:
		if not relation['synsets'] or not relation['object']['synsets'] or not relation['subject']['synsets']:
			continue
		if (relation["synsets"][0]).split(".")[1]=="v":
			obj_idx=all_nouns_dict[relation['object']['synsets'][0]]
			sub_idx=all_nouns_dict[relation['subject']['synsets'][0]]
			pred_idx=all_verbs_dict[relation["synsets"][0]]
			# co_occur_table[pred_idx][obj_idx]+=1
			co_occur_table[pred_idx][sub_idx]+=1

ret=dict(
	nouns=all_nouns_dict,
	verbs=all_verbs_dict,
	table=co_occur_table)

pickle.dump(ret,open("../data/processed/dumped/verb_cooccurence_visual.pkl","wb"))
