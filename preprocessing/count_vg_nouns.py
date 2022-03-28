import json
import re
import sys
from collections import Counter

base_path = '/user_data/yuchenz2/raw_data_verb_alignment/vg/'
all_relationships=json.load(open(base_path+"relationships.json", "rb"))

nouns_syn=list()

for image in all_relationships:
    if not image["relationships"]:
        continue
    for relations in image['relationships']:
        if not relations['synsets'] or not relations['object']['synsets'] or not relations['subject']['synsets']:
            continue
        if relations['object']['synsets']:
            temp = relations['object']['synsets'][0]
            temp2 = temp.split(".")
            word = temp2[0]
            pos = temp2[1]
            sense = temp2[2]
            if pos == "n" and sense == "01" and ("_" not in word) and ("-" not in word):
                nouns_syn.append(temp)
        if relations['subject']['synsets']:
            temp = relations['subject']['synsets'][0]
            temp2 = temp.split(".")
            word = temp2[0]
            pos = temp2[1]
            sense = temp2[2]
            if pos == "n" and sense == "01" and ("_" not in word) and ("-" not in word):
                nouns_syn.append(temp)

sorted_dict = sorted(dict(Counter(nouns_syn)).items(), key=lambda x: x[1], reverse=True)

with open('../data/concepts/vg_noun_count_all.txt',"w") as f:
    for i in sorted_dict:
        f.write(str(i[0])+" "+str(i[1])+"\n")

with open('../data/concepts/vg_noun_concept_least20.txt',"w") as f:
    for i in sorted_dict[:210]:
        f.write(str(i[0])+"\n")

with open('../data/concepts/vg_noun_concept_least10.txt',"w") as f:
    for i in sorted_dict[:309]:
        f.write(str(i[0])+"\n")