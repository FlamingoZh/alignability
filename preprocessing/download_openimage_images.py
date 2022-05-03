import pickle
import csv

from subprocess import Popen, PIPE
import OIDv4_ToolKit

def load_concept(concept_file):
	words = list()
	with open("../data/concepts/"+concept_file, 'r', encoding="utf-8") as f:
		for line in f:
			word = str(line).replace('\n','').replace('\r','')
			words.append(word)
	return words

concepts=load_concept("openimage_concept.txt")
base_path="/user_data/yuchenz2/raw_data_verb_alignment/openimage/"

openimage_concepts=list()
with open('OIDv4_ToolKit/OID/csv_folder/class-descriptions-boxable.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		openimage_concepts.append(row[1])
#print(openimage_concepts)

intersect_concepts=list()
for i in concepts:
	if i in openimage_concepts:
		intersect_concepts.append(i)
	else:
		print(i," not in openimage concept list.")

## python main.py downloader --classes ../../data/concepts/openimage_concept.txt --type_csv validation --limit 1
	

