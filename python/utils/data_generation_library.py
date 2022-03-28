from nltk.stem import WordNetLemmatizer
import numpy as np
import os
import sys
#sys.path.append('..')
import random
import pickle
import torch.optim
import torch.nn.parallel
from PIL import Image
import cv2
import json
import shutil
import pprint

#print(os.getcwd())

from .utils_funcs import load_pretrained_Glove, call_pretrained_BERT, call_fintuned_BERT_sentence, sample_sentence_from_corpus

from .swav_utils import resnet50 as resnet_models
from .mit_video_utils import video_models
from .mit_video_utils.video_utils import extract_frames

# Lemmatize verbs in MiT dataset from present participle to simple present
def lemmatize(words):
    L_words = list()
    L = WordNetLemmatizer()
    for word in words:
        if "singing" in word:
            word='singing'
        elif "speaking" in word:
            word="speaking"
        elif "playing" in word:
            word="playing"
        L_word = L.lemmatize(word, "v")
        L_words.append(L_word)
    L_dict=dict()
    for i,L_word in enumerate(L_words):
        if L_word in L_dict.keys():
            L_dict[L_word].append(words[i])
        else:
            L_dict[L_word]=[words[i]]
    return L_dict

### vision encoders

def swav_encoder():
    return 0

def openimage_encoder():
    return 0

def vrd_noun_encoder():
    return 0

def vrd_verb_encoder():
    return 0

def vg_noun_encoder(concepts,n_sample,name,cuda=False):
    print("Loading swav model ...")
    model = resnet_models.__dict__['resnet50'](
        normalize=True,
        hidden_mlp=2048,
        output_dim=128,
        nmb_prototypes=3000,
    )
    if cuda:
        model=model.cuda()
        cc = torch.load('../data/pretrained_models/swav_800ep_pretrain.pth.tar')
    else:
        cc = torch.load('../data/pretrained_models/swav_800ep_pretrain.pth.tar', map_location="cpu")
    
    model_state_dict = {}
    for key in cc:
        model_state_dict[key[7:]] = cc[key]
        # print(key[7:],key)
    model.load_state_dict(model_state_dict)

    model.eval()
    transform = video_models.load_transform()
    print("Finish loading swav model.")

    # Please modify the base path to your data directory
    base_path = '/user_data/yuchenz2/raw_data_verb_alignment/vg/'

    #dump_path=base_path + "vg_noun_" + str(n_sample) + "_cropped_image/"
    dump_path = base_path + name +"_cropped_image/"
    if os.path.isdir(dump_path):
        shutil.rmtree(dump_path)
    os.makedirs(dump_path)
    print("cropped images are dumped to:",dump_path)

    print("Start loading relationships.json...")
    annotations = json.load(open(base_path+"relationships.json", "rb"))
    print("Finish!")

    shuffled_relations=list()
    for image_annotation in annotations:
        image_id=str(image_annotation["image_id"])+".jpg"
        if not image_annotation["relationships"]:
            continue
        for relation in image_annotation["relationships"]:
            shuffled_relations.append((image_id,relation))

    image_embeddings_dict=dict()
    for i,target_obj in enumerate(concepts):
        print("Start sampling concept", target_obj, '('+str(i+1)+'/'+str(len(concepts))+')'+"...")
        random.shuffle(shuffled_relations)
        i_image=0
        embedding_list=list()
        for (image_id,relation) in shuffled_relations:
            if not relation['object']['synsets'] or not relation['subject']['synsets']:
                continue
            object=relation['object']['synsets'][0]
            subject=relation['subject']['synsets'][0]
            # flag denotes whether we find the corresponding object in this loop
            flag=0
            if object==target_obj:
                whole_path1 = base_path+'VG_100K/' + image_id
                whole_path2 = base_path+'VG_100K_2/' + image_id
                if os.path.isfile(whole_path1):
                    img = cv2.imread(whole_path1)
                elif os.path.isfile(whole_path2):
                    img = cv2.imread(whole_path2)
                else:
                    print("error! didn't find img:",image_id)

                h=int(relation['object']['h'])
                w=int(relation['object']['w'])
                x=int(relation['object']['x'])
                y=int(relation['object']['y'])
                crop_img = img[y:y+h, x:x+w]
                save_dir=dump_path + target_obj + "/"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                savepath =  save_dir + target_obj + "_" + str(i_image) + '.jpg'
                cv2.imwrite(savepath, crop_img)
                i_image+=1
                flag=1
            elif subject==target_obj:
                whole_path1 = base_path+'VG_100K/' + image_id
                whole_path2 = base_path+'VG_100K_2/' + image_id
                if os.path.isfile(whole_path1):
                    img = cv2.imread(whole_path1)
                elif os.path.isfile(whole_path2):
                    img = cv2.imread(whole_path2)
                else:
                    print("error! didn't find img:", image_id)

                h = int(relation['subject']['h'])
                w = int(relation['subject']['w'])
                x = int(relation['subject']['x'])
                y = int(relation['subject']['y'])
                crop_img = img[y:y + h, x:x + w]
                #cv2.imshow("cropped", crop_img)
                #cv2.waitKey(0)
                save_dir=dump_path + target_obj + "/"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                savepath =  save_dir + target_obj + "_" + str(i_image) + '.jpg'
                cv2.imwrite(savepath, crop_img)
                i_image+=1
                flag=1
                # savepath = base_path + "temp/" + target_obj + '.jpg'
                # cv2.imwrite(savepath, crop_img)
            if flag:
                PIL_image=Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))
                if cuda:
                    transformed_image = transform(PIL_image).cuda()
                else:
                    transformed_image = transform(PIL_image)
                with torch.no_grad():
                    transformed_image=torch.unsqueeze(transformed_image,0)
                    embedding, _ = model(transformed_image)
                if cuda:
                    embedding_list.append(torch.squeeze(embedding).data.cpu().numpy())
                else:
                    embedding_list.append(torch.squeeze(embedding).numpy())
            if i_image==n_sample:
                break
        print("length of embedding_list:",target_obj,len(embedding_list))
        image_embeddings_dict[target_obj.split(".")[0]]=embedding_list

    print("Finish processing images.")
    return image_embeddings_dict

def vg_verb_encoder(concepts,n_sample,name,cuda=False):
    print("Loading swav model ...")
    model = resnet_models.__dict__['resnet50'](
        normalize=True,
        hidden_mlp=2048,
        output_dim=128,
        nmb_prototypes=3000,
    )
    if cuda:
        model=model.cuda()
        cc = torch.load('../data/pretrained_models/swav_800ep_pretrain.pth.tar')
    else:
        cc = torch.load('../data/pretrained_models/swav_800ep_pretrain.pth.tar', map_location="cpu")
    
    model_state_dict = {}
    for key in cc:
        model_state_dict[key[7:]] = cc[key]
        # print(key[7:],key)
    model.load_state_dict(model_state_dict)

    model.eval()
    transform = video_models.load_transform()
    print("Finish loading swav model.")

    # Please modify the base path to your data directory
    base_path = '/user_data/yuchenz2/raw_data_verb_alignment/vg/'

    #dump_path=base_path + "vg_verb_" + str(n_sample) + "_cropped_image/"
    dump_path = base_path + name + "_cropped_image/"
    if os.path.isdir(dump_path):
        shutil.rmtree(dump_path)
    os.makedirs(dump_path)
    print("cropped images are dumped to:",dump_path)

    print("Start loading relationships.json...")
    annotations = json.load(open(base_path+"relationships.json", "rb"))
    print("Finish!")

    shuffled_relations=list()
    for image_annotation in annotations:
        image_id=str(image_annotation["image_id"])+".jpg"
        if not image_annotation["relationships"]:
            continue
        for relation in image_annotation["relationships"]:
            shuffled_relations.append((image_id,relation))

    image_embeddings_dict=dict()
    for i,target_verb in enumerate(concepts):
        print("Start sampling concept", target_verb, '('+str(i+1)+'/'+str(len(concepts))+')'+"...")
        random.shuffle(shuffled_relations)
        i_image=0
        embedding_list=list()
        for (image_id,relation) in shuffled_relations:
            if not relation['synsets'] or not relation['object']['synsets'] or not relation['subject']['synsets']:
                continue
            verb=relation['synsets'][0]
            if verb==target_verb:
                whole_path1 = base_path+'VG_100K/' + image_id
                whole_path2 = base_path+'VG_100K_2/' + image_id
                if os.path.isfile(whole_path1):
                    img = cv2.imread(whole_path1)
                elif os.path.isfile(whole_path2):
                    img = cv2.imread(whole_path2)
                else:
                    print("error! didn't find img:",image_id)

                h1=int(relation['object']['h'])
                w1=int(relation['object']['w'])
                x1=int(relation['object']['x'])
                y1=int(relation['object']['y'])
                h2 = int(relation['subject']['h'])
                w2 = int(relation['subject']['w'])
                x2 = int(relation['subject']['x'])
                y2 = int(relation['subject']['y'])
                crop_img = img[min(y1,y2):max(y1+h1,y2+h2), min(x1,x2):max(x1+w1,x2+w2)]
                #cv2.imshow("cropped", crop_img)
                #cv2.waitKey(0)
                # savepath = base_path + "temp/" + target_verb + '.jpg'
                # cv2.imwrite(savepath, crop_img)
                save_dir=dump_path + target_verb + "/"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                savepath =  save_dir + target_verb + "_" + str(i_image) + '.jpg'
                cv2.imwrite(savepath, crop_img)
                i_image+=1
                PIL_image=Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))
                if cuda:
                    transformed_image = transform(PIL_image).cuda()
                else:
                    transformed_image = transform(PIL_image)
                with torch.no_grad():
                    transformed_image=torch.unsqueeze(transformed_image,0)
                    embedding, _ = model(transformed_image)
                if cuda:
                    embedding_list.append(torch.squeeze(embedding).data.cpu().numpy())
                else:
                    embedding_list.append(torch.squeeze(embedding).numpy())
            if i_image==n_sample:
                break
        print("length of embedding_list:",target_verb,len(embedding_list))        
        image_embeddings_dict[target_verb.split(".")[0]]=embedding_list

    print("Finish processing images.")
    return image_embeddings_dict

def process_mit_video(words,model_name,n_sample,cuda=False):
    print("Loading video model "+model_name+"...")
    if model_name=='resnet3d50':
        model = video_models.load_model('resnet3d50')
    elif model_name=='swav':
        model = resnet_models.__dict__['resnet50'](
            normalize=True,
            hidden_mlp=2048,
            output_dim=128,
            nmb_prototypes=3000,
        )
        if cuda:
            model=model.cuda()
            cc = torch.load('../data/pretrained_models/swav_800ep_pretrain.pth.tar')
        else:
            cc = torch.load('../data/pretrained_models/swav_800ep_pretrain.pth.tar', map_location="cpu")
        
        model_state_dict = {}
        for key in cc:
            model_state_dict[key[7:]] = cc[key]
            # print(key[7:],key)
        model.load_state_dict(model_state_dict)
    else:
        print('unrecognized video model, select resnet3d50 as default.')
        model = video_models.load_model('resnet3d50')
    model.eval()
    transform = video_models.load_transform()
    print("Finish loading video model.")

    # Please modify the base path to your data directory
    base_path = '/user_data/yuchenz2/raw_data_verb_alignment/mit/Moments_in_Time_Raw/training/'
    video_embeddings_dict=dict()
    L_dict = lemmatize(words)
    for i,word in enumerate(L_dict.keys()):
        print("Start sampling word", word, '('+str(i+1)+'/'+str(len(list(L_dict.keys())))+')'+"...")
        relative_directories=L_dict[word]
        absolute_directoties=[''.join([base_path, r_path, "/"]) for r_path in relative_directories]
        file_names=list()
        file_flag=False
        for a_dir in absolute_directoties:
            files = os.listdir(a_dir)
            if not files:
                print(a_dir, "is an empty directory")
            else:
                for x in files:
                    if x.endswith('mp4'):
                        file_names.append(''.join([a_dir,x]))
                        file_flag = True
        if not file_flag:
            print("Error! For word",word,", cannot find any video in", absolute_directoties)
            sys.exit(1)
        # flag and i_video variables jointly control the number of sample videos to meet the required number
        flag = False
        i_video = 0
        embedding_list = list()
        while not flag:
            selected_video = random.sample(file_names, k=1)[0]
            try:
                frames = extract_frames(selected_video, 16)
                #print('extract_frames_complete.')
                if model_name == 'resnet3d50':
                    transformed_frame = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)
                    with torch.no_grad():
                        embedding = model(transformed_frame)
                elif model_name == 'swav':
                    if cuda:
                        transformed_frame = torch.stack([transform(frame).cuda() for frame in frames], 0)
                    else:
                        transformed_frame = torch.stack([transform(frame) for frame in frames], 0)
                    with torch.no_grad():
                        embedding, _ = model(transformed_frame) #embedding of shape [16,3,224,224]
                    embedding = torch.mean(embedding, 0).unsqueeze(0)
                else:
                    # default: resnet3d50
                    transformed_frame = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)
                    with torch.no_grad():
                        embedding = model(transformed_frame)
                i_video += 1
                if cuda:
                    embedding_list.append(torch.squeeze(embedding).data.cpu().numpy())
                else:
                    embedding_list.append(torch.squeeze(embedding).numpy())
                if i_video == n_sample:
                    flag = True
            except:
                print("have problem processing",selected_video, "try other videos as alternative")
        video_embeddings_dict[word] = embedding_list

    print("Finish processing videos.")
    return video_embeddings_dict

def mit_resnet3d50_encoder(concepts,n_sample,name,cuda=False):
    visual_embeddings_dict=process_mit_video(concepts,'resnet3d50',n_sample,cuda)
    return visual_embeddings_dict

def mit_swav_encoder(concepts,n_sample,name,cuda=False):
    visual_embeddings_dict=process_mit_video(concepts,'swav',n_sample,cuda)
    return visual_embeddings_dict

### language encoders

def glove_pretrained_encoder(concepts,n_sample,cuda):
    language_embeddings_dict=dict()
    pretrained_glove=load_pretrained_Glove()
    for word in concepts:
        # Just store one language embedding per concept because pretrained model just has one
        language_embeddings_dict[word]=[pretrained_glove[word]]
    print("Finish processing language embeddings.")
    return language_embeddings_dict

def glove_pretrained_encoder_vg(concepts,n_sample,cuda):
    language_embeddings_dict=dict()
    pretrained_glove=load_pretrained_Glove()
    # drop wordnet annotation suffixes in concepts
    concepts=[concept.split(".")[0] for concept in concepts]
    for word in concepts:
        # Just store one language embedding per concept because pretrained model just has one
        language_embeddings_dict[word]=[pretrained_glove[word]]
    print("Finish processing language embeddings.")
    return language_embeddings_dict

def glove_pretrained_encoder_mit(concepts,n_sample,cuda):
    language_embeddings_dict=dict()
    pretrained_glove=load_pretrained_Glove()
    # transform concepts from present participle to simple present
    concepts=list(lemmatize(concepts).keys())
    for word in concepts:
        # Just store one language embedding per concept because pretrained model just has one
        language_embeddings_dict[word]=[pretrained_glove[word]]
    print("Finish processing language embeddings.")
    return language_embeddings_dict

def bert_pretrained_encoder_vg(concepts,n_sample,cuda):
    pretrained_BERT=call_pretrained_BERT
    # drop wordnet annotation suffixes in concepts
    concepts=[concept.split(".")[0] for concept in concepts]
    # sample n_sample of contexts for each word
    packed_samples=sample_sentence_from_corpus(concepts,n_sample)
    # generate contextualized embeddings
    language_embeddings_dict=pretrained_BERT(concepts,packed_samples,cuda)
    print("Finish processing language embeddings.")
    return language_embeddings_dict

def bert_pretrained_encoder_vg_mask(concepts,n_sample,cuda):
    pretrained_BERT=call_pretrained_BERT
    # drop wordnet annotation suffixes in concepts
    concepts=[concept.split(".")[0] for concept in concepts]
    # sample n_sample of contexts for each word
    packed_samples=sample_sentence_from_corpus(concepts,n_sample)
    # generate contextualized embeddings
    language_embeddings_dict=pretrained_BERT(concepts,packed_samples,cuda,mask=True)
    print("Finish processing language embeddings.")
    return language_embeddings_dict

def bert_finetuned_encoder_vg_mask(concepts,n_sample,dataset,cuda,finetune_path,packed_samples):
    if "verb" in dataset:
        pos="verb"
    else:
        pos="noun"
    fintuned_BERT=call_fintuned_BERT_sentence
    # drop wordnet annotation suffixes in concepts
    concepts=[concept.split(".")[0] for concept in concepts]
    # sample n_sample of contexts for each word
    #packed_samples=sample_sentence_from_corpus(concepts,n_sample,'verb','wiki_en')
    packed_samples=pickle.load(open(packed_samples,"rb"))
    # generate contextualized embeddings
    language_embeddings_dict=fintuned_BERT(concepts,packed_samples,finetune_path,pos,cuda=cuda)
    print("Finish processing language embeddings.")
    return language_embeddings_dict

def bert_pretrained_encoder_mit(concepts,n_sample,cuda):
    pretrained_BERT=call_pretrained_BERT
    # transform concepts from present participle to simple present
    concepts=list(lemmatize(concepts).keys())
    # sample n_sample of contexts for each word
    packed_samples=sample_sentence_from_corpus(concepts,n_sample)
    # generate contextualized embeddings
    language_embeddings_dict=pretrained_BERT(concepts,packed_samples,cuda)
    print("Finish processing language embeddings.")
    return language_embeddings_dict

if __name__ == '__main__':
    print("data_generation_library.py")
    # extract_frames(1)
    # print(lemmatize(['eating','playing+videogames','playing+the+violin']))
    bert_pretrained_encoder_mit(['eat','find'],2,False)
