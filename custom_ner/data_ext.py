import jsonlines
import json

def tagging(orig,master):
	#print(orig,master)
	final={}
	final['sentence']=orig
	dummy=[]
	for word in orig:
		tags=master[word]
		dummy.append(tags)
	final['tags']=dummy
	return final

def pre_process(text,labels):
	print('text: ',text,' labels: ',labels)
	orig=text.split(" ")
	words=text.split(" ")
	dummy=text.split(" ")
	master={}
	for i in range(len(labels)):
		tag=labels[i]['label']
		token_start=labels[i]['token_start']
		token_end=labels[i]['token_end']
		if token_end>=len(orig):
			pass
		else:
			if (token_start-token_end)!=0:
				indices=[]
				flag=token_start
				while flag<=token_end:
					indices.append(flag)
					flag+=1
				for index in indices:
					words.remove(dummy[index])
					master[dummy[index]]=tag
			else:
				words.remove(dummy[token_start])
				master[dummy[token_start]]=tag

	for word in words:
		master[word]="other"

	final_tags=tagging(orig,master)
	return final_tags

master=[]
with jsonlines.open('ner_usbank.jsonl','r') as reader:
	flag=1
	for obj in reader:
		if obj['answer']=='accept':
			print(obj)
			text=obj['text']
			if 'spans' in obj:
				labels=obj['spans']
				annoted=pre_process(text,labels)
				master.append(annoted)

with open('result_usbank_1.json', 'w') as fp:
		json.dump(master, fp)