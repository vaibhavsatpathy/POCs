import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer

def preprocess(text):
    text = str(text.lower())
    # Remove all the special characters
    text = re.sub(r'\W', ' ', text)
    #text = re.sub(r'[^a-zA-Z ]+', '', text)
    # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    #tokenize the texts using nltk
    text = word_tokenize(text)
    #text = [word for word in text if word not in stop_words]
    #Lemmatize the words
    #word_net_lemmatizer = WordNetLemmatizer()
    #text = [word_net_lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    print(text)
    return text

temp=[]
with open('USbank.txt','r') as f:
    for sent in f.readlines():
        final=preprocess(sent)
        temp.append(final)

with open('USbank_updated.txt','w') as f1:
    for sent in temp:
        f1.write(sent)
        f1.write("\n")