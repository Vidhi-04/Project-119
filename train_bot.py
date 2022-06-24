#Text Data Preprocessing Lib
import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np

words = []
classes = []
word_tags_list = []
ignore_words = ["?", "!", ",", ".", "'"]
train_data_file = open("intents.json").read()
intents = json.loads(train_data_file)
# function for appending stem words
def get_stem_words(words, ignore_words):
        stem_word = []
        for word in words : 
                if words not in ignore_words :
                        w = stemmer.stem(word.lower())
                        stem_word.append(w)
                return stem_word                
for intent in intents["intents"] :
        for pattern in intent["pattern"] :
                pattern_word = nltk.word_tokenize(pattern)
                words.extend(pattern_word)
                word_tags_list.append((pattern_word, intent["tag"]))
        # Add all words of patterns to list
        
        # Add all tags to the classes list
        if intent["tag"] not in classes:
                classes.append(intent["tag"])
                stem_words = get_stem_words(words, ignore_words)
print(stem_words),
print(word_tags_list),
print(classes)                
#Create word corpus for chatbot
def create_bot_corpus(stem_words, classes) :
        stem_words = sorted(list(set(stem_words)))
        classes = sorted(list(set(classes)))
        pickle.dump(stem_words, open("words.pkl", "wb"))
        pickle.dump(classes, open("classes.pkl", "wb"))
        return stem_words, classes
stem_words, classes = create_bot_corpus(stem_words, classes)
print(stem_words)
print(classes)
training_data = []
number_of_tags = len(classes)
labels = [0]*number_of_tags
for word_tags in word_tags_list :
        bag_of_words = []
        pattern_word = word_tags[0]
        for word in pattern_word :
                index = pattern_word.index(word)
                word.stemmer.stem(word.lower())
                pattern_word[index] = words
        for word in stem_words :
                if word in pattern_word:
                        bag_of_words.append(1)
                else:
                        bag_of_words.append(0)
        labels_encoding = list(labels)
        tag = word_tags[1]
        tag_index = classes.index("tag")
        labels_encoding[tag_index] = 1
        training_data.append([bag_of_words, labels_encoding])

def train_preprocess_data(training_data):
        training_data = np.array(training_data, dtype = "object")
        train_x = list(training_data[:,0])
        train_y = list(training_data[:,1])
        return train_x, train_y
train_x, train_y = train_preprocess_data(training_data)

