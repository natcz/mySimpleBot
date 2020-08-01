import tensorflow as tf
import numpy as np
import json
import tflearn
import random
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
nltk.download('punkt')

file = open('intents.json')  # opening file and getting all the data out of it
data = json.load(file)

words = []
tok_words = []  # list of lists tokenized words
tok_words_tags = []
labels = []  # tags

# preprocessing the data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokenized_wrd = nltk.word_tokenize(
            pattern)  # list of tokenized words "string of words" -> ["string","of", "words"]
        words.extend(tokenized_wrd)  # appends all the words to 'words' list
        tok_words.append(tokenized_wrd)
        tok_words_tags.append(intent["tag"])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = list(set([stemmer.stem(w.lower()) for w in words if w != "?"]))  # stem() gets  word’s stem based on ARLSTem

training = []
output = []

for x, doc in enumerate(tok_words):
    bag = []

    tokenized_wrd = [stemmer.stem(w.lower()) for w in doc]

    for w in words:             #0-not in set  , 1- in set
        if w in tokenized_wrd:
            bag.append(1)
        else:
            bag.append(0)

    output_row = [0 for _ in range(len(labels))]
    output_row[labels.index(tok_words_tags[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

with open("data.pickle", "wb") as f:  #change dictionary into a pickle file
    pickle.dump((words, labels, training, output), f)

tf.reset_default_graph()  #Clear the default graph stack and reset the global default graph.

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 7)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=5000, batch_size=7, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=5000, batch_size=7, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    b = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                b[i] = 1

    return np.array(b)


def chat():
    print("Say hi to me!  (if you want to quit, please input 'Q'")
    name = input("Type your name:")
    while True:
        inp = input(str(name)+": ")
        if inp.lower() == "q":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))


chat()
