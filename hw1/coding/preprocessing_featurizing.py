import os
import re
import copy
import json

from collections import defaultdict

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# %%
class PreProcessing(object):
    def __init__(self, file):
        self.file = file

    def read(self):
        with open(self.file, "r", encoding='latin1') as f:
            data = f.read()
        return data

    def word_stemming(self):
        ps = PorterStemmer()

        sentence = self.read()
        words = word_tokenize(sentence)

        # Remove all non-word characters and punctuations
        for i in range(len(words)):
            words[i] = re.sub(r'\W', '', words[i])
            words[i] = re.sub(r'\s+', '', words[i])

        # Remove empty string in the list
        words = list(filter(None, words))

        # Stemming
        for i in range(len(words)):
            words[i] = ps.stem(words[i], to_lowercase=True)
        return words

    def bag_of_words(self, word2count: dict):
        words = self.word_stemming()
        for word in words:
            word2count[word] += 1
        return word2count


def get_most_frequent(low_frequency, high_frequency, word2count):
    most_freq = dict(
        filter(lambda elem: high_frequency >= elem[1] >= low_frequency,
               word2count.items()))
    sorted_most_freq = dict(sorted(most_freq.items(), key=lambda x: x[1],
                                   reverse=True))
    return sorted_most_freq


class FeatureGenerator(object):
    def __init__(self, word2count):
        self.word2count = word2count

    def sort_dict(self):
        return dict(sorted(self.word2count.items()))

    def featurize_email(self, file_path):
        words = PreProcessing(file_path).word_stemming()
        sorted_word2count = self.sort_dict()
        features = copy.deepcopy(sorted_word2count)
        features = dict.fromkeys(features, 0)
        for word in words:
            if word in features:
                features[word] += 1
        return list(features.values())


# %%
path = os.path.abspath('../')
path = path + '/enron1'
email_path = []
for folder in os.listdir(path):
    if 'spam' in folder or 'ham' in folder:
        email_path.append(os.path.join(path, folder))

word2count = defaultdict(int)
for folder_path in email_path:
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            pp = PreProcessing(file_path)
            pp.bag_of_words(word2count)

#%% Get most frequent words to reduce the complexity
most_freq = get_most_frequent(50, 2000, word2count)
print(most_freq)

# %%
featurized_emails = {}
for folder_path in email_path:
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            fg = FeatureGenerator(most_freq)
            features = fg.featurize_email(file_path)
            featurized_emails[file] = features

# %%
with open("featurized_emails_10_2000.json", "w") as outfile:
    json.dump(featurized_emails, outfile)
