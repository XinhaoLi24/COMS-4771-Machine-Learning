import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from classifier import NaiveBayesClassifier, KNN

rcparams = {"font.size": 14,
            "legend.frameon": False,
            "xtick.top": True,
            "xtick.direction": "in",
            "xtick.minor.visible": False,
            "xtick.major.size": 10,
            "xtick.minor.size": 10,
            "ytick.right": True,
            "ytick.direction": "in",
            "ytick.minor.visible": False,
            "ytick.major.size": 10,
            "ytick.minor.size": 10}

#%%
df = pd.read_json('./featurized_emails_most_50.json')

df = df.transpose()
df = df.reset_index(drop=False, inplace=False)
df.rename(columns={'index': 'file'}, inplace=True)
# df = df.sample(frac=0.01, random_state=1)


def convert_spam_label(row):
    if 'spam' in row['file']:
        return 1
    else:
        return 0


df['spam'] = df.apply(lambda row: convert_spam_label(row), axis=1)



#%%
fracs = [1, 2, 5, 10, 15, 20]
# accuracy_nbc = []
accuracy_knn = []
for split in fracs:
    train = df.sample(frac=0.75, random_state=1000)
    test = df.drop(train.index)

    X_train = train.drop(['spam', 'file'], axis=1)
    y_train = pd.DataFrame(train['spam'], columns=['spam'])

    X_test = test.drop(['spam', 'file'], axis=1)
    y_test = pd.DataFrame(test['spam'], columns=['spam'])

    # nbc = NaiveBayesClassifier()
    # nbc.fit(X_train, y_train)
    # y_test_predict = nbc.predict(X_test)
    # # y_test = y_test['spam'].values.tolist()
    # accuracy_nbc.append(accuracy_score(y_test_predict, y_test))

    knn_1 = KNN(k=split, distance_metric='euclidean')
    knn_1.fit(X_train, y_train)

    y_test_predict = knn_1.predict(X_test)
    # y_test = y_test['spam'].values.tolist()
    accuracy_knn.append(accuracy_score(y_test_predict, y_test))


#%% Plot
# print(accuracy_nbc)
print(accuracy_knn)

plt.rcParams.update(rcparams)
# plt.plot(fracs, accuracy_nbc, label='Naive Bayes')
plt.plot(fracs, accuracy_knn)
plt.scatter(fracs, accuracy_knn, linewidths=8)
plt.xlabel('Number NN')
plt.ylabel('Accuracy for the test set')
plt.legend()
plt.tight_layout()
plt.savefig('number-NN-compare.pdf', dpi=300)
plt.show()


#%%
files = ['./featurized_emails_most_50.json',
         './featurized_emails_10_2000.json',
         './featurized_emails_10_5000.json']
accuracy_nbc = []
accuracy_knn = []
for file in files:

    df = pd.read_json(file)

    df = df.transpose()
    df = df.reset_index(drop=False, inplace=False)
    df.rename(columns={'index': 'file'}, inplace=True)

    # df = df.sample(frac=0.01, random_state=1)

    def convert_spam_label(row):
        if 'spam' in row['file']:
            return 1
        else:
            return 0


    df['spam'] = df.apply(lambda row: convert_spam_label(row), axis=1)

    train = df.sample(frac=0.75, random_state=1000)
    test = df.drop(train.index)

    X_train = train.drop(['spam', 'file'], axis=1)
    y_train = pd.DataFrame(train['spam'], columns=['spam'])

    X_test = test.drop(['spam', 'file'], axis=1)
    y_test = pd.DataFrame(test['spam'], columns=['spam'])

    nbc = NaiveBayesClassifier()
    nbc.fit(X_train, y_train)
    y_test_predict = nbc.predict(X_test)
    # y_test = y_test['spam'].values.tolist()
    accuracy_nbc.append(accuracy_score(y_test_predict, y_test))

    knn_2 = KNN(k=2, distance_metric='euclidean')
    knn_2.fit(X_train, y_train)

    y_test_predict = knn_2.predict(X_test)
    # y_test = y_test['spam'].values.tolist()
    accuracy_knn.append(accuracy_score(y_test_predict, y_test))

#%%
barWidth = 1
# plt.plot(fracs, accuracy_nbc, label='Naive Bayes')
x = [">50", "10-2000", "10-5000"]
br1 = np.arange(len(x))*4
# br1 = [10, 20, 30]
br2 = [x+barWidth for x in br1]
plt.bar(br1, accuracy_nbc, label='Naive Bayes')
plt.bar(br2, accuracy_knn, label='KNN')
# plt.plot(x, accuracy_knn)
# plt.scatter(fracs, accuracy_knn, linewidths=8)
plt.xlabel('Words frequency')
plt.ylabel('Accuracy for the test set')
plt.xticks([r*4 + barWidth for r in range(len(x))],
        [">50", "10-2000", "10-5000"])
plt.legend()
plt.tight_layout()
plt.ylim(ymin=0.8, ymax=0.95)
plt.savefig('diff-set-compare.pdf', dpi=300)
plt.show()


#%%
dists = ['euclidean', 'manhattan', 'max_distance']
# accuracy_nbc = []
accuracy_knn = []
for dist in dists:
    train = df.sample(frac=0.75, random_state=1000)
    test = df.drop(train.index)

    X_train = train.drop(['spam', 'file'], axis=1)
    y_train = pd.DataFrame(train['spam'], columns=['spam'])

    X_test = test.drop(['spam', 'file'], axis=1)
    y_test = pd.DataFrame(test['spam'], columns=['spam'])

    # nbc = NaiveBayesClassifier()
    # nbc.fit(X_train, y_train)
    # y_test_predict = nbc.predict(X_test)
    # # y_test = y_test['spam'].values.tolist()
    # accuracy_nbc.append(accuracy_score(y_test_predict, y_test))

    knn_1 = KNN(k=2, distance_metric=dist)
    knn_1.fit(X_train, y_train)

    y_test_predict = knn_1.predict(X_test)
    # y_test = y_test['spam'].values.tolist()
    accuracy_knn.append(accuracy_score(y_test_predict, y_test))


#%% Plot

barWidth = 1
# plt.plot(fracs, accuracy_nbc, label='Naive Bayes')
# x = ["", "10-2000", "10-5000"]
br1 = np.arange(len(dists))*2
# br1 = [10, 20, 30]
br2 = [x+barWidth for x in br1]
# plt.bar(br1, accuracy_nbc, label='Naive Bayes')
plt.bar(br2, accuracy_knn, label='KNN')
# plt.plot(x, accuracy_knn)
# plt.scatter(fracs, accuracy_knn, linewidths=8)
plt.xlabel('Different distance metrics')
plt.ylabel('Accuracy for the test set')
plt.xticks([r*2 + barWidth for r in range(len(dists))],
        dists)
plt.legend()
plt.ylim(ymin=0.75, ymax=0.95)
plt.tight_layout()

plt.savefig('diff-metrics-compare.pdf', dpi=300)
plt.show()
