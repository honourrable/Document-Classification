from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nltk.tokenize.treebank import TreebankWordDetokenizer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from datetime import timedelta
from sklearn.svm import SVC
import itertools
import zeyrek
import nltk
import copy
import time
import os
import re


start_time_total = time.monotonic()
document_number = [0]
all_documents = []
y_true = []


# Adding number label to data and storing all feature names in a string list
def initialize_y_true():
    for i in range(10):
        y_true.append(0)
    for i in range(25):
        y_true.append(1)
    for i in range(35):
        y_true.append(2)
    for i in range(35):
        y_true.append(3)
    for i in range(10):
        y_true.append(4)
    for i in range(20):
        y_true.append(5)
    for i in range(35):
        y_true.append(6)
    for i in range(26):
        y_true.append(7)
    for i in range(11):
        y_true.append(8)

    return


initialize_y_true()
target_names = ['acil saglik', 'cocuk sagligi', 'covid19', 'diyet ve beslenme', 'erkek sagligi', 'estetik ve guzellik',
                'genel saglik', 'kadin sagligi', 'kronik hastaliklar']



# DATA READING
# The function which read a single .txt file
def read_text_file(f_path):
    with open(f_path, encoding="utf8") as file:
        line = file.read().replace("\n", " ")
        line = line.strip()
        file.close()
        all_documents.append(line)
        document_number[0] += 1


# The dunction that starts data reading operation, it runs read_text_file() inside
def get_data():
    path = r"C:\Users\Onur\PycharmProjects\DocumentClassification\data\acil saglik"
    os.chdir(path)

    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            read_text_file(file_path)

    path = r"C:\Users\Onur\PycharmProjects\DocumentClassification\data\cocuk sagligi"
    os.chdir(path)

    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            read_text_file(file_path)

    path = r"C:\Users\Onur\PycharmProjects\DocumentClassification\data\covid19"
    os.chdir(path)

    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            read_text_file(file_path)

    path = r"C:\Users\Onur\PycharmProjects\DocumentClassification\data\diyet ve beslenme"
    os.chdir(path)

    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            read_text_file(file_path)

    path = r"C:\Users\Onur\PycharmProjects\DocumentClassification\data\erkek sagligi"
    os.chdir(path)

    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            read_text_file(file_path)

    path = r"C:\Users\Onur\PycharmProjects\DocumentClassification\data\estetik ve guzellik"
    os.chdir(path)

    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            read_text_file(file_path)

    path = r"C:\Users\Onur\PycharmProjects\DocumentClassification\data\genel saglik"
    os.chdir(path)

    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            read_text_file(file_path)

    path = r"C:\Users\Onur\PycharmProjects\DocumentClassification\data\kadin sagligi"
    os.chdir(path)

    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            read_text_file(file_path)

    path = r"C:\Users\Onur\PycharmProjects\DocumentClassification\data\kronik hastalıklar"
    os.chdir(path)

    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            read_text_file(file_path)

    return


get_data()
print("\nClass number              :", len(target_names))
print("Total document number     :", document_number[0])
all_documents_row = copy.deepcopy(all_documents)



# DATA PREPROCESSING
start_time = time.monotonic()
# This function includes the operations of text preprocessing; convertion to lower case, whitespace removal etc
def preprocessing():
    for idx in range(len(all_documents)):
        turkish_chars_space = " çğıöşü"
        all_documents[idx] = all_documents[idx].lower()
        all_documents[idx] = re.sub(r'[^a-z' + turkish_chars_space + ']', '', all_documents[idx])
        all_documents[idx] = all_documents[idx].strip()

    return

# To remove the duplicate words in documents with root word length limit to avoid undesired situations which may occur
# after removal of punctuations
def remove_duplicate():
    for idx in range(len(all_documents)):
        temp = []
        title_tokens_temp = nltk.word_tokenize(all_documents[idx])
        [temp.append(word) for word in title_tokens_temp if word not in temp]
        all_documents[idx] = TreebankWordDetokenizer().detokenize(temp)

    return

# The function that finds the probable roots of each word which are called lemma and it only gets nouns
def lemmatization():
    analyzer = zeyrek.MorphAnalyzer()
    stop_words = set(stopwords.words('turkish'))
    counter_nouns = 0
    counter_non_nouns = 0

    for idx in range(len(all_documents)):
        tokens = nltk.word_tokenize(all_documents[idx])
        stopped_words_title = [j for j in tokens if j not in stop_words]

        result = ''
        lemma = ''
        for words in stopped_words_title:

            lemmas = analyzer.lemmatize(words)
            try:
                lemma = lemmas[0][1][0]

            except Exception:
                pass

            title_pos = analyzer.analyze(words)
            partitioned_string = str(title_pos[0][0].formatted).partition(' ')
            partitioned_string = partitioned_string[0]

            if partitioned_string.__contains__("Noun"):
                result += lemma + ' '
                counter_nouns += 1
            else:
                counter_non_nouns += 1

            # result += lemma + ' '

        result = result.lower()
        all_documents[idx] = result

    print("Number of non noun words  :", counter_non_nouns)
    print("Number of noun words      :", counter_nouns)

    return

def remove_duplicate_noise():
    for idx in range(len(all_documents)):
        temp = []
        title_tokens_temp = nltk.word_tokenize(all_documents[idx])
        [temp.append(word) for word in title_tokens_temp if word not in temp and 3 <= len(word) <= 20]
        all_documents[idx] = TreebankWordDetokenizer().detokenize(temp)

    return

def print_data():
    print("\n\nDocuments before and after row data preprocessing:\n")
    index = 0
    for (item_row, item_new) in zip(all_documents_row, all_documents):
        print("\n", index + 1, '. document:')
        item_row = ' '.join(item_row.split())
        print("Row:", item_row.strip())
        print("Preprocessed:", item_new)
        index += 1

    return

preprocessing()
remove_duplicate()
lemmatization()
remove_duplicate()
# print_data()

end_time = time.monotonic()
time_preprocessing = timedelta(seconds=end_time - start_time)



# CREATION OF BAG OF WORDS WITH TERM FREQUENCY
bag_of_words = {}

def create_bow():
    for item in all_documents:
        title_tokens = nltk.word_tokenize(item)
        for token in title_tokens:
            if token not in bag_of_words:
                bag_of_words[token] = 1
            else:
                bag_of_words[token] += 1

    return

create_bow()
bag_of_words = {k: v for k, v in sorted(bag_of_words.items(), key=lambda item: item[1], reverse=True)}
bag_of_words = dict(itertools.islice(bag_of_words.items(), 250))



# TRANSFORMATION TO VECTORS
vectors_bow = []

def create_vectors_bow():
    for item in all_documents:
        vector = []
        tokens = nltk.word_tokenize(item)
        for token in bag_of_words:
            if token in tokens:
                vector.append(1)
            else:
                vector.append(0)
        vectors_bow.append(vector)

    print("The length of BOW-vectors :", len(vectors_bow[0]))

    return

def create_vectors_d2v():
    tokenized = []
    for d in all_documents:
        tokenized.append(nltk.word_tokenize(d))
    tagged_data = [TaggedDocument(d, [j]) for j, d in enumerate(tokenized)]

    d2vmodel = Doc2Vec(tagged_data, vector_size=len(tagged_data), window=2, min_count=1, workers=4, epochs=100)
    d2vmodel.save("test_doc2vec.model")
    d2vmodel = Doc2Vec.load("test_doc2vec.model")

    doc2vectors = []
    for idx in range(document_number[0]):
        doc2vectors.append(d2vmodel.dv[idx])
    scaled_vectors = StandardScaler().fit_transform(doc2vectors)

    print("The length of D2V-vectors :", len(scaled_vectors[0]))

    return scaled_vectors

create_vectors_bow()
vectors_d2v = create_vectors_d2v()



# CLASSIFICATION
start_time = time.monotonic()

def neural_network(X_train, X_test, y_train, y_test):
    nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=300)
    nn_classifier.fit(X_train, y_train)
    y_pred = nn_classifier.predict(X_test)
    nn_matrix = confusion_matrix(y_test, y_pred)
    print("\nClassificiation \n\nNeural Network \nConfussion matrix:\n\n", nn_matrix, sep='')
    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return

def decision_tree(X_train, X_test, y_train, y_test):
    dtree_classifier = DecisionTreeClassifier()
    dtree_classifier = dtree_classifier.fit(X_train, y_train)
    y_pred = dtree_classifier.predict(X_test)
    dtree_matrix = confusion_matrix(y_test, y_pred)
    print("Decision Tree \nConfussion matrix:\n\n", dtree_matrix, sep='')
    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return

def random_forest(X_train, X_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    rf_matrix = confusion_matrix(y_test, y_pred)
    print("Random Forest \nConfussion matrix:\n\n", rf_matrix, sep='')
    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return

def logistic_reg(X_train, X_test, y_train, y_test):
    lr_classifier = LogisticRegression(random_state=0, max_iter=300)
    lr_classifier.fit(X_train, y_train)
    y_pred = lr_classifier.predict(X_test)
    lr_matrix = confusion_matrix(y_test, y_pred)
    print("Logistic Regression \nConfussion matrix:\n\n", lr_matrix, sep='')
    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return

def naive_bayes(X_train, X_test, y_train, y_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    mnb_classifier = MultinomialNB()
    mnb_classifier.fit(X_train_scaled, y_train)
    y_pred = mnb_classifier.predict(X_test_scaled)
    mnb_matrix = confusion_matrix(y_test, y_pred)
    print("Multinomial Naive Bayes \nConfussion matrix:\n\n", mnb_matrix, sep='')
    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return

def knn(X_train, X_test, y_train, y_test):
    knn_classifier = KNeighborsClassifier(n_neighbors=len(target_names), metric='minkowski', p=2)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    knn_matrix = confusion_matrix(y_test, y_pred)
    print("K Nearest Neighbours \nConfussion matrix:\n\n", knn_matrix, sep='')
    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return

def svm(X_train, X_test, y_train, y_test):
    svm_classifier = SVC(kernel='linear', random_state=0)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    svm_matrix = confusion_matrix(y_test, y_pred)
    print("\nSupport Vectore Machines \nConfussion matrix:\n\n", svm_matrix, sep='')
    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    return

X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(vectors_bow, y_true, test_size=0.1, random_state=0)

neural_network(X_train_bow, X_test_bow, y_train_bow, y_test_bow)
decision_tree(X_train_bow, X_test_bow, y_train_bow, y_test_bow)
random_forest(X_train_bow, X_test_bow, y_train_bow, y_test_bow)
logistic_reg(X_train_bow, X_test_bow, y_train_bow, y_test_bow)
naive_bayes(X_train_bow, X_test_bow, y_train_bow, y_test_bow)
svm(X_train_bow, X_test_bow, y_train_bow, y_test_bow)
knn(X_train_bow, X_test_bow, y_train_bow, y_test_bow)

X_train_d2v, X_test_d2v, y_train_d2v, y_test_d2v = train_test_split(vectors_d2v, y_true, test_size=0.1, random_state=0)

neural_network(X_train_d2v, X_test_d2v, y_train_d2v, y_test_d2v)
decision_tree(X_train_d2v, X_test_d2v, y_train_d2v, y_test_d2v)
random_forest(X_train_d2v, X_test_d2v, y_train_d2v, y_test_d2v)
logistic_reg(X_train_d2v, X_test_d2v, y_train_d2v, y_test_d2v)
naive_bayes(X_train_d2v, X_test_d2v, y_train_d2v, y_test_d2v)
svm(X_train_d2v, X_test_d2v, y_train_d2v, y_test_d2v)
knn(X_train_d2v, X_test_d2v, y_train_d2v, y_test_d2v)



end_time = time.monotonic()
time_classification = timedelta(seconds=end_time - start_time)

end_time_total = time.monotonic()
time_total = timedelta(seconds=end_time_total - start_time_total)
print("\nData preprocessing time :", time_preprocessing)
print("Classification time     :", time_classification)
print("Execution time          :", time_total)
