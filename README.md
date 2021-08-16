# DocumentClassification

This study is done to create a program which will classify the documents by using some efficient algorithms. Dataset contains 207 documents over 9 categories. Each document is
stored in ".txt" files. The program is created by using Python and its relevant libraries.

## Main Steps

- Data read
- Data preprocessing
  - Removal of punctuation
  - Removal of numbers
  - Removal of duplicate words
  - Removal of whitespaces
  - Lower case
  - Tokenization
  - Lemmatization
- Bag of Words creation
- Vector transformation
  - Bag of Words
  - Document2Vector
- Classification

## System success measurement

The algorithm Multinomial Naive Bayes with Bag of Words method's success rate by using precision, recall and F1-score is shown below. Also confussion matrix is created and
displayed.

![confusson_matrix](https://user-images.githubusercontent.com/57035819/129564203-26384052-539b-49d5-8181-4de5daa82201.png) &emsp; ![success_metrics](https://user-images.githubusercontent.com/57035819/129564173-5aced893-d0d4-468e-93b5-c831f70af1b0.png)

In this study, few methods are implemented and tested. For example, in preprocessing
only noun tpye words are used, others not. All type words method is also used but it didn't perform better, thus only noun form is processed. Also, there is a Bag of Words vector
limit, which is 250. Without limiting the size, maximum size of vector is also run on system but by limiting better results are gained. In addition, in train and test splitting part, the different sizes of train and test data are tried and 0.1 test size performed well.

### Author

- [honourrable](https://github.com/honourrable)
