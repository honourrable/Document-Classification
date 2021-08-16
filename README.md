# DocumentClassification

This study is done to create a program which will classify the documents by using some efficient algorithms. Dataset contains 207 documents over 9 categories. Each document is
stored in ".txt" files.

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

![success_metrics](https://user-images.githubusercontent.com/57035819/129564173-5aced893-d0d4-468e-93b5-c831f70af1b0.png) &nbsp; ![confusson_matrix](https://user-images.githubusercontent.com/57035819/129564203-26384052-539b-49d5-8181-4de5daa82201.png)

