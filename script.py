from sklearn.linear_model import LogisticRegression
import os
import io
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def main():
    data = DataFrame({'message': [], 'class': []})
    data = data.append(dataFrameFromDirectory('G:/PyCharmProjects/logistic_regression/pos', 'funny'))
    data = data.append(dataFrameFromDirectory('G:/PyCharmProjects/logistic_regression/neg', 'not funny'))

    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(data['message'].values)

    # classifier = LogisticRegression()
    classifier = MultinomialNB()
    targets = data['class'].values
    classifier.fit(counts, targets)

    examples = ['Romania is my country.']
    example_counts = vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)

    print("Result for your sentence. I think that the sentence is: " + str(predictions))


def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                lines.append(line)
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)


if __name__ == "__main__":
    main()
