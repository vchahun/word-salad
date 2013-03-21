import os
import argparse
import numpy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
import features

MASK = set(['norm-name', 'meta'])

def read_dataset(filename):
    X, y = [], []
    points = list(features.filter(features.extract(filename), MASK))
    for f, v in points:
        X.append(f.todict())
        y.append(v)
    return X, y

def mae(y_pred, y_truth):
    return numpy.average(numpy.abs(numpy.exp(y_pred) - numpy.exp(y_truth)))

def main():
    parser = argparse.ArgumentParser(description='Run regression experiments')
    parser.add_argument('prefix', help='directory which contains {train,dev,test}.json.gz')
    args = parser.parse_args()

    vectorizer = DictVectorizer()

    print('Loading training data...')
    X_train, y_train = read_dataset(os.path.join(args.prefix, 'train.json.gz'))
    X_train = vectorizer.fit_transform(X_train)
    y_train = numpy.log(y_train)

    print('Loading development data...')
    X_dev, y_dev = read_dataset(os.path.join(args.prefix, 'dev.json.gz'))
    X_dev = vectorizer.transform(X_dev)
    y_dev = numpy.log(y_dev)

    print('Training...')
    errors = []
    for penalty in (100, 10, 1, 0.1, 0.01):
        model = Ridge(alpha=penalty)
        print('Penalty: {0}'.format(penalty))
        model.fit(X_train, y_train)
        error = mae(model.predict(X_dev), y_dev)
        errors.append((error, penalty, model))
        print('Dev MAE: {0}'.format(error))

    best_error, best_penalty, best_model = min(errors)

    print('Loading evaluation data...')
    X_test, y_test = read_dataset(os.path.join(args.prefix, 'test.json.gz'))
    X_test = vectorizer.transform(X_test)
    y_test = numpy.log(y_test)

    print('Tuned penalty: {} (MAE={})'.format(best_penalty, best_error))
    print('Test MAE: {0}'.format(mae(best_model.predict(X_test), y_test)))

if __name__ == '__main__':
    main()
