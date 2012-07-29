from itertools import izip
import math
import creg
import features

PREFIX='/path/to/data'
MASK = set(['norm-name', 'meta'])

def dataset(filename):
    points = features.filter(features.extract(filename), MASK)
    return creg.RealvaluedDataset((f.todict(), math.log(v)) for f, v in points)

def mae(model, data):
    pred = model.predict(data)
    return sum(abs(math.exp(p) - math.exp(r)) for ((_, p), r) in izip(data, pred))/len(data)

print('Loading training data...')
train_data = dataset(PREFIX+'/train.json.gz')

print('Loading development data...')
dev_data = dataset(PREFIX+'/dev.json.gz')

print('Training...')
errors = []
model = creg.LinearRegression()
for penalty in (100, 10, 1, 0.1, 0.01, 1e-3, 1e-4):
    print('Penalty: {0}'.format(penalty))
    model.fit(train_data, l1=penalty, delta=1e-9)
    error = mae(model, dev_data)
    errors.append((error, penalty))
    print(model.weights)
    print('Dev MAE: {0}'.format(error))

_, best_penalty = min(errors)
final_model = creg.LinearRegression()
final_model.fit(train_data, l1=best_penalty, delta=1e-9)

print('Loading evaluation data...')
test_data = dataset(PREFIX+'/test.json.gz')

print('Test MAE: {0}'.format(mae(model, test_data)))
