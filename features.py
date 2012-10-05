from collections import deque
import gzip
import json
import re

def gzip_or_text(filename):
    if filename.endswith('.gz'):
        return gzip.open(filename)
    return open(filename)

class FeatureVector(dict):
    def todict(self):
        def kv():
            for fname, fval in self.iteritems():
                yield ':'.join(fname), fval
        return dict(kv())

META = ('Smoking', 'ByAppointmentOnly', 'HappyHour', 'NoiseLevel', 'OutdoorSeating', 'DriveThru', 'GoodForDancing', 'WiFi', 'RestaurantsTakeOut', 'RestaurantsGoodForGroups', 'AcceptsInsurance', 'RestaurantsTableService', 'CoatCheck', 'WheelchairAccessible', 'Alcohol', 'BusinessAcceptsCreditCards', 'Caters', 'DogsAllowed', 'HasTV', 'RestaurantsAttire', 'RestaurantsDelivery', 'AgesAllowed', 'GoodForKids', 'RestaurantsReservations', 'GoodForMeal', 'Ambience', 'BusinessParking', 'Music')

_stop = set(['and', 'n\'', 'or', 'with', 'w', 'without']
        + ['a', 'the', 'of', 'in', 'on']
        + ['de', 'di', 'o', 'con', 'a', 'la', 'al', 'alla', 's', 'ai', 'e', 'et', 'y']
        + ['l', 'lb', 'oz', 'pt', 'qt', 'pint', 'quart', 'pack', 'pc', 'pcs', 'half']
        + ['lg', 'sm', 'med', 'large', 'small', 'medium'])

wRE = re.compile('[^a-zA-Z\']')
spRE = re.compile('\s+')
def tokenize(name):
    name = wRE.sub(' ', name)
    name = spRE.sub(' ', name).strip()
    name = name.lower()
    return name.split()

""" N-gram features """
def ngrams(tokens):
    ngram = deque(maxlen = 3)
    for token in tokens:
        ngram.append(token)
        yield (1, token)
        if len(ngram) >= 2:
            yield (2, '%s %s' % (ngram[-2], ngram[-1]))
        if len(ngram) == 3:
            yield (3, ' '.join(ngram))

def name(item):
    name = tokenize(item['name'])
    nname = sorted(set(w for w in name if w not in _stop))
    if not nname:
        yield ('norm-name', '?')
    else:
        yield ('norm-name', ' '.join(nname))
        for (c, w) in ngrams(name):
            yield ('name', str(c), w)

def description(item):
    desc = tokenize(item['description'])
    if not desc:
        yield ('desc', '?')
    else:
        for (c, w) in ngrams(desc):
            yield ('desc', str(c), w)

def extract(filename):
    with gzip_or_text(filename) as f:
        for line in f:
            restaurant = json.loads(line)
            base = FeatureVector()
            base['meta', 'city', restaurant['city']] = 1
            base['meta', 'avg_rating'] = restaurant['avg_rating']
            for neighborhood in restaurant['neighborhoods']:
                base['meta', 'neighborhood', neighborhood] = 1
            for category in restaurant['categories']:
                base['meta', 'category', category] = 1
            for info in META:
                if info in restaurant['info']:
                    for item in restaurant['info'][info].split(','):
                        val = item.strip().lower()
                        if val:
                            base['meta', info, val] = 1
            for item in restaurant['items']:
                features = FeatureVector(base)
                for feat in name(item):
                    features[feat] = 1
                for feat in description(item):
                    features[feat] = 1
                yield features, item['price']

def filter(stream, mask):
    for features, response in stream:
        features = FeatureVector((k, v) for k, v in features.iteritems() if k[0] in mask)
        yield features, response
