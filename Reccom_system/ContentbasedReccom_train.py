import sys
import time
import json
import re
from math import log2
from pyspark import SparkContext


def formatting(text: str, stopwords: list):
    text = re.sub('[^a-zA-Z ]+', ' ', text)  # replace non-letter characters by space
    text = re.sub(' +', ' ', text)  # convert multiple spaces to one single white space
    text = text.lower().split(' ')
    formatted = [w for w in text if w not in stopwords and w != '']  # remove stopwords
    return formatted


def tf(words: list):
    temp = dict()
    for w in words:
        if w not in temp.keys():
            temp[w] = 1
        else:
            temp[w] += 1  # fij in TF
    temp = {k: v for (k, v) in temp.items() if v > 4}  # remove rare words
    max_word_num = max(list(temp.values()), default=1)  # max(k)fkj in TF
    for w in temp.keys():
        temp[w] = temp[w] / max_word_num
    return temp


def word_doc_idf(word: list, N: int):
    temp = dict()
    for w in word:
        if w not in temp.keys():
            temp[w] = 1
        else:
            temp[w] += 1
    for k in temp.keys():
        temp[k] = log2(N / temp[k])
    return temp


def tf_to_idf(tfs: dict, idf: dict):
    for w in tfs.keys():
        tfs[w] = tfs[w] * idf[w]
    return sorted([(k, v) for k, v in tfs.items()], key=lambda p: -p[1])


def usr_features(usr_bid: dict):
    for u, b in usr_bid.items():
        res = []
        for i in b:
            if i in business_profile.keys():
                res.extend(business_profile[i])
        usr_bid[u] = res
    return usr_bid


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 4:
        print('Please rerun and confirm the parameters.')
    else:
        starttime = time.time()
        input_path = arg[1]  # train_file
        output_path = arg[2]  # model_file
        stopwords_path = arg[3]

        with open(stopwords_path) as f:
            stopwords = f.read().split()  # list of stopwords

        # build RDD and preprocess
        sc = SparkContext()
        rdd = sc.textFile(input_path).map(lambda x: json.loads(x)) \
            .map(lambda x: (x['business_id'], x['text'])).cache()
        rdd_formatted = rdd.reduceByKey(lambda x, y: x + y).filter(lambda x: x[0] is not None and x[1] is not None) \
            .map(lambda x: (x[0], formatting(x[1], stopwords))).persist()  # 0:businessid:str, 1:words:list

        # IDF of words
        num_doc = rdd.map(lambda x: x[0]).count()  # number of all docs; N in IDF
        words_idf = word_doc_idf(rdd_formatted.map(lambda x: list(set(x[1]))) \
                                 .reduce(lambda x, y: x + y), num_doc)  # k:words, v:number of docs mentioned the word

        # build business profiles
        business_profile = rdd_formatted.reduceByKey(lambda x, y: x.extend(y)) \
            .mapValues(lambda l: tf(l)).mapValues(lambda d: tf_to_idf(d, words_idf)) \
            .mapValues(lambda l: l[:200]).mapValues(lambda l: [p[0] for p in l]) \
            .collectAsMap()  # k:businessid, v:200 feature words

        # build user profiles
        user_profile = sc.textFile(input_path).map(lambda x: json.loads(x)) \
            .map(lambda x: (x['user_id'], [x['business_id']])) \
            .reduceByKey(lambda x, y: x + y).mapValues(lambda l: list(set(l))) \
            .collectAsMap()
        user_profile = usr_features(user_profile)

        # write into model file
        with open(output_path, 'w') as f:
            f.write(json.dumps(business_profile) + '\n')
            f.write(json.dumps(user_profile))

        print("Duration:", (time.time() - starttime))
