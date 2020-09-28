import sys
import time
import json
from itertools import combinations
from pyspark import SparkContext


def minhash(rowindexes, rownum, hashnum):
    signatures = []  # to store hash signatures
    hashfunc1 = lambda x: (3 * x + 2) % rownum
    hashfunc2 = lambda x: (5 * x + 3) % rownum
    for i in range(1, hashnum + 1):
        hashfunc = lambda x: ((i * hashfunc1(x) + i * hashfunc2(x) + i * i) % 33569) % rownum
        # find signature using current hash function
        rowindexes_hash = [hashfunc(x) for x in list(rowindexes)]
        signatures.append(min(rowindexes_hash))
    return signatures


def breaktobands(signature: list, b: int, r: int):
    return [((i, tuple(signature[1][i * r:i * r + r])), [signature[0]]) for i in range(b)]


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 3:
        print('Please rerun and confirm the parameters.')
    else:
        starttime = time.time()
        input_path = arg[1]
        output_path = arg[2]

        # build RDD
        sc = SparkContext()
        rdd = sc.textFile(input_path).map(lambda x: json.loads(x)) \
            .map(lambda x: (x['business_id'], x['user_id']) if x['stars'] is not None else None) \
            .distinct().cache()
        usrid = rdd.map(lambda x: x[1]).distinct().collect()
        num_row = len(usrid)  # number of distinct users
        rdd_prepare = rdd.map(lambda x: (x[0], [x[1]])) \
            .reduceByKey(lambda x, y: x + y) \
            .mapValues(lambda usrs: [usrid.index(u) for u in usrs])\
            .cache()  # key: business_id; value: [row num of reviewed usrs]
        dict_business = rdd_prepare.collectAsMap()

        # minhash
        num_hash = 35  # number of hash functions
        rdd_minhash = rdd_prepare.mapValues(lambda x: minhash(x, num_row, num_hash))  # (business_id, signature:list)

        # LSH
        b = 35  # number of bands
        r = int(num_hash / b)
        candidate_pairs = rdd_minhash.flatMap(lambda x: breaktobands(x, b, r)) \
            .reduceByKey(lambda x, y: x + y) \
            .map(lambda x: tuple(set(x[1]))).filter(lambda x: len(x) >= 2) \
            .flatMap(lambda x: list(combinations(x, 2)))\
            .distinct().collect()

        # generate result from candidates
        similar_result = dict()
        for p in candidate_pairs:  # p:tuple
            c1 = set(dict_business[p[0]])  # set of row_num of usrs
            c2 = set(dict_business[p[1]])
            similarity = len(c1.intersection(c2)) / len(c1.union(c2))
            if similarity >= 0.05:
                similar_result[frozenset(p)] = similarity  # 0:b1, 1:b2, 2:sim
        out = [{'b1': list(k)[0], 'b2': list(k)[1], 'sim': v} for k, v in similar_result.items()]

        # write to output
        with open(output_path, 'w') as f:
            for i in range(len(out)):
                f.write(json.dumps(out[i])+'\n')
        print("Duration:", (time.time() - starttime))
