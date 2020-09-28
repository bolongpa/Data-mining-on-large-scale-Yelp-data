import sys
import time
import json
from math import sqrt
from pyspark import SparkContext


def cos_distance(usr_id: str, business_id: str):
    if usr_id not in user_profile.keys() or business_id not in business_profile.keys():
        return 0
    else:
        cnt = 0
        usr_feature = user_profile[usr_id]
        business_feature = business_profile[business_id]
        for w in business_feature:
            if w in usr_feature:
                cnt += 1
        denominator = sqrt(len(usr_feature) * 200)
        return cnt / denominator


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 4:
        print('Please rerun and confirm the parameters.')
    else:
        starttime = time.time()
        tes_file_path = arg[1]  # train_file
        model_file_path = arg[2]  # model_file
        output_path = arg[3]

        # read model
        with open(model_file_path, 'r') as m:
            business_profile = json.loads(m.readline())
            user_profile = json.loads(m.readline())

        # build RDD and predict
        sc = SparkContext()
        rdd = sc.textFile(tes_file_path).map(lambda x: json.loads(x)) \
            .map(lambda x: (x['user_id'], x['business_id'])).collect()

        res = []
        for target_pair in rdd:
            sim = cos_distance(target_pair[0], target_pair[1])
            if sim >= 0.01:
                prediction = dict()
                prediction['user_id'] = target_pair[0]
                prediction['business_id'] = target_pair[1]
                prediction['sim'] = sim
                res.append(prediction)

        with open(output_path, 'w') as f:
            for i in res:
                f.write(json.dumps(i) + '\n')

        print("Duration:", (time.time() - starttime))
