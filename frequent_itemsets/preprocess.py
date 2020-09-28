import csv
import json
from pyspark import SparkContext


if __name__ == "__main__":
    # build RDD
    sc = SparkContext()
    business_rdd = sc.textFile('business.json')\
        .map(lambda x: json.loads(x))\
        .filter(lambda x: x['state'] == 'NV')\
        .map(lambda x: (x['business_id'], 1))  # all business_id with state being NV
    review_rdd = sc.textFile('review.json')\
        .map(lambda x: json.loads(x))\
        .map(lambda x: (x['business_id'], x['user_id']))  # all business_id and user_id
    join = business_rdd.join(review_rdd)\
        .map(lambda x: (x[1][1], x[0]))\
        .collect()
    with open('preprocess.csv', 'w') as result_file:
        wr = csv.writer(result_file)
        wr.writerows([('user_id', 'business_id')])  # header
        wr.writerows(join)
