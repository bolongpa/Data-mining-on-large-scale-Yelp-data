import os

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell")

import sys
from pyspark.sql import SparkSession
from pyspark import SparkContext
from itertools import combinations
import time
from graphframes import *


if __name__ == "__main__":
    arg = sys.argv
    if len(arg) != 4:
        print('Please check the parameters and run again.')
    else:
        start_time = time.time()
        filter_threshold = int(arg[1])
        input_file_path = arg[2]
        community_output_file_path = arg[3]

        # construct graph
        sc = SparkContext()
        sc.setLogLevel('WARN')
        rdd = sc.textFile(input_file_path)
        header = rdd.take(1)
        edge_pair = rdd.filter(lambda line: line != header[0]).map(lambda x: x.split(','))\
            .map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y)\
            .flatMapValues(lambda l: list(combinations(l, 2))).map(lambda x: (frozenset(x[1]), [x[0]]))\
            .reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) >= filter_threshold)\
            .map(lambda x: list(x[0])).flatMap(lambda x: [(x[0], x[1]), (x[1], x[0])])  # all filtered user pairs
        spark = SparkSession(sc)
        edges = edge_pair.toDF(["src", "dst"])  # 498*2 edges in dataframe
        #edges.show(5)
        vertices = edge_pair.flatMap(lambda x: [tuple([x[0]]), tuple([x[1]])])\
            .distinct().toDF(["id"])  # 222 vertices in dataframe (only keep vertices occurred in edge pairs)
        #vertices.show(5)

        g = GraphFrame(vertices, edges)
        community_result = g.labelPropagation(maxIter=5)
        result_rdd = community_result.rdd.map(tuple).map(lambda x: (x[1], [x[0]]))\
            .reduceByKey(lambda x, y: x + y).mapValues((lambda l: sorted(l)))\
            .sortBy(lambda x: (len(x[1]), x[1][0])).map(lambda x: x[1]).collect()

        with open(community_output_file_path, 'w') as f:
            for community in result_rdd:
                f.write(str(community)[1:-1] + '\n')

        print("Duration:", (time.time() - start_time))
