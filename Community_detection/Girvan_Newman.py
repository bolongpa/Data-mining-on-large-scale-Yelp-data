import sys
from pyspark import SparkContext
from itertools import combinations
import time
import copy


def betweenness_1node(root: str, graph):
    # BFS
    queue = [root]  # queue order of visiting
    visit = [root]  # nodes visited or to visit soon
    level = dict()  # level of each node in BFS tree
    level[root] = 0
    child_parent = {}  # {child:[parents in bfs],}
    child_parent[root] = [0]  # initialize the root parent as 1
    while queue:
        node = queue.pop(0)
        for c in graph[node]:  # travel neighbors of current node
            if c not in visit:
                queue.append(c)
                visit.append(c)
                level[c] = level[node] + 1
                child_parent[c] = [node]
            elif level[c] == level[node] + 1:
                child_parent[c] += [node]
    # compute betweennesses
    node_weight = dict()
    for c_order in range(len(visit)-1, 0, -1):  # travel from right to left except root
        child = visit[c_order]
        if child not in node_weight.keys():
            node_weight[child] = 1
        else:
            node_weight[child] += 1
        div = sum([len(child_parent[p]) for p in child_parent[child]])
        for p in child_parent[child]:
            edgeweight = node_weight[child] / div * len(child_parent[p])
            yield (frozenset([child, p]), edgeweight)
            if p not in node_weight.keys():
                node_weight[p] = edgeweight
            else:
                node_weight[p] += edgeweight


def reachable(i, j, commu):  # check if node i, j in the same community
    for c in commu:
        if i in c and j in c:
            return True
    return False


def getCommu(node, graph):
    # BFS
    root = node
    queue = [root]  # queue order of visiting
    visit = [root]  # nodes visited or to visit soon
    while queue:
        node = queue.pop(0)
        for c in graph[node]:  # travel neighbors of current node
            if c not in visit:
                queue.append(c)
                visit.append(c)
    return visit


def collectCommu(graph):
    communities = []
    vertices = list(graph.keys())
    while vertices:
        singlecom = getCommu(vertices[0], graph)
        communities.append(singlecom)
        vertices = [v for v in vertices if v not in singlecom]
        tempv = list(graph.keys())
        for v in tempv:
            if v in singlecom:
                del graph[v]
            else:
                graph[v] = list(set(graph[v]) - set(singlecom))
    return communities


if __name__ == "__main__":
    arg = sys.argv
    if len(arg) != 5:
        print('Please check the parameters and run again.')
    else:
        start_time = time.time()
        filter_threshold = int(arg[1])
        input_file_path = arg[2]
        betweenness_output_file_path = arg[3]
        community_output_file_path = arg[4]

        # construct graph
        sc = SparkContext()
        rdd = sc.textFile(input_file_path)
        header = rdd.take(1)
        rdd_clear = rdd.filter(lambda line: line != header[0]).map(lambda x: x.split(','))\
            .map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y)\
            .flatMapValues(lambda l: list(combinations(l, 2))).map(lambda x: (frozenset(x[1]), [x[0]]))\
            .reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) >= filter_threshold)\
            .map(lambda x: list(x[0])).cache()
        graph = rdd_clear.flatMap(lambda x: [(x[0], [x[1]]), (x[1], [x[0]])])\
            .reduceByKey(lambda x, y: x + y).mapValues(lambda x: list(set(x)))\
            .collectAsMap()  # {node:[connected nodes],}
        # compute betweennesses and write into file
        betweenness = rdd_clear.flatMap(lambda x: x).distinct().flatMap(lambda x: betweenness_1node(x, graph))\
            .reduceByKey(lambda x, y: x + y)\
            .map(lambda x: (tuple(sorted(list(x[0]))), x[1] / 2)).sortBy(lambda x: (-x[1], x[0][0])).collect()

        with open(betweenness_output_file_path, 'w') as f:
            for b in betweenness:
                f.write(str(b)[1:-1] + '\n')

        # detect best communities
        m = rdd_clear.count()  # the edge number of the original graph: 498
        vertices = rdd_clear.flatMap(lambda x: list(x)).distinct().collect()  # all nodes(str) in list, 222
        degree = rdd_clear.flatMap(lambda x: [(x[0], [x[1]]), (x[1], [x[0]])])\
            .reduceByKey(lambda x, y: x + y).mapValues(lambda x: len(x)).collectAsMap()  # {node:degree,}
        Q = 0
        tempbetweenness = betweenness
        tempgraph = copy.deepcopy(graph)  # {node:[connected nodes],} (the same as graph)
        resultgraph = tempgraph
        resultcommu = []
        while tempbetweenness:
            tempQ = 0
            removal = tempbetweenness.pop(0)[0]  # tuple (node1, node2)
            tempgraph[removal[0]].remove(removal[1])
            tempgraph[removal[1]].remove(removal[0])
            # compute new betweenness
            tempbetweenness = rdd_clear.flatMap(lambda x: x).distinct()\
                .flatMap(lambda x: betweenness_1node(x, tempgraph)).reduceByKey(lambda x, y: x + y)\
                .map(lambda x: (tuple(sorted(list(x[0]))), x[1] / 2)).sortBy(lambda x: (-x[1], x[0][0])).collect()
            # get current communities
            tempg = copy.deepcopy(tempgraph)
            communities = collectCommu(tempg)
            for n1 in vertices:
                for n2 in vertices:
                    if reachable(n1, n2, communities):  # n1 and n2 in the same community
                        if n2 in graph[n1]:  # connected in original graph
                            A = 1
                        else:
                            A = 0
                        tempQ += (A - 0.5 * degree[n1] * degree[n2] / m) * 0.5 / m
            if tempQ < Q:
                break  # break the loop when Q starts decreasing
            else:
                Q = tempQ
                resultgraph = tempgraph
                resultcommu = communities
                print(len(communities), Q)

        # generate communities
        communities = []
        while vertices:
            n = vertices[0]
            com = getCommu(n, resultgraph)  # one single community containing vertex n
            communities.append(com)
            vertices = [v for v in vertices if v not in com]
        # sort communities and write into file
        commu_rdd = sc.parallelize(resultcommu).map(lambda x: sorted(x)).sortBy((lambda x: (len(x), x[0]))).collect()
        with open(community_output_file_path, 'w') as f:
            for community in commu_rdd:
                f.write(str(community)[1:-1] + '\n')

        print("Duration:", (time.time() - start_time))
