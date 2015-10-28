#!/usr/bin/env python
import numpy as np
import zerorpc
import json

__author__ = 'Dennis Schwartz'


class TensorNet(object):
    """Tensor representation for a multiplex network"""
    network = ''

    # def __init__(self, nodes, edges, layers):
    #     self.nodes = nodes
    #     self.edges = edges
    #     self.layers = layers
    #     self.tensor = {}

    @staticmethod
    def create_layer_tensor(nodes, edges):
        size = len(nodes)
        res = np.zeros((size, size))
        for edge in edges:
            print edge
            s = edge[0] - 1
            t = edge[1] - 1
            res[s][t] = 1
        serialized = json.dumps(res.tolist())
        print serialized
        return serialized

# if __name__ == "__main__":
#
#     e = [[1, 2, 1], [1, 3, 1], [1, 4, 1], [1, 5, 1], [1, 2, 2], [1, 3, 2], [1, 4, 2], [1, 5, 2]]
#     net = TensorNet([1, 2, 3, 4, 5], e, ["A", "B"])
#     tensor = net.create_layer_tensor(net.nodes, [[1, 2, 1], [1, 3, 1], [1, 4, 1], [1, 5, 1]])
#     print(tensor)

s = zerorpc.Server(TensorNet())
s.bind("tcp://0.0.0.0:4242")
s.run()

