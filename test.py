#!/usr/bin/env python
from scipy.linalg import block_diag
import numpy as np
import zerorpc
# import json

__author__ = 'Dennis Schwartz'

settings = {
    "undirected": True
}


class TensorNet(object):
    """Tensor representation for a multiplex network"""

    def __init__(self):
        self.layers_tensor = {}
        self.node_tensors = []
        self.network_tensor = {}

    # Edges format: [[src, trg, layer], [src, trg, layer]]
    def initialize(self, nodes, edges, layers):
        for i in xrange(len(layers)):
            self.node_tensors[i] = self.create_nodes_tensor(nodes, get_edges_by_layer(edges, layers[i]))
        self.layers_tensor = self.create_layers_tensor(len(nodes), layers, 1)
        block = block_diag(self.node_tensors)
        self.network_tensor = np.add(block, np.kron(self.layers_tensor, np.eye(len(nodes))))

    @staticmethod
    def create_nodes_tensor(nodes, edges):
        size = len(nodes)
        res = np.zeros((size, size))
        for edge in edges:
            print edge
            src = edge[0] - 1
            trg = edge[1] - 1
            res[src][trg] = 1
        if settings["undirected"]:
            res = np.subtract(np.add(res, np.transpose(res)), np.diag(np.diag(res)))
        return res

    @staticmethod
    def create_layers_tensor(layers, size, omega):
        if not len(layers):
            return 0
        layers_tensor = np.multiply(np.ones(size, size), omega)
        print layers_tensor  # TODO: Remove this
        layers_tensor = np.subtract(layers_tensor, np.diag(np.diag(layers_tensor)))
        return layers_tensor


def get_edges_by_layer(edges, l):
    res = []
    for edge in edges:
        if edge[2] == l:
            res.append(edge)
    return res


# if __name__ == "__main__":
#
#     e = [[1, 2, 1], [1, 3, 1], [1, 4, 1], [1, 5, 1], [1, 2, 2], [1, 3, 2], [1, 4, 2], [1, 5, 2]]
#     net = TensorNet([1, 2, 3, 4, 5], e, ["A", "B"])
#     tensor = net.create_layer_tensor(net.nodes, [[1, 2, 1], [1, 3, 1], [1, 4, 1], [1, 5, 1]])
#     print(tensor)

s = zerorpc.Server(TensorNet())
s.bind("tcp://0.0.0.0:4242")
s.run()
