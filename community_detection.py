#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import igraph
from igraph import *
import os

clusters = None
directed = False
weights = None
vertex_weights = None


def find_communities(g, method):
    if method == 'edge_betweenness':
        return Graph.community_edge_betweenness(g, clusters, directed, weights).as_clustering()
    elif method == 'leading_eigenvector':
        return Graph.community_leading_eigenvector(g, clusters)
    elif method == 'spinglass':
        return Graph.community_spinglass(g)
    elif method == 'walktrap':
        return Graph.community_walktrap(g, weights, steps=4).as_clustering()
    elif method == 'fastgreedy':
        return Graph.community_fastgreedy(g.simplify(multiple=True, loops=True, combine_edges=None),
                                          weights).as_clustering()
    elif method == 'infomap':
        return Graph.community_infomap(g, weights, vertex_weights, trials=10)
    elif method == 'multilevel':
        return Graph.community_multilevel(g, weights, return_levels=False)
    elif method == 'label_propagation':
        return Graph.community_label_propagation(g, weights, initial=None, fixed=None)
    elif method == 'optimal_modularity':
        return Graph.community_optimal_modularity(g)
    elif method == 'leiden':
        return Graph.community_leiden(g)


methods = ['edge_betweenness', 'leading_eigenvector', 'spinglass', 'walktrap', 'fastgreedy', 'infomap', 'multilevel',
           'label_propagation', 'optimal_modularity']
