import torch
import cv2
import numpy as np
import networkx as nx
import math
import plotly.graph_objects as go
from networkx.drawing.nx_agraph import graphviz_layout
from graphviz import Digraph

class Node():
    def __init__(self, uid, op='Conv2d', input_shape=None, output_shape=None, info=None, params=None):
        """
        uid: unique id for the layer in graph
        op: torch.nn layer name
        params: layer parameters for nn.function
        info: info for graph visualization {'label': label, 'x': x, 'y': y}
        """
        self.id = uid
        self.op = op
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.params = params if params else {}
        self.info = info if info else {}

class Graph():
    def __init__(self):
        self.reset()

    def reset(self):
        self.nodes = []
        self.edges = []
        self.nd_id = 0

    def build(self):
        """Generate a Networkx graph.
        Returns a Networkx Digraph object.
        """
        graph = nx.DiGraph()
        nodes = []
        for k, n in enumerate(self.nodes):
            label = "{} {}".format(n.op, n.info['label'])
            nodes.append((str(n.id), {'label': label, 'shape': 'box'}))

        graph.add_nodes_from(nodes)
        for a, b in self.edges:
            graph.add_edge(str(a), str(b))
        return graph

    def build_dot(self, sel_nd = -1):
        """Generate a GraphViz Dot graph.
        Returns a GraphViz Digraph object.
        """
        dot = Digraph()
        dot.attr("node", shape="box", style="filled")
        for k, n in enumerate(self.nodes):
            label = "{} {}".format(n.id, n.op)
            if sel_nd == n.id:
                label += '<<<<'
            dot.node(str(n.id), label)

        for a, b in self.edges:
            dot.edge(str(a), str(b))

        return dot

def preprocess(img):
    """pre process input drawing
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    threshold = cv2.bitwise_not(threshold)
    return threshold


def find_tip(points, convex_hull):
    """find arrow tip
    """
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)
    if len(indices) > 2:
        for i in range(2):
            j = indices[i] + 2
            if j > length - 1:
                j = length - j
            if np.all(points[j] == points[indices[i - 1] - 2]):
                return tuple(points[j]), tuple(furthest_point(points[j], points, convex_hull)[0])

    furthest = 0
    for a in convex_hull:
        pnt, d = furthest_point(points[a], points, convex_hull)
        if furthest < d:
            furthest = d
            fpnt = (points[a], pnt)
    return tuple(fpnt[0]), tuple(fpnt[1])


def get_edge(point_1, point_2, graph):
    """get the nodes close to point1 and point2 and create an edge between them
    """
    if len(graph.nodes) < 2:
        return
    nd_1 = closest_node(point_1, graph.nodes)
    nd_2 = closest_node(point_2, graph.nodes)
    if nd_1.id < nd_2.id:  # earlier node
        graph.edges.append((nd_1.id, nd_2.id))  # (id, dict{})
    else:
        graph.edges.append((nd_2.id, nd_1.id))


def furthest_point(point, approx, hull):
    """find the furthest point from point inside a contour
    """
    furthest = 0
    for i in hull:
        d = math.dist(point, approx[i])
        if furthest < d:
            furthest = d
            point2 = approx[i]
    return point2, furthest


def closest_node(point, nodes):
    """find the node that is closest to a point in drawing
    """
    mindist = float('inf')
    mindist_nd = 0
    for nd in nodes:
        d = math.dist(point, (int(nd.info['x']), int(nd.info['y'])))
        if d < mindist:
            mindist = d
            mindist_nd = nd
    return mindist_nd


def building_graph(img, graph):
    """build the graph from image
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    added = False
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        # finding center point of shape
        M = cv2.moments(cnt)
        x, y = 0, 0
        if M['m00'] != 0.0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])

        if len(graph.nodes) > 1 and \
                6 > sides > 3 and \
                ((sides + 2 == len(approx)) or (sides + 1 == len(approx))):
            arrow_tip, arrow_end = find_tip(approx[:, 0, :], hull.squeeze())
            cv2.circle(img, arrow_tip, 3, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, arrow_end, 3, (0, 255, 0), cv2.FILLED)
            get_edge(arrow_tip, arrow_end, graph)
            added = True
        elif len(approx) <= 6 and len(approx) >= 3:
            cv2.putText(img, 'Quadrilateral', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            label = 'layer'
            # todo: from text or select
            # todo: select nn.function params
            node = Node(graph.nd_id, op='Conv2d',
                        info={'label': label, 'x': x, 'y': y},
                        params={'in_channels':64, 'out_channels':64, 'kernel_size':3})
            graph.nd_id += 1
            graph.nodes.append(node)
            added = True
        else:
            cv2.putText(img, 'circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return added

def visualize_graph(graph, title=""):
    """visualize a graph using plotly
    G: Graph
    """
    G = graph.build()
    positions = graphviz_layout(G)
    edge_x = []
    edge_y = []
    #     edge_w = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)

    for node in graph.nodes:
        node_text.append(node.op)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        text=node_text,
        mode='markers+text',
        hoverinfo='text',
        marker_symbol=1,
        marker=dict(
            reversescale=True,
            color=[],
            size=10,
            line_width=1))

    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>' + title,
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

