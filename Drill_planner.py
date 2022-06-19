import pandas as pd
import  matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sberpm
import pm4py

class DrillPlaner():
  def __init__(self, Miner, dataholder, data, time_idx):
    super().__init__()
    self.Miner = Miner(dataholder)
    self.Miner.apply()
    self.data = data
    self.dataholder = dataholder
    self.time_idx = time_idx
    self.graph_nodes = list(self.Miner.graph.nodes.keys())
    self.graph_edges = list(self.Miner.graph.edges.keys())
  def graph_graphviz(self):
    painter = GraphvizPainter()
    painter.apply(self.Miner.graph)
    self.painter = painter
    self.painter.show()
  #  painter.write_graph(f"{self.Miner}.png", format='png')
  def digraph(self, tol = 2):
    G = nx.DiGraph()
    time = np.round(self.data[self.time_idx].values,tol)
    for k,w in zip(self.Miner.graph.edges.keys(),time):
      G.add_edge(k[0],k[1],weight =w )
    self.G = G
    return G
  def shortest_paths(self, source, target, weight = 'weight'):
    #provides a dictionary of all paths of the shortest length wrt weight in case the shortest path is non-unique
    gen = nx.shortest_simple_paths(self.G, source, target, weight=weight)
    a = gen.__next__()
    la = nx.path_weight(self.G, a, weight=weight)
    sp = {tuple(a): la}
    lb = 0
    while True:
      b = gen.__next__()
      lb = nx.path_weight(self.G, b, weight=weight)
      if lb<=la:
        sp[tuple(b)] = lb
      else:
        break
    return sp
  def oneshortest(self, source, target, weight = 'weight'):
    #Bellman-Ford shortest; precaution against ncycles
    a = nx.bellman_ford_path(self.G, source, target, weight = weight)
    return {tuple(a): nx.path_weight(self.G, a, weight=weight)}
  def plot_graph(self):
    nx_graph(self.G,plot_shortest_path=True)

      #https://vdocuments.net/metrics-in-process-discovery-2015-12-21-metrics-in-process-discovery-fabian.html?page=17
  def IoU_metric(self, graph2):
      #structural similarity
    return len(set(self.graph_nodes).intersection(set(graph2.graph_nodes)))/len(set(self.graph_nodes).union(set(graph2.graph_nodes)))
  def NAM_metric(self, graph2):
      # Construct the coverability tree for two miners
      # Get the principal transition sequences from the trees.
      # Return the PTS-Based Similarity Measure based on the sequences found.
      #behavioural similarity
    all_nodes = list(set(self.graph_nodes).union(set(graph2.graph_nodes)))
    NAM1 = np.zeros((len(all_nodes), len(all_nodes)))
    NAM2 = np.copy(NAM1)
    for NAM, graph in ((NAM1, self),(NAM2, graph2)):
      for edge in graph.graph_edges:
        i = all_nodes.index(edge[0])
        j = all_nodes.index(edge[1])
        NAM[i,j] = 1
    return np.trace((NAM1-NAM2).T@(NAM1-NAM2))



# networkx graph plotting
def nx_graph(G,plot_shortest_path = False):
  plt.subplots(1,1,figsize=(15,10))


  elarge = [(u, v) for (u, v, d) in G.edges(data=True)] #if d["weight"] > 0.5]
  esmall = [(u, v) for (u, v, d) in G.edges(data=True)] #if d["weight"] <= 0.5]


  pos = nx.spring_layout(G, seed=100,iterations=10) # positions for all nodes - seed for reproducibility


  # nodes
  nx.draw_networkx_nodes(G, pos, node_size=500,node_color='black')#cmap='plasma')
  # edges
  nx.draw_networkx_edges(G, pos, edgelist=elarge, width=8,edge_color='black')
  #nx.draw_networkx_edges(
  #   G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="black", style="dashed")



    # node labels
  nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G, pos, edge_labels)

  ax = plt.gca()
  ax.margins(0.001)

  if plot_shortest_path:
    path = nx.shortest_path(G,source='startevent',target='endevent')
    path_edges = list(zip(path,path[1:]))
    nx.draw_networkx_nodes(G,pos,nodelist=path,node_color='r',node_size=1000)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=8,edge_color='red')
  plt.axis("off")
  plt.tight_layout()
  plt.show()


class GraphMiner():
  def __init__(self, name, predecessor, assoc_time, parent_time):
    self.parent = predecessor
    self.name = name
    self.time = assoc_time
    self.parent_time = parent_time


class DrillGraphMiner():
  def __init__(self, data, time_name, code_name):
    graph_codes = []
    codes = data[code_name].to_numpy()  # can pass codes and times instead of raw data, suitably changing the code
    times = np.round(data[time_name].to_numpy(), 2)
    events = ['startevent'] + list(set(codes)) + ['endevent']

    for i in range(len(codes[1:])):
      graph_codes.append(GraphMiner(codes[i + 1], codes[i], times[i + 1], times[i]))
    graph_codes.append(GraphMiner('endevent', codes[-1], times[-1] + 1e-10, times[-1]))
    graph_codes.insert(0, GraphMiner(codes[0], 'startevent', times[0], times[0] - 1e-10))
    gr = np.zeros((len(events), len(events)))

    for bullet in graph_codes:  # digraph from row to column
      gr[events.index(bullet.parent), events.index(bullet.name)] = bullet.time
    self.graph_matrix = gr
    self.nodes = events

  def to_dfg(self):
    self.dfg = nx.from_numpy_matrix(self.graph_matrix, create_using=nx.DiGraph())
    miner_nodes = {a: self.nodes[a] for a in self.dfg.nodes}
    self.dfg = nx.relabel_nodes(self.dfg, miner_nodes)
    self.graph_edges = list(map(lambda x: (str(x[0]), str(x[1])), list(self.dfg.edges())))
    self.graph_nodes = list(map(str, self.dfg.nodes()))
    return self.dfg