import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sberpm
import pm4py
from sklearn.preprocessing import LabelEncoder
from sberpm.metrics import ActivityMetric, TransitionMetric, IdMetric, TraceMetric, UserMetric


class DrillPlaner():
    def __init__(self, Miner, dataholder, data, time_idx,stats):
        super().__init__()
        self.Miner = Miner(dataholder)
        self.Miner.apply()
        self.data = data
        self.dataholder = dataholder
        self.time_idx = time_idx
        self.graph_nodes = list(self.Miner.graph.nodes.keys())
        self.graph_edges = list(self.Miner.graph.edges.keys())
        self.stats = stats

    def calculate_sberpm_metrics(self):
        '''
        activity metric - metrics by activities (group by activity_column)
        TransitionMetric – metrics by transitions (two consecutive activities) (group by unique transitions)
        IdMetric– metrics by ids (group by id_column)
        TraceMetric – metrics by event traces (group by unique traces)
        UserMetric – metrics by users (group by user_column)
        '''
        activity_metric = ActivityMetric(self.data_holder, time_unit='day')
        activity_metric.apply()

        transition_metric = TransitionMetric(self.data_holder, time_unit='day')
        transition_metric.apply()

        id_metric = IdMetric(self.data_holder, time_unit='day')
        id_metric.apply()

        trace_metric = TraceMetric(self.data_holder, time_unit='day')
        trace_metric.apply()

        return acitvity_metric, transition_metric, id_metric, trace_metric



    def graph_graphviz(self):
        painter = GraphvizPainter()
        painter.apply(self.Miner.graph)
        self.painter = painter
        self.painter.show()

    #  painter.write_graph(f"{self.Miner}.png", format='png')
    def digraph(self, tol=2):
        G = nx.DiGraph()
        time = np.round(self.data[self.time_idx].values, tol)
        for k, w in zip(self.Miner.graph.edges.keys(), time):
            G.add_edge(k[0], k[1], weight=w)
        self.G = G
        return G

    def plan_calculation(self,plan: list, digraph, type_of_phase, algorithm='dijkstra'):
       '''
       Here you can

       '''

        plan.append('endevent')
        path = []
        time = []
        stats_time = []

        for i, s in enumerate(plan):
            if i == 0:
                p = nx.shortest_path(self.G, source='startevent', target=s, method=algorithm)
            else:
                p = nx.shortest_path(self.G, source=plan[i - 1], target=plan[i], method=algorithm)

            if len(p) > 2:
                for j in range(0, len(p) - 1):
                    pn = nx.shortest_path(self.G, source=p[j], target=p[j + 1], method=algorithm)
                    t = nx.path_weight(self.G, pn, weight='weight')
                    path.append(pn)
                    time.append(t)

            else:

                path.append(p)
                t = nx.path_weight(self.G, path[i], weight='weight')
                time.append(t)
        sum_time = np.array(time).sum()

        for i in range(0, len(path) - 1):
            try:
                code = int(path[i][1])
                name_of_operation = data[data['Operation code'] == code]['Type of work'].iloc[0]

                # find best time in offset
                median_time = self.stats.loc[(stats['Operation code'] == code) & (self.stats['Phase'] == type_of_phase)][
                    'median'].values
                stats_time.append(median_time[0])

                if code < 300:
                    ROP_mean = ROP_phase[ROP_phase['Phase'] == type_of_phase]['mean'].values
                    print(
                        f'Code of operation {path[i][1]}. Operation time in trace {time[i]} h. Median time in offset  {round(median_time[0], 2)} h. {name_of_operation}'
                        f' Mean rate of penetration of this section {round(ROP_mean[0], 2)} m/h')
                else:
                    print(
                        f'Code of operation {path[i][1]}. Operation time in trace {time[i][0]} h. Median time in offset  {round(median_time[0], 2)} h.  {name_of_operation}')



            except (ValueError, IndexError):
                continue
        print(f'Average time for plan in graph {round(sum_time, 2)} h')
        print(f'Average time for plan in stats {round(np.array(stats_time).sum(), 2)} h')

        return path, time

    def shortest_paths(self, source, target, weight='weight'):
        # provides a dictionary of all paths of the shortest length wrt weight in case the shortest path is non-unique
        gen = nx.shortest_simple_paths(self.G, source, target, weight=weight)
        a = gen.__next__()
        la = nx.path_weight(self.G, a, weight=weight)
        sp = {tuple(a): la}
        lb = 0
        while True:
            b = gen.__next__()
            lb = nx.path_weight(self.G, b, weight=weight)
            if lb <= la:
                sp[tuple(b)] = lb
            else:
                break
        return sp

    def oneshortest(self, source, target, weight='weight'):
        # Bellman-Ford shortest; precaution against ncycles
        a = nx.bellman_ford_path(self.G, source, target, weight=weight)
        return {tuple(a): nx.path_weight(self.G, a, weight=weight)}

    def plot_graph(self):
        nx_graph(self.G, plot_shortest_path=True)

        # https://vdocuments.net/metrics-in-process-discovery-2015-12-21-metrics-in-process-discovery-fabian.html?page=17

    def IoU_metric(self, graph2):
        # structural similarity
        return len(set(self.graph_nodes).intersection(set(graph2.graph_nodes))) / len(
            set(self.graph_nodes).union(set(graph2.graph_nodes)))

    def NAM_metric(self, graph2):
        # Construct the coverability tree for two miners
        # Get the principal transition sequences from the trees.
        # Return the PTS-Based Similarity Measure based on the sequences found.
        # behavioural similarity
        all_nodes = list(set(self.graph_nodes).union(set(graph2.graph_nodes)))
        NAM1 = np.zeros((len(all_nodes), len(all_nodes)))
        NAM2 = np.copy(NAM1)
        for NAM, graph in ((NAM1, self), (NAM2, graph2)):
            for edge in graph.graph_edges:
                i = all_nodes.index(edge[0])
                j = all_nodes.index(edge[1])
                NAM[i, j] = 1
        return np.trace((NAM1 - NAM2).T @ (NAM1 - NAM2))


# networkx graph plotting
def nx_graph(G, plot_shortest_path=False):
    plt.subplots(1, 1, figsize=(15, 10))

    elarge = [(u, v) for (u, v, d) in G.edges(data=True)]  # if d["weight"] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True)]  # if d["weight"] <= 0.5]

    pos = nx.spring_layout(G, seed=100, iterations=10)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='black')  # cmap='plasma')
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=8, edge_color='black')
    # nx.draw_networkx_edges(
    #   G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="black", style="dashed")

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.001)

    if plot_shortest_path:
        path = nx.shortest_path(G, source='startevent', target='endevent')
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='r', node_size=1000)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=8, edge_color='red')
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
    def __init__(self, data, time_name, code_name,prepare_dataset = True):
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

        if prepare_dataset:
            self.dataset = self.prepare_features(data)
            self.node_features = self.dataset.to_numpy()
            self.dataset['source'] = self.dataset['Operation code']
            self.dataset['target'] = self.operation_lag(self.dataset, ['Operation code'], 1)


   # @staticmethod
    def prepare_features(self,dataset):
        le = LabelEncoder()
        dataset['Section'] = le.fit_transform(dataset['Section'])
        dataset['Phase'] = le.fit_transform(dataset['Phase'])
        dataset = dataset[['Section', 'Phase', 'Time, h (in grains)', 'Operation code']]
        return dataset

   # @staticmethod
    def operation_lag(self,data, code_column, lag, next_operation=False):
        df = np.zeros_like(data[code_column].values)
        df[lag:] = data[code_column].values[:-lag]
        if next_operation:
            df[:-lag] = data[code_column].values[lag:]
        return df

    def to_dfg(self):
        self.dfg = nx.from_numpy_matrix(self.graph_matrix, create_using=nx.DiGraph())
        miner_nodes = {a: self.nodes[a] for a in self.dfg.nodes}
        self.dfg = nx.relabel_nodes(self.dfg, miner_nodes)
        self.graph_edges = list(map(lambda x: (str(x[0]), str(x[1])), list(self.dfg.edges())))
        self.graph_nodes = list(map(str, self.dfg.nodes()))
        return self.dfg
