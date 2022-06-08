import pandas as pd
import  matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sberpm
import pm4py


class DrillDataHolder:
    def __init__(self, log_path,
                 nrows = None,
                 event_id_column = None,
                 activity_column = None,
                 duration_column = None,
                 start_timestamp_column = None,
                 end_timestamp_column = None,
                 miner_type = 'DFG',
                ):
        """
           log_path should contain a path to csv or excel file with 3 columns: phases, activity name and timestamp
        -------
           Optional you can

           """

        if log_path.split('.')[-1] == 'csv':
            raw_data = pd.read_csv(log_path, nrows=nrows)
        elif log_path.split('.')[-1] in ['xlsx', 'xls']:
            raw_data = pd.read_excel(log_path, nrows=nrows)
        else:
            raise ValueError(f"Only 'csv', 'xls(x)' and 'txt' file formats are supported, "
                             f"but given file path ends with '{log_path.split('.')[-1]}'")

        self.event_id_column = event_id_column
        self.activity_column = activity_column
        self.miner_type = miner_type
        self.duration_column = duration_column
        self.start_timestamp_column = start_timestamp_column
        self.end_timestamp_column = end_timestamp_column

        self.data = self.data_preprocessing(raw_data)

        self.df = pm4py.format_dataframe(self.data, case_id='event id', activity_key='Activity name',
                                         timestamp_key='Timestamp')

    @staticmethod
    def neighbour_duplicates(self, data):
        while (data[self.activity_column].diff() == 0).astype(int).sum() != 0:

            data['diff'] = self.data[self.activity_column].diff()
            duplicate = data[data['diff'] == 0].index - 1

            for i in duplicate:
                self.data[self.duration_column][i] = data[self.duration_column][i + 1] + data[self.duration_column][i]
            data = data.drop(index=duplicate + 1)
            data = data.drop(columns=['diff'])
        return data


  #  def get_unique_phases(self):
#
   #     return self.data[self.event_id].unique()


    def data_preprocessing(self, data):
        data = self.neighbour_duplicates(data)
        return data

    def visualize_graph(self, miner_type = self.miner_type, nx_graph=False):
        if miner_type == 'HeuristicNet':
            heu_model = pm4py.discover_heuristics_net(self.df)
            pm4py.view_heuristics_net(heu_model)

        if miner_type == 'PetriNetAlphaPlus':
            net4, im4, fm4 = pm4py.discover_petri_net_alpha_plus(self.df)
            pm4py.view_petri_net(net4, im4, fm4)
        if miner_type == 'DFG':
            dfg, start_activities, end_activities = pm4py.discover_dfg(self.df)
            if  nx_graph:
                G = nx.DiGraph()
                time = np.round_(self.data['Time, h (in grains)'].values, 2)
                for k, w in zip(dfg.keys(), time):
                    G.add_edge(k[0], k[1], weight=w)

                self.networkx_graph(G)

            else:
                pm4py.view_dfg(dfg, start_activities, end_activities)

    @staticmethod
    def networkx_graph(G):
        plt.subplots(1, 1, figsize=(15, 10))

        elarge = [(u, v) for (u, v, d) in G.edges(data=True)]  # if d["weight"] > 0.5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True)]  # if d["weight"] <= 0.5]

        pos = nx.spring_layout(G, seed=100, iterations=10)  # positions for all nodes - seed for reproducibility

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='r')  # cmap='plasma')
        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=8, edge_color='black')
        nx.draw_networkx_edges(
            G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="black", style="dashed")

        # node labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.001)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def get_unique_activities(self):
        activities = pm4py.algo.filtering.log.attributes.attributes_filter.get_attribute_values(self.df, "concept:name")
        return activities

    def check_or_calc_duration(self):
        """
        Calculates duration if it is not calculated.
        """
        if self.duration_column is None:
            dframe = self.data
            id_column = self.id_column
            start_timestamp_column = self.start_timestamp_column
            end_timestamp_column = self.end_timestamp_column
            duration_column = 'duration'
            if start_timestamp_column is None and end_timestamp_column is None:
                raise RuntimeError('Cannot calculate time difference, '
                                   'because both "start_timestamp_column" and "end_timestamp_column" are None.')
            elif start_timestamp_column is not None and end_timestamp_column is not None:
                dframe[duration_column] = dframe[end_timestamp_column] - dframe[start_timestamp_column]
            else:
                if self.start_timestamp_column is not None and self.end_timestamp_column is None:
                    start_timestamp_col = dframe[start_timestamp_column]
                    end_timestamp_col = dframe[start_timestamp_column].shift(-1)
                    start_id_col = dframe[id_column]
                    end_id_col = dframe[id_column].shift(-1)
                else:
                    start_timestamp_col = dframe[end_timestamp_column].shift(1)
                    end_timestamp_col = dframe[end_timestamp_column]
                    start_id_col = dframe[id_column].shift(1)
                    end_id_col = dframe[id_column]
                dframe[duration_column] = end_timestamp_col - start_timestamp_col
                different_id_mask = start_id_col != end_id_col
                dframe.loc[different_id_mask, duration_column] = None

            dframe[duration_column] = dframe[duration_column] / pd.Timedelta(seconds=1)
            self.data = dframe
            self.duration_column = duration_column