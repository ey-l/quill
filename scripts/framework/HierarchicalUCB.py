import sys
import os

ppath = sys.path[0] + '/../../'
sys.path.append(os.path.join(ppath, 'code'))
sys.path.append(os.path.join(ppath, 'code', 'framework'))

from import_packages import *
from discretizers import *
from SearchSpace import *
from utils import *
from framework_utils import *

class Edge():
    def __init__(self, _s, _e):
        self._s = _s
        self._e = _e

    def __repr__(self):
        return f"({self.__class__.__name__}) {repr(self._s)} -> {repr(self._e)}"

class Node:
    def __init__(self, ID):
        self.ID = int(ID)
        self.count = 0
        self.value = 0.0

    def __repr__(self):
        return f"({self.__class__.__name__}) {self.ID}"

class IntermediateNode(Node):
    def __init__(self, ID):
        super().__init__(ID)
        # Only have clusterID if it is an actual cluster
        self.clusterID = None

    def __repr__(self):
        return f"({self.__class__.__name__}) {self.ID} (clusterID {self.clusterID}) ({self.count}, {self.value})"

class LeafNode(Node):
    def __init__(self, ID, candidate):
        """
        x: float, semantic distance
        y: float, utility
        """
        super().__init__(ID)
        self.KLDiv = candidate.KLDiv
        self.l2_norm = candidate.l2_norm
        self.gpt_distance = candidate.gpt_distance
        self.utility = candidate.utility

    def __repr__(self):
        return f"({self.__class__.__name__}) {self.ID} ({self.KLDiv}, {self.l2_norm}, {self.gpt_distance}, {self.utility})"

class HierarchicalUCB():
    def __init__(self, root):
        self.root = root
        self.explored_nodes = []

    def __repr__(self):
        return f"({self.__class__.__name__}) {self.root}"
    
    def initialize(self, Z, cluster_assigns:List, search_space):
        self.nodes = {}
        self.edges = []
        self.clusters = {}
        self.cluster_nodes = []
        self.root = None
        for c in range(len(np.unique(cluster_assigns))):
            node_ids = np.where(np.array(cluster_assigns) == c-1)[0]
            #cluster = [search_space.candidates[i] for i in cluster_indices]
            self.clusters[c] = list(node_ids)

        #print(self.clusters)    
        self._build(Z, search_space)
        self._assign_cluster_nodes()
    
    def _get_all_successors(self, node):
        # Get all successors of a node, for all levels of the tree
        successors = []
        for edge in self.edges:
            if edge._s == node:
                if isinstance(edge._e, LeafNode): successors.append(edge._e)
                #successors.append(edge._e)
                successors.extend(self._get_all_successors(edge._e))
        return successors
    
    def _get_all_predecessors(self, node):
        # Get all predecessors of a node, for all levels of the tree
        predecessors = []
        for edge in self.edges:
            if edge._e == node:
                predecessors.append(edge._s)
                predecessors.extend(self._get_all_predecessors(edge._s))
        return predecessors

    def _assign_cluster_nodes(self):
        node_successors = {}
        for node in self.nodes.values():
            if isinstance(node, LeafNode): continue
            node_successors[node.ID] = self._get_all_successors(node)
        #print(node_successors)
        for cluster, node_ids in self.clusters.items():
            # compare if two lists are identical, despite of the order
            for node_id, node in self.nodes.items():
                if isinstance(node, LeafNode): continue
                if set([n.ID for n in node_successors[node_id]]) == set(node_ids):
                    node.clusterID = cluster
                    self.cluster_nodes.append(node)
                    break
        print("Cluster assignment done!")
    
    def initialize_ucb(self, n_samples=5):
        """
        To initialize, we randomly sample n_samples of leaf nodes
        and update the affected intermediate nodes's ucb value
        """
        #sampled_clusters = np.random.choice(self.cluster_nodes, n_samples)
        for cluster in self.cluster_nodes:
            self._explore_one_node_in_list(self.clusters[cluster.clusterID])
    
    def _explore_one_node_in_list(self, ls:List):
        """
        Explore one node in the list of nodes.
        :param ls: List of node IDs
        """
        explored_ids = [n.ID for n in self.explored_nodes]
        if set(ls) <= set(explored_ids): 
            print("All points in cluster sampled!")
            # TODO: Sample the second highest UCB cluster
            return None

        while True:
            sampled_node_id = np.random.choice(ls)
            if sampled_node_id not in explored_ids: break
        sampled_node = self.nodes[sampled_node_id]
        self.explored_nodes.append(sampled_node)
        print(f"Sampled node: {sampled_node}")
        sampled_node.count += 1
        # This is actuall the reward
        sampled_node.value += sampled_node.utility + sampled_node.l2_norm
        # Update intermediate nodes
        self._update_intermediate_nodes(sampled_node)
        return sampled_node

    def _get_children(self, node):
        children = []
        for edge in self.edges:
            if edge._s == node:
                children.append(edge._e)
        return children
    
    def _select_node(self, current_node):
        """
        Compare the two children at each tree level.
        Keep going down the branch with a higher UCB value 
        until we reach either an intermediate node with a clusterID not None or an intermediate node with count == 0.
        """
        nodes = []
        while True:
            nodes.append(current_node)
            if current_node.clusterID is not None:
                sampled_node = self._explore_one_node_in_list(self.clusters[current_node.clusterID])
                break
            if current_node.count == 0:
                ls = self._get_all_successors(current_node)
                ls = [n.ID for n in ls]
                sampled_node = self._explore_one_node_in_list(ls)
                break

            children = self._get_children(current_node)
            subtotal_count = sum([c.count for c in children])
            ucb_values = [c.value + 2*np.sqrt(2*np.log(subtotal_count)/float(c.count)) for c in children]
            # Both children are intermediate nodes
            if isinstance(children[0], IntermediateNode) and isinstance(children[1], IntermediateNode):
                if ucb_values[0] > ucb_values[1]:
                    current_node = children[0]
                else:
                    current_node = children[1]
            #else: print("???")
            elif isinstance(children[0], IntermediateNode) and isinstance(children[1], LeafNode):
                current_node = children[0]
            elif isinstance(children[1], IntermediateNode) and isinstance(children[0], LeafNode):
                current_node = children[1]
            else: 
                print("???")
                sampled_node = None
                break
        
        #nodes = nodes[:-1] # remove the last node -- the one that has been explored
        return sampled_node, nodes
    
    def explore_a_valid_node(self, current_node=None):
        """
        Explore a valid node in the tree.
        """
        if current_node is None: current_node = self.root
        sampled_node, nodes = self._select_node(current_node)
        done_explored_node = nodes.pop()
        # All LeafNodes in the most desired cluster have been explored; so we backtrack
        while sampled_node is None:
            current_node = nodes.pop()
            other_child = self._get_children(current_node)
            other_child = [c for c in other_child if c.ID != done_explored_node.ID][0]
            print(f"Backtracked to node: {current_node}")
            print(f"Other child: {other_child}")
            # Get the other child of the parent node
            sampled_node, _ = self._select_node(other_child)
            print(f"Sampled node: {sampled_node}")
            if sampled_node is None: done_explored_node = current_node
        return sampled_node, nodes
    
    def _update_intermediate_nodes(self, child):
        """
        Update intermediate nodes's ucb value recursively.
        """
        parent = self._get_parent(child)
        while parent is not None:
            parent.count += 1
            n = parent.count
            value = parent.value
            reward = child.value
            parent.value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
            print(f"Updated parent: {parent}")
            child = parent
            parent = self._get_parent(parent)
    
    def _get_parent(self, node):
        for edge in self.edges:
            if edge._e == node:
                return edge._s
        return None

    def _build(self, Z, search_space):
        intermediate_node_id = len(search_space.candidates)
        # Create nodes from Z
        for i in range(len(Z)):
            # 3 cases: both are leaf nodes, both are intermediate nodes, one is leaf and the other is intermediate
            if Z[i][0] < len(search_space.candidates) and Z[i][1] < len(search_space.candidates):
                # Both are leaf nodes
                left_node = LeafNode(Z[i][0],search_space.candidates[int(Z[i][0])])
                right_node = LeafNode(Z[i][1],search_space.candidates[int(Z[i][1])])
                intermediate_node = IntermediateNode(intermediate_node_id)
                self.nodes[Z[i][0]] = left_node
                self.nodes[Z[i][1]] = right_node
                self.nodes[intermediate_node_id] = intermediate_node
                intermediate_node_id += 1
                # Create edges
                left_edge = Edge(intermediate_node, left_node)
                right_edge = Edge(intermediate_node, right_node)
                self.edges.append(left_edge)
                self.edges.append(right_edge)

            elif Z[i][0] >= len(search_space.candidates) and Z[i][1] >= len(search_space.candidates):
                # Both are intermediate nodes
                intermediate_node = IntermediateNode(intermediate_node_id)
                self.nodes[intermediate_node_id] = intermediate_node
                intermediate_node_id += 1
                # Create edges
                left_edge = Edge(intermediate_node, self.nodes[Z[i][0]])
                right_edge = Edge(intermediate_node, self.nodes[Z[i][1]])
                self.edges.append(left_edge)
                self.edges.append(right_edge)
            else:
                # One is leaf and the other is intermediate
                if Z[i][0] < len(search_space.candidates):
                    # Left node is leaf
                    left_node = LeafNode(Z[i][0],search_space.candidates[int(Z[i][0])])
                    intermediate_node = IntermediateNode(intermediate_node_id)
                    self.nodes[Z[i][0]] = left_node
                    self.nodes[intermediate_node_id] = intermediate_node
                    intermediate_node_id += 1
                    # Create edge
                    left_edge = Edge(intermediate_node, left_node)
                    right_edge = Edge(intermediate_node, self.nodes[Z[i][1]])
                    self.edges.append(left_edge)
                    self.edges.append(right_edge)
                else:
                    # Right node is leaf
                    right_node = LeafNode(Z[i][1],search_space.candidates[int(Z[i][1])])
                    intermediate_node = IntermediateNode(intermediate_node_id)
                    self.nodes[Z[i][1]] = right_node
                    self.nodes[intermediate_node_id] = intermediate_node
                    intermediate_node_id += 1
                    # Create edge
                    left_edge = Edge(intermediate_node, self.nodes[Z[i][0]])
                    right_edge = Edge(intermediate_node, right_node)
                    self.edges.append(left_edge)
                    self.edges.append(right_edge)

        # Set root node
        self.root = self.nodes[intermediate_node_id-1]



if __name__ == '__main__':
    dataset = 'pima'
    use_case = 'modeling'
    # read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()

    for attr in attributes:
        f_quality = []
        f_runtime = []
        # load experiment data
        if attr == "BMI":
            data = pd.read_csv(os.path.join(ppath, 'experiment_data', dataset, use_case, f'{attr}.csv'))
            ss = TestSearchSpace(data)
            break
    
    semantic_metric = 'l2_norm'
    search_space = ss
    datapoints, gt_pareto_points, points_df = get_pareto_front(ss.candidates, semantic_metric)

    parameters = {'t': 0.5, 'criterion': 'distance'}
    t = parameters['t']
    criterion = parameters['criterion']
    X = np.array([p.distribution for p in search_space.candidates])
    X = pairwise_distance(X, metric=wasserstein_distance)
    Z = linkage(X, method='ward')
    #fig = plt.figure(figsize=(25, 10))
    #dn = dendrogram(Z, color_threshold=t)
    agg_clusters = fcluster(Z, t=t, criterion=criterion)
    agg_clusters = [x-1 for x in agg_clusters] # 0-indexing
    #print(Z)

    avg_distance_results = []
    for round in range(1):
        # Create dendrogram
        dendrogram = HierarchicalUCB(None)
        dendrogram.initialize(Z, agg_clusters, search_space)
        print(dendrogram.root)
        print(dendrogram.cluster_nodes)

        # Initialize UCB
        dendrogram.initialize_ucb()
        for i in range(10):
            print(f"Round {i}")
            print(dendrogram.explore_a_valid_node())
        #print(dendrogram.nodes)
        explored_nodes = dendrogram.explored_nodes
        explored_points, est_pareto_points, _ = get_pareto_front(explored_nodes, semantic_metric)
        print(est_pareto_points)
        avg_dist = average_distance(gt_pareto_points, est_pareto_points, debug=True)
        avg_distance_results.append(avg_dist)

        # Sort the points for plotting
        gt_pareto_points = sorted(gt_pareto_points, key=lambda x: x[0])
        est_pareto_points = sorted(est_pareto_points, key=lambda x: x[0])
        # Plot the Pareto front
        gt_pareto_points = np.array(gt_pareto_points)
        est_pareto_points = np.array(est_pareto_points)
        #datapoints = np.array(datapoints)
        # Set size of the plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(explored_points[0], explored_points[1], c='gray', label='Explored Points', marker='x',)
        ax.plot(gt_pareto_points[:, 0], gt_pareto_points[:, 1], '+-', c='red', label='Ground Truth')
        ax.plot(est_pareto_points[:, 0], est_pareto_points[:, 1], 'x-', c='green', label='Estimated')
        ax.legend(bbox_to_anchor=(1, 1),ncol=3)
        ax.set_xlabel('Semantic Distance', fontsize=14)
        ax.set_ylabel('Utility', fontsize=14)
        ax.set_title('Pareto Curve Estimated vs. Ground-Truth', fontsize=14)

        fig.savefig(os.path.join(ppath, 'code', 'plots', f'HUCB_{attr}_{round}.png'), bbox_inches='tight')
    
    # plot the average distance as a boxplot
    fig, ax = plt.subplots()
    ax.boxplot(avg_distance_results)
    ax.set_xlabel('UCB')
    ax.set_ylabel('Average Distance')
    fig.savefig(os.path.join(ppath, 'code', 'plots', f'HUCB_{attr}_boxplot.png'), bbox_inches='tight')