import numpy as np
import random
from scipy.spatial import KDTree
from scipy.optimize import minimize

class Node:
    def __init__(self, config, parent=None):
        self.config = np.array(config)  # [kappa1, phi1, ell1, ..., kappaN, phiN, ellN]
        self.parent = parent
        self.cost = 0 if parent is None else parent.cost + np.linalg.norm(self.config - parent.config)

class RRTStar:
    def __init__(self, start, goal, kappa_limits=[[0,10],[0,20]], phi_limits = [[-4,4],[-4,4]], ell_limits=[[0.25, 0.392],[0.20,0.392]], max_iter=1000, step_size=0.05, radius=0.1):
        self.start = Node(start)
        self.goal = Node(goal)
        self.kappa_limits = kappa_limits
        self.phi_limits = phi_limits
        self.ell_limits = ell_limits
        self.max_iter = max_iter
        self.step_size = step_size
        self.radius = radius
        self.tree = [self.start]

    # def sample_config(self):
    #     # Ensure each list has exactly two elements
    #     assert len(self.kappa_limits) == 2, "kappa_limits should have exactly two elements"
    #     assert len(self.phi_limits) == 2, "phi_limits should have exactly two elements"
    #     assert len(self.ell_limits) == 2, "ell_limits should have exactly two elements"

    #     # Flatten the limits into a single list
    #     limits = self.kappa_limits + self.phi_limits + self.ell_limits

    #     # Generate a sample configuration with the correct dimension
    #     sampled_values = [random.uniform(k[0], k[1]) for k in limits]

    #     return np.array(sampled_values)

    def sample_config(self):
        return np.array([
            random.uniform(k[0], k[1]) for k in self.kappa_limits + self.phi_limits + self.ell_limits
        ])

    # def nearest_node(self, sample):
    #     nodes = np.array([node.config for node in self.tree])
    #     tree = KDTree(nodes)
    #     print(sample)
    #     _, index = tree.query(sample)
    #     return self.tree[index]
    
    def nearest_node(self, sample):
        nodes = np.array([node.config for node in self.tree])  # (n, 6)
        print(nodes)
        tree = KDTree(nodes)
        print(sample, tree)
        KDTree()
        _, index = tree.query(sample.reshape(1, -1))  # Reshape to (1, 6) for a single sample
        return self.tree[index]


    def steer(self, from_node, to_sample):
        direction = to_sample - from_node.config
        direction = direction / np.linalg.norm(direction) * self.step_size
        new_config = np.clip(from_node.config + direction, 
                             np.concatenate(self.kappa_limits + self.phi_limits + self.ell_limits)[:, 0],
                             np.concatenate(self.kappa_limits + self.phi_limits + self.ell_limits)[:, 1])
        return new_config

    def collision_free(self, config):
        # Placeholder: Implement obstacle collision checking
        return True

    def near_nodes(self, new_node):
        nodes = np.array([node.config for node in self.tree])
        tree = KDTree(nodes)
        indices = tree.query_ball_point(new_node.config, self.radius)
        return [self.tree[i] for i in indices]

    def choose_parent(self, new_node, near_nodes):
        best_parent = new_node.parent
        min_cost = new_node.cost
        for node in near_nodes:
            temp_cost = node.cost + np.linalg.norm(new_node.config - node.config)
            if temp_cost < min_cost and self.collision_free(new_node.config):
                min_cost = temp_cost
                best_parent = node
        new_node.parent = best_parent
        new_node.cost = min_cost

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            temp_cost = new_node.cost + np.linalg.norm(new_node.config - node.config)
            if temp_cost < node.cost and self.collision_free(node.config):
                node.parent = new_node
                node.cost = temp_cost

    def extract_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append(node.config)
            node = node.parent
        return path[::-1]

    def plan(self):
        for _ in range(self.max_iter):
            sample = self.sample_config()
            print(_, sample)
            nearest = self.nearest_node(sample)
            new_config = self.steer(nearest, sample)
            if not self.collision_free(new_config):
                continue
            new_node = Node(new_config, parent=nearest)
            near_nodes = self.near_nodes(new_node)
            self.choose_parent(new_node, near_nodes)
            self.tree.append(new_node)
            self.rewire(new_node, near_nodes)
            if np.linalg.norm(new_node.config - self.goal.config) < self.step_size:
                return self.extract_path(new_node)
        return None
