import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# Base Exceptions
class MultiverseError(Exception):
    """Base class for multiverse simulation errors"""
    pass


class InvalidStateError(MultiverseError):
    """Raised when attempting to access invalid universe state"""
    pass


class UniverseNotFoundError(MultiverseError):
    """Raised when target universe cannot be located"""
    pass


@dataclass
class Universe:
    id: int
    timeline: List[str]
    current_state: str
    probability: float

    def __str__(self) -> str:
        """String representation of Universe"""
        return f"Universe(id={self.id}, state={self.current_state}, prob={self.probability:.2f})"


@dataclass
class UniverseNode:
    id: int
    state: str
    parent_id: Optional[int] = None
    children: List[int] = field(default_factory=list)
    probability: float = 1.0

    def __str__(self) -> str:
        """String representation of UniverseNode"""
        return f"Node(id={self.id}, state={self.state}, prob={self.probability:.2f})"


class MultiverseSimulator:
    def __init__(self):
        self.universes: List[Universe] = []
        self.universe_counter = 0
        self.possible_states = ["alpha", "beta", "gamma"]
        self.current_universe_id = None
        self.tracker = MultiverseTracker()
        self.tracker.simulator = self  # Set simulator reference immediately

    def create_universe(self, state: str) -> Universe:
        """Create a new universe with the given state"""
        if state not in self.possible_states:
            raise InvalidStateError(f"Invalid state: {state}")

        self.universe_counter += 1
        universe = Universe(
            id=self.universe_counter,
            timeline=[state],
            current_state=state,
            probability=1.0
        )
        self.universes.append(universe)

        # Set as current if first universe
        if self.current_universe_id is None:
            self.current_universe_id = universe.id

        return universe

    def split_universe(self, universe: Universe) -> List[Universe]:
        """Split a universe into multiple new universes"""
        if not self.validate_universe(universe):
            raise InvalidStateError("Invalid universe for splitting")

        # Create new universes with random states
        available_states = random.sample(self.possible_states, 2)
        new_universes = []

        for state in available_states:
            self.universe_counter += 1
            new_universe = Universe(
                id=self.universe_counter,
                timeline=universe.timeline + [state],
                current_state=state,
                probability=universe.probability * 0.5
            )
            new_universes.append(new_universe)
            self.universes.append(new_universe)

        # Record the split in tracker
        self.tracker.record_split(universe.id, [u.id for u in new_universes])
        return new_universes

    def shift_universe(self, target_state: str) -> Universe:
        """Shift to closest universe matching target state"""
        if not isinstance(target_state, str):
            raise TypeError("Target state must be string")

        if target_state not in self.possible_states:
            raise InvalidStateError(f"Invalid state: {target_state}")

        candidates = [u for u in self.universes if u.current_state == target_state]
        if not candidates:
            raise UniverseNotFoundError(f"No universe found with state: {target_state}")

        best_match = self._find_best_match(candidates)
        self._execute_shift(best_match)
        return best_match

    def get_current_universe(self) -> Optional[Universe]:
        """Get the currently active universe"""
        if self.current_universe_id is None:
            return None
        return next((u for u in self.universes if u.id == self.current_universe_id), None)

    def validate_universe(self, universe: Universe) -> bool:
        """Validate universe integrity"""
        return all([
            isinstance(universe.id, int),
            0 <= universe.probability <= 1,
            all(s in self.possible_states for s in universe.timeline),
            universe.current_state == universe.timeline[-1]
        ])

    def _find_best_match(self, candidates: List[Universe]) -> Universe:
        """Find the best matching universe from candidates"""
        current = self.get_current_universe()
        if current:
            return min(
                candidates,
                key=lambda u: self._timeline_distance(current.timeline, u.timeline)
            )
        return max(candidates, key=lambda u: u.probability)

    def _execute_shift(self, target_universe: Universe) -> None:
        """Execute universe shift with safety checks"""
        if not self.validate_universe(target_universe):
            raise InvalidStateError("Target universe validation failed")

        old_id = self.current_universe_id
        self.current_universe_id = target_universe.id

        if old_id is not None:
            self.tracker.record_shift(old_id, target_universe.id)

    def _timeline_distance(self, t1: List[str], t2: List[str]) -> float:
        """Calculate similarity between timelines"""
        min_len = min(len(t1), len(t2))
        matches = sum(1 for i in range(min_len) if t1[i] == t2[i])
        return 1 - (matches / max(len(t1), len(t2)))


class MultiverseTracker:
    def __init__(self):
        self.universe_graph = nx.DiGraph()
        self.current_position = None
        self.simulator = None

    def record_shift(self, from_id: int, to_id: int):
        """Record universe transition"""
        self.universe_graph.add_edge(from_id, to_id, type='shift')
        self.current_position = to_id

    def record_split(self, parent_id: int, child_ids: List[int]):
        """Record universe branching"""
        for child_id in child_ids:
            self.universe_graph.add_edge(parent_id, child_id, type='split')

    def visualize(self):
        """Generate multiverse map visualization"""
        enhanced_visualize(self.simulator)


class UniverseAnalyzer:
    def __init__(self, simulator: MultiverseSimulator):
        self.simulator = simulator
        self.transition_costs = defaultdict(float)

    def calculate_entropy(self, universe: Universe) -> float:
        """Calculate universe state entropy"""
        state_counts = defaultdict(int)
        for state in universe.timeline:
            state_counts[state] += 1
        probs = [count / len(universe.timeline) for count in state_counts.values()]
        return -sum(p * np.log2(p) for p in probs)

    def find_optimal_path(self, start_id: int, target_state: str) -> List[int]:
        """Find lowest-cost path between universes"""
        G = self.simulator.tracker.universe_graph
        paths = nx.shortest_path(
            G, start_id,
            weight=lambda u, v, d: self._transition_cost(u, v)
        )
        return paths

    def _transition_cost(self, u: int, v: int) -> float:
        """Calculate cost between two universe states"""
        return self.transition_costs[(u, v)]


class MultiverseOptimizer:
    def __init__(self, analyzer: UniverseAnalyzer):
        self.analyzer = analyzer
        self.cache = {}

    def optimize_shift(self, current: Universe, target_state: str) -> Universe:
        """Find optimal universe shift path"""
        key = (current.id, target_state)
        if key in self.cache:
            return self.cache[key]

        candidates = self._get_candidate_paths(current, target_state)
        best_path = min(candidates,
                        key=lambda p: self._path_cost(p))

        self.cache[key] = best_path
        return best_path

    def _path_cost(self, path: List[int]) -> float:
        """Calculate total path transition cost"""
        return sum(self.analyzer.transition_costs[i, j]
                   for i, j in zip(path, path[1:]))

    def _get_candidate_paths(self, current: Universe, target_state: str) -> List[List[int]]:
        """Get all possible paths to target state"""
        # This is a placeholder implementation
        return [[current.id]]


# Visualization helper functions
def create_node_labels(sim: MultiverseSimulator) -> Dict[int, str]:
    """Create detailed labels for each universe node"""
    labels = {}
    for universe in sim.universes:
        prob_pct = f"{universe.probability * 100:.1f}%"
        timeline_str = "→".join(universe.timeline)
        labels[universe.id] = f"U{universe.id}\n{universe.current_state}\n{prob_pct}\n{timeline_str}"
    return labels


def create_edge_colors(graph: nx.DiGraph) -> List[str]:
    """Create colors for edges based on type"""
    return ['blue' if graph[u][v]['type'] == 'shift' else 'green' for u, v in graph.edges()]


def create_node_colors(sim: MultiverseSimulator) -> List[str]:
    """Create colors for nodes based on state"""
    color_map = {
        'alpha': '#FF9999',  # Light red
        'beta': '#99FF99',  # Light green
        'gamma': '#9999FF'  # Light blue
    }
    # Only get colors for nodes that exist in the graph
    return [color_map[sim.universes[i-1].current_state] 
            for i in sim.tracker.universe_graph.nodes()]
def enhanced_visualize(sim: MultiverseSimulator):
    """Generate an enhanced multiverse map visualization"""
    print(f"Graph nodes: {list(sim.tracker.universe_graph.nodes())}")
    print(f"Universe IDs: {[u.id for u in sim.universes]}")
    plt.figure(figsize=(15, 10))

    # Create layout
    pos = nx.spring_layout(sim.tracker.universe_graph, k=2, iterations=50)

    # Get edge colors
    edge_colors = create_edge_colors(sim.tracker.universe_graph)

    # Draw edges with different styles
    nx.draw_networkx_edges(sim.tracker.universe_graph, pos,
                           edge_color=edge_colors,
                           width=2,
                           arrowsize=20,
                           edge_cmap=plt.cm.Blues)

    # Draw nodes with custom colors and sizes
    node_colors = create_node_colors(sim)
    nx.draw_networkx_nodes(sim.tracker.universe_graph, pos,
                           node_color=node_colors,
                           node_size=2500,
                           alpha=0.7)

    # Add labels with universe information
    labels = create_node_labels(sim)
    nx.draw_networkx_labels(sim.tracker.universe_graph, pos,
                            labels,
                            font_size=8,
                            font_weight='bold')

    # Add title and legend
    plt.title("Multiverse Navigation Map", fontsize=16, pad=20)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', label='Universe Shift'),
        plt.Line2D([0], [0], color='green', label='Universe Split'),
        plt.scatter([0], [0], color='#FF9999', label='Alpha State'),
        plt.scatter([0], [0], color='#99FF99', label='Beta State'),
        plt.scatter([0], [0], color='#9999FF', label='Gamma State')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    # Add grid and style
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    # Show plot
    plt.show()


def main():
    # Create a simulator
    print("Creating simulator...")
    sim = MultiverseSimulator()

    # Create initial universes
    print("\nCreating initial universes...")
    u1 = sim.create_universe("alpha")
    print(f"Created universe 1: {u1}")
    u2 = sim.create_universe("beta")
    print(f"Created universe 2: {u2}")

    # Create multiple splits to make visualization more interesting
    print("\nCreating multiple universe splits...")
    splits = []
    for _ in range(3):
        new_universes = sim.split_universe(random.choice(sim.universes))
        splits.extend(new_universes)
        print(f"Created split: {new_universes}")

    # Perform some shifts
    print("\nPerforming universe shifts...")
    for state in ["alpha", "beta", "gamma"]:
        try:
            new_universe = sim.shift_universe(state)
            print(f"Shifted to {state} state: {new_universe}")
        except MultiverseError as e:
            print(f"Could not shift to {state}: {e}")

    # Visualize the multiverse
    print("\nGenerating enhanced multiverse visualization...")
    sim.tracker.visualize()

    # Create analyzer and optimizer
    print("\nAnalyzing universe properties...")
    analyzer = UniverseAnalyzer(sim)

    # Print entropy for each universe
    for universe in sim.universes:
        entropy = analyzer.calculate_entropy(universe)
        print(f"Universe {universe.id} entropy: {entropy:.3f}")


if __name__ == "__main__":
    main()