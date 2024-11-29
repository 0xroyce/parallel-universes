# Parallel Universe Simulator

A Python-based simulation of parallel universes with visualization capabilities, demonstrating concepts from quantum mechanics and multiverse theory.

## Overview

This project implements a parallel universe simulator that allows for:
- Creation of new universes with different states
- Universe splitting (similar to quantum decoherence)
- Universe state transitions
- Timeline tracking and entropy calculations
- Interactive visualization of the multiverse structure

## Features

- **Universe Management**
  - Create new universes with specific states
  - Split existing universes into multiple branches
  - Track universe probabilities and timelines

- **State Transitions**
  - Shift between parallel universes
  - Calculate optimal paths between states
  - Track transition history

- **Visualization**
  - Interactive graph visualization
  - Color-coded states and transitions
  - Probability and timeline display
  - Detailed universe information

- **Analysis Tools**
  - Entropy calculations
  - Timeline similarity metrics
  - Path optimization

## Requirements

```python
networkx
matplotlib
numpy
```

## Installation

```bash
# Clone the repository
git clone https://github.com/0xroyce/parallel-universes.git

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Create a simulator instance
sim = MultiverseSimulator()

# Create initial universes
u1 = sim.create_universe("alpha")
u2 = sim.create_universe("beta")

# Split a universe
new_universes = sim.split_universe(u1)

# Shift to a different universe
new_universe = sim.shift_universe("beta")

# Visualize the multiverse
sim.tracker.visualize()
```

## Project Structure

```
parallel-universe-simulator/
├── parallel.py        # Main implementation
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Technical Details

- Universe states are represented as nodes in a directed graph
- Transitions between universes are tracked as edges
- Timeline similarity is calculated using a custom distance metric
- Entropy calculations help measure universe complexity
- Visualization uses networkx and matplotlib

## Origin

This project was created through collaboration with Nova and Atom, AI agents available at [ExParadox](https://exparadox.com). The implementation combines concepts from:
- Quantum mechanics (Many-worlds interpretation)
- Graph theory
- Information theory
- Scientific visualization

## License

This project is open-source and available under the MIT License.

## Credits

- Original concept and implementation by Nova and Atom via ExParadox
- Project organization and development by [@0xroyce](https://github.com/0xroyce)
