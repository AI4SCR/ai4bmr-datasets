import networkx as nx
import numpy as np

from .ContactGraphBuilder import ContactGraphBuilder
from .KNNGraphBuilder import KNNGraphBuilder
from .RadiusGraphBuilder import RadiusGraphBuilder

GRAPH_BUILDERS = {
    "knn": KNNGraphBuilder,
    "contact": ContactGraphBuilder,
    "radius": RadiusGraphBuilder,
}


def build_graph(mask: np.ndarray, topology: str, **kwargs) -> nx.Graph:
    """Build graph from mask using specified topology."""

    if topology not in GRAPH_BUILDERS:
        raise ValueError(
            f"invalid graph topology {topology}. Available topologies are {GRAPH_BUILDERS.keys()}"
        )

    # Instantiate graph builder object
    builder = GRAPH_BUILDERS[topology]()

    # Build graph and get key
    g = builder.build_graph(mask, **kwargs)
    return g
