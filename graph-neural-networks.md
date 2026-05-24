# Graph Neural Networks (GNNs)

Overview
- GNNs operate on graph-structured data to learn node, edge, or whole-graph representations for prediction tasks.

Important subtopics
- Message passing networks (GCN, GAT)
- Node classification, link prediction, graph classification
- Graph sampling and scalability (GraphSAGE, Cluster-GCN)

Key notes
- Represent relations explicitly; choose aggregation functions (mean, sum, max) based on task.
- Be mindful of oversmoothing and the receptive field as depth increases.

Quick example (node classification)
- Train a simple GCN with adjacency + node features to predict node labels on a citation graph.

Mermaid pipeline
```mermaid
flowchart LR
  A[Graph (nodes+edges)] --> B[Compute node features]
  B --> C[Message passing layers]
  C --> D[Readout / classifier]
  D --> E[Predictions]
```

Notes on images
- Add a small graph embedding t-SNE at `images/gnn_embeddings.png`.
