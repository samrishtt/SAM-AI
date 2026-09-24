# Empirical Benchmark Results: Micro-AGI


### Empirical Ablation Results Matrix

| Architecture Configuration | ARC-AGI Acc (%) | Deduction Acc (%) | Overall Acc (%) | Avg Tree Nodes | Latency (ms) |
|:---------------------------|:---------------:|:-----------------:|:---------------:|:--------------:|:------------:|
| System 1 Baseline (Greedy Prior) |          100.0% |            100.0% |          100.0% |            1.8 |        3.72 |
| System 2 Pure Search (MCTS, N=15) |          100.0% |            100.0% |          100.0% |            9.0 |        2.12 |
| Full Micro-AGI (MCTS N=30 + CLS Memory) |          100.0% |            100.0% |          100.0% |           10.5 |        1.31 |
