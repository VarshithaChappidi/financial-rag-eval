Questions: 34

| Config | Hit@1 | Hit@5 | MRR@5 | Avg latency (ms) |
|---|---|---|---|---|
| dense only | 0.50 | 0.82 | 0.62 | 23 |
| BM25 only | 0.29 | 0.76 | 0.47 | 37 |
| hybrid (RRF) | 0.56 | 0.82 | 0.66 | 68 |
| hybrid, no company filter | 0.53 | 0.76 | 0.61 | 52 |
| hybrid + reranker | 0.65 | 0.97 | 0.77 | 1392 |
