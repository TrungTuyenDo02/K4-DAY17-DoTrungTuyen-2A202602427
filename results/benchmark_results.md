# Day 17 Memory Benchmark

compact_threshold_tokens=800, compact_keep_messages=4, profile_confidence_threshold=0.6

## Standard Benchmark
Dataset: `data/conversations.json` | mode: offline

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                2006 |                     18654 |                      0 |                0.3 |                       0 |             0 |
| Advanced |                2227 |                     34160 |                      1 |                1   |                     979 |             0 |

Advanced vs Baseline: agent tokens +11.0%, prompt tokens +83.1%, recall +1.00

## Long-Context Stress Benchmark
Dataset: `data/advanced_long_context.json` | mode: offline

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                 389 |                     23017 |                      0 |                0.3 |                       0 |             0 |
| Advanced |                 469 |                     11412 |                      1 |                1   |                     546 |             4 |

Advanced vs Baseline: agent tokens +20.6%, prompt tokens -50.4%, recall +1.00
