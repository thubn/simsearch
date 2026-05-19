# Priority 2 Scaling Verification

## binary
- Total latency by N: N=100: 0.005685 ms, N=200: 0.00576612 ms, N=313: 0.00629785 ms
- Rescore latency by N: N=100: 1e-12 ms, N=200: 1e-12 ms, N=313: 1e-12 ms
- Accuracy by N: N=100: ndcg=1.0000, jaccard=1.0000, N=200: ndcg=1.0000, jaccard=1.0000, N=313: ndcg=1.0000, jaccard=1.0000

## float32_avx2
- Total latency by N: N=100: 0.00312004 ms, N=200: 0.00732205 ms, N=313: 0.0114383 ms
- Rescore latency by N: N=100: 1e-12 ms, N=200: 1e-12 ms, N=313: 1e-12 ms
- Accuracy by N: N=100: ndcg=0.7028, jaccard=1.0000, N=200: ndcg=1.0000, jaccard=1.0000, N=313: ndcg=1.0000, jaccard=1.0000

## two_step_RF10
- Total latency by N: N=100: 0.00876514 ms, N=200: 0.0152445 ms, N=313: 0.0234781 ms
- Rescore latency by N: N=100: 0.00311458 ms, N=200: 0.00690473 ms, N=313: 0.0128801 ms
- Accuracy by N: N=100: ndcg=1.0000, jaccard=1.0000, N=200: ndcg=0.3759, jaccard=0.3072, N=313: ndcg=0.2511, jaccard=0.1905

## two_step_mf_RF10
- Total latency by N: N=100: 0.0365065 ms, N=200: 0.0645217 ms, N=313: 0.100628 ms
- Rescore latency by N: N=100: 0.0294557 ms, N=200: 0.0563881 ms, N=313: 0.0902926 ms
- Accuracy by N: N=100: ndcg=0.7114, jaccard=1.0000, N=200: ndcg=0.3759, jaccard=0.3072, N=313: ndcg=0.2511, jaccard=0.1905

## Checklist
- Float32 and binary linearity should be assessed from the table above; no conclusion is forced by this script.
- Two-step Step 1 dominance should be assessed by comparing binary scan plus candidate selection against rescore plus final top-k.
- Unexpected trends should be reported in `docs/revision_experiment_results_summary.md` after the full run.
