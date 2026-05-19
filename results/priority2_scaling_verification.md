# Priority 2 Scaling Verification

## binary
- Total latency by N: N=100: 0.00610636 ms, N=500: 0.0107971 ms, N=1000: 0.0133922 ms
- Rescore latency by N: N=100: 1e-12 ms, N=500: 1e-12 ms, N=1000: 1e-12 ms
- Accuracy by N: N=100: ndcg=0.9101, jaccard=1.0000, N=500: ndcg=0.6899, jaccard=0.5964, N=1000: ndcg=0.6581, jaccard=0.5438

## float32_avx2
- Total latency by N: N=100: 0.00291647 ms, N=500: 0.0232931 ms, N=1000: 0.0389088 ms
- Rescore latency by N: N=100: 1e-12 ms, N=500: 1e-12 ms, N=1000: 1e-12 ms
- Accuracy by N: N=100: ndcg=0.7196, jaccard=1.0000, N=500: ndcg=0.9999, jaccard=0.9998, N=1000: ndcg=1.0000, jaccard=1.0000

## two_step_RF10
- Total latency by N: N=100: 0.00901219 ms, N=500: 0.0382994 ms, N=1000: 0.0703125 ms
- Rescore latency by N: N=100: 0.00330588 ms, N=500: 0.0195699 ms, N=1000: 0.0393968 ms
- Accuracy by N: N=100: ndcg=0.9101, jaccard=1.0000, N=500: ndcg=0.9999, jaccard=0.9998, N=1000: ndcg=1.0000, jaccard=1.0000

## two_step_mf_RF10
- Total latency by N: N=100: 0.035593 ms, N=500: 0.158019 ms, N=1000: 0.313006 ms
- Rescore latency by N: N=100: 0.028201 ms, N=500: 0.140913 ms, N=1000: 0.284742 ms
- Accuracy by N: N=100: ndcg=0.9924, jaccard=1.0000, N=500: ndcg=0.9783, jaccard=0.9729, N=1000: ndcg=0.9866, jaccard=0.9836

## Checklist
- Float32 and binary linearity should be assessed from the table above; no conclusion is forced by this script.
- Two-step Step 1 dominance should be assessed by comparing binary scan plus candidate selection against rescore plus final top-k.
- Unexpected trends should be reported in `docs/revision_experiment_results_summary.md` after the full run.
