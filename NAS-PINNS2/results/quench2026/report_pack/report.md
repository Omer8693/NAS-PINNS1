# Quench2026 Report Pack

## Files
- `tables/quality_summary.csv`
- `tables/runtime_summary.csv`
- `tables/pipeline_stage_matrix.csv`
- `tables/refine5000_stage_matrix.csv`
- `plots/pipeline_stage_heatmap_log10.png`
- `plots/refine5000_stage_heatmap_log10.png`
- `plots/search_pareto_cloud.png`
- `plots/runtime_vs_objective.png`
- `plots/adam_convergence_curves.png`
- `tables/fair_same_arch_per_seed.csv`
- `tables/fair_same_arch_aggregate.csv`
- `plots/fair_same_arch_objective_boxplot.png`
- `plots/fair_same_arch_runtime_bar.png`

## Top Result
- best method: `nsga2` | arch `L6_N132` | best_after_refine_obj `0.00492846`

## Quality Table
| method   | best_architecture   | pipeline_best_stage   |   pipeline_best_obj |   seed_adam_obj |   lbfgs5000_obj |   pso_obj | best_after_refine_stage   |   best_after_refine_obj |   improvement_vs_pipeline_pct |   rank_after_refine |   quality_score_vs_best_pct |
|:---------|:--------------------|:----------------------|--------------------:|----------------:|----------------:|----------:|:--------------------------|------------------------:|------------------------------:|--------------------:|----------------------------:|
| nsga2    | L6_N132             | lbfgs                 |           4.31022   |         97.2477 |      0.00492846 |   809.894 | lbfgs                     |              0.00492846 |                       99.8857 |                   1 |                     100     |
| baseline | L5_N96              | lbfgs                 |           3.26091   |         19.5636 |      0.00814861 |  3260.53  | lbfgs                     |              0.00814861 |                       99.7501 |                   2 |                      60.482 |
| nsga3    | L6_N141             | lbfgs                 |           5.4517    |         94.1063 |      0.0151231  |  2025.56  | lbfgs                     |              0.0151231  |                       99.7226 |                   3 |                      32.589 |
| bayesian | L5_N121             | adam                  |           0.0273101 |        148.88   |      0.021403   |   864.708 | lbfgs                     |              0.021403   |                       21.6297 |                   4 |                      23.027 |

## Runtime Table
| group               | scenario                     | method     | run_dir                                                                                         |   layers |   neurons |   param_count | best_stage   |   best_objective |   run_time_seconds | arch    |
|:--------------------|:-----------------------------|:-----------|:------------------------------------------------------------------------------------------------|---------:|----------:|--------------:|:-------------|-----------------:|-------------------:|:--------|
| method_refine5000   | nsga2_lbfgs5000              | nsga2      | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/lbfgs/nsga2/L6_N132    |        6 |       132 |        143283 | lbfgs        |       0.00492846 |          1858.18   | nan     |
| baseline_refine5000 | baseline_from_adam_lbfgs5000 | baseline   | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/baseline_adam_refine5000/lbfgs/L5_N96      |        5 |        96 |         57942 | lbfgs        |       0.00814861 |          1219.7    | nan     |
| method_refine5000   | nsga3_lbfgs5000              | nsga3      | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/lbfgs/nsga3/L6_N141    |        6 |       141 |        163200 | lbfgs        |       0.0151231  |          1742.48   | nan     |
| fair_same_arch      | fair_seed44_lbfgs            | fixed_arch |                                                                                                 |        6 |       132 |           nan | lbfgs        |       0.0164979  |          1439.32   | L6_N132 |
| fair_same_arch      | fair_seed43_lbfgs            | fixed_arch |                                                                                                 |        6 |       132 |           nan | lbfgs        |       0.0187972  |          1429.3    | L6_N132 |
| fair_same_arch      | fair_seed42_lbfgs            | fixed_arch |                                                                                                 |        6 |       132 |           nan | lbfgs        |       0.0208041  |          1466.15   | L6_N132 |
| method_refine5000   | bayesian_lbfgs5000           | bayesian   | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/lbfgs/bayesian/L5_N121 |        5 |       121 |         91167 | lbfgs        |       0.021403   |          1554.7    | nan     |
| pipeline_final      | bayesian_pipeline_final      | bayesian   | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/pipeline/bayesian/final/L5_N121            |        5 |       121 |         91167 | adam         |       0.0273101  |           714.937  | nan     |
| baseline            | baseline_original            | baseline   | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/baseline/L5_N96                            |        5 |        96 |         57942 | lbfgs        |       3.26091    |           647.247  | nan     |
| pipeline_final      | nsga2_pipeline_final         | nsga2      | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/pipeline/nsga2/final/L6_N132               |        6 |       132 |        143283 | lbfgs        |       4.31022    |           903.765  | nan     |
| pipeline_final      | nsga3_pipeline_final         | nsga3      | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/pipeline/nsga3/final/L6_N141               |        6 |       141 |        163200 | lbfgs        |       5.4517     |           926.457  | nan     |
| fair_same_arch      | fair_seed42_adam             | fixed_arch |                                                                                                 |        6 |       132 |           nan | adam         |       8.33141    |           875.379  | L6_N132 |
| baseline_refine5000 | baseline_from_adam_pso       | baseline   | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/baseline_adam_refine5000/pso/L5_N96        |        5 |        96 |         57942 | adam         |      19.5636     |            14.1951 | nan     |
| fair_same_arch      | fair_seed43_adam             | fixed_arch |                                                                                                 |        6 |       132 |           nan | adam         |      46.2702     |           868.575  | L6_N132 |
| method_refine5000   | nsga3_pso                    | nsga3      | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/pso/nsga3/L6_N141      |        6 |       141 |        163200 | adam         |      94.1063     |            91.6431 | nan     |
| method_refine5000   | nsga2_pso                    | nsga2      | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/pso/nsga2/L6_N132      |        6 |       132 |        143283 | adam         |      97.2477     |            84.6718 | nan     |
| method_refine5000   | bayesian_pso                 | bayesian   | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/pso/bayesian/L5_N121   |        5 |       121 |         91167 | adam         |     148.88       |            65.3704 | nan     |
| fair_same_arch      | fair_seed44_adam             | fixed_arch |                                                                                                 |        6 |       132 |           nan | adam         |     182.688      |           850.783  | L6_N132 |
| fair_same_arch      | fair_seed44_pso              | fixed_arch |                                                                                                 |        6 |       132 |           nan | pso          |     772.034      |            23.1715 | L6_N132 |
| fair_same_arch      | fair_seed42_pso              | fixed_arch |                                                                                                 |        6 |       132 |           nan | pso          |    4584.99       |            23.3553 | L6_N132 |
| fair_same_arch      | fair_seed43_pso              | fixed_arch |                                                                                                 |        6 |       132 |           nan | pso          |    4939.19       |            23.0983 | L6_N132 |

## Fair Same-Architecture (Strict) Per-Seed
|   seed | arch    |   layers |   neurons |   adam_obj |   lbfgs_obj |   pso_obj |   adam_runtime_s |   lbfgs_runtime_s |   pso_runtime_s |   lbfgs_better_than_adam |   pso_better_than_adam |
|-------:|:--------|---------:|----------:|-----------:|------------:|----------:|-----------------:|------------------:|----------------:|-------------------------:|-----------------------:|
|     42 | L6_N132 |        6 |       132 |    8.33141 |   0.0208041 |  4584.99  |          875.379 |           1466.15 |         23.3553 |                        1 |                      0 |
|     43 | L6_N132 |        6 |       132 |   46.2702  |   0.0187972 |  4939.19  |          868.575 |           1429.3  |         23.0983 |                        1 |                      0 |
|     44 | L6_N132 |        6 |       132 |  182.688   |   0.0164979 |   772.034 |          850.783 |           1439.32 |         23.1715 |                        1 |                      0 |

## Fair Same-Architecture Aggregate
| stage   |   objective_mean |   objective_std |   runtime_mean_s |   runtime_std_s |   improve_vs_adam_pct | arch    |
|:--------|-----------------:|----------------:|-----------------:|----------------:|----------------------:|:--------|
| adam    |       79.0966    |     74.87       |         864.912  |       10.3699   |                0      | L6_N132 |
| lbfgs   |        0.0186997 |      0.00175936 |        1444.92   |       15.5547   |               99.9764 | L6_N132 |
| pso     |     3432.07      |   1886.48       |          23.2084 |        0.108119 |            -4239.08   | L6_N132 |

## Note
- `quality_score_vs_best_pct` objective tabanli bir skordur; klasik classification accuracy degildir.