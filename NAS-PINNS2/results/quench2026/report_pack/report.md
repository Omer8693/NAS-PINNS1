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
| group               | scenario                     | method   | run_dir                                                                                         |   layers |   neurons |   param_count | best_stage   |   best_objective |   run_time_seconds |
|:--------------------|:-----------------------------|:---------|:------------------------------------------------------------------------------------------------|---------:|----------:|--------------:|:-------------|-----------------:|-------------------:|
| method_refine5000   | nsga2_lbfgs5000              | nsga2    | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/lbfgs/nsga2/L6_N132    |        6 |       132 |        143283 | lbfgs        |       0.00492846 |          1858.18   |
| baseline_refine5000 | baseline_from_adam_lbfgs5000 | baseline | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/baseline_adam_refine5000/lbfgs/L5_N96      |        5 |        96 |         57942 | lbfgs        |       0.00814861 |          1219.7    |
| method_refine5000   | nsga3_lbfgs5000              | nsga3    | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/lbfgs/nsga3/L6_N141    |        6 |       141 |        163200 | lbfgs        |       0.0151231  |          1742.48   |
| method_refine5000   | bayesian_lbfgs5000           | bayesian | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/lbfgs/bayesian/L5_N121 |        5 |       121 |         91167 | lbfgs        |       0.021403   |          1554.7    |
| pipeline_final      | bayesian_pipeline_final      | bayesian | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/pipeline/bayesian/final/L5_N121            |        5 |       121 |         91167 | adam         |       0.0273101  |           714.937  |
| baseline            | baseline_original            | baseline | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/baseline/L5_N96                            |        5 |        96 |         57942 | lbfgs        |       3.26091    |           647.247  |
| pipeline_final      | nsga2_pipeline_final         | nsga2    | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/pipeline/nsga2/final/L6_N132               |        6 |       132 |        143283 | lbfgs        |       4.31022    |           903.765  |
| pipeline_final      | nsga3_pipeline_final         | nsga3    | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/pipeline/nsga3/final/L6_N141               |        6 |       141 |        163200 | lbfgs        |       5.4517     |           926.457  |
| baseline_refine5000 | baseline_from_adam_pso       | baseline | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/baseline_adam_refine5000/pso/L5_N96        |        5 |        96 |         57942 | adam         |      19.5636     |            14.1951 |
| method_refine5000   | nsga3_pso                    | nsga3    | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/pso/nsga3/L6_N141      |        6 |       141 |        163200 | adam         |      94.1063     |            91.6431 |
| method_refine5000   | nsga2_pso                    | nsga2    | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/pso/nsga2/L6_N132      |        6 |       132 |        143283 | adam         |      97.2477     |            84.6718 |
| method_refine5000   | bayesian_pso                 | bayesian | /home/coder/NAS-PINNS1/NAS-PINNS2/results/quench2026/best_adam_lbfgs5000/pso/bayesian/L5_N121   |        5 |       121 |         91167 | adam         |     148.88       |            65.3704 |

## Note
- `quality_score_vs_best_pct` objective tabanli bir skordur; klasik classification accuracy degildir.