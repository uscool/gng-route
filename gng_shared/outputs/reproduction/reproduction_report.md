## Table 4: routing performance (test split, n = 10557)
| Router | Area | Area50 | QNC | vs paper |
|---|---|---|---|---|
| GNG-shared | 0.7853 | 0.3808 | 0.7214 | MATCH |
| UniRoute | 0.7818 | 0.3773 | 0.7491 | MATCH |
| KNN | 0.7785 | 0.3760 | 1.0000 | MATCH |
| GNG-isolated | 0.7775 | 0.3776 | 1.0000 | MATCH |
| SVM | 0.7503 | 0.3605 | 1.0000 | MATCH |
| MLP | 0.6826 | 0.3169 | 1.0000 | CHECK |
| GNG-shared (decayed updates) | 0.7831 | 0.3789 | 0.7426 | MATCH |

## Table 5: bootstrap over 1000 resamples
| Router | Area (mean [95% CI]) | Reaches a_max | paper area / reach |
|---|---|---|---|
| GNG-shared | 0.7852 [0.7794, 0.7906] | 99% | 0.7852 / 99% |
| UniRoute | 0.7817 [0.7760, 0.7872] | 92% | 0.7817 / 92% |
| KNN | 0.7783 [0.7730, 0.7836] | 10% | 0.7783 / 10% |
| GNG-isolated | 0.7774 [0.7715, 0.7827] | 0% | 0.7774 / 0% |
| SVM | 0.7502 [0.7439, 0.7561] | 0% | 0.7502 / 0% |
| MLP | 0.6825 [0.6757, 0.6888] | 0% | 0.6812 / 0% |
| GNG-shared (decayed updates) | 0.7829 [0.7773, 0.7886] | 79% | 0.7829 / 79% |

## Paired differences, GNG-shared minus router (area)
| Router | mean diff | 95% CI |
|---|---|---|
| UniRoute | +0.0035 | [+0.0015, +0.0056] |
| KNN | +0.0068 | [+0.0041, +0.0094] |
| GNG-isolated | +0.0078 | [+0.0052, +0.0102] |
| SVM | +0.0350 | [+0.0309, +0.0393] |
| MLP | +0.1027 | [+0.0972, +0.1086] |
| GNG-shared (decayed updates) | +0.0022 | [+0.0003, +0.0043] |

QNC difference vs UniRoute when both reach a_max: -0.071 (n = 921 resamples)

## GNG-shared over 5 training seeds
| seed | nodes | edges | area | area50 | qnc |
|---|---|---|---|---|---|
| 0 | 107 | 661 | 0.7853 | 0.3808 | 0.7214 |
| 1 | 107 | 672 | 0.7858 | 0.3818 | 0.7232 |
| 2 | 108 | 787 | 0.7872 | 0.3822 | 0.5667 |
| 3 | 105 | 665 | 0.7850 | 0.3813 | 0.7311 |
| 4 | 111 | 753 | 0.7847 | 0.3808 | 0.7239 |

area mean 0.7856, SD 0.0010 (paper: 0.7856, 0.0010); QNC range 0.567 to 0.731
