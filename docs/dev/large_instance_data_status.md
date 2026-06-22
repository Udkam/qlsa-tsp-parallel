# 大规模 TSPLIB95 实例数据状态

- 数据目录：`data`
- 已检查 tier：L1, L2, L3
- 已存在实例数：21
- 缺失实例数：2
- Inventory CSV：`results\final\large_instance_inventory.csv`

## 实例清单

| tier | instance | exists | dimension | edge_weight_type | BKS | status |
|---|---|---:|---:|---|---:|---|
| L1 | ch130 | true | 130 | EUC_2D | 6110 | present |
| L1 | ch150 | true | 150 | EUC_2D | 6528 | present |
| L1 | kroA150 | true | 150 | EUC_2D | 26524 | present |
| L1 | d198 | true | 198 | EUC_2D | 15780 | present |
| L1 | tsp225 | true | 225 | EUC_2D | 3916 | present |
| L1 | pr226 | true | 226 | EUC_2D | 80369 | present |
| L1 | gil262 | true | 262 | EUC_2D | 2378 | present |
| L1 | a280 | true | 280 | EUC_2D | 2579 | present |
| L2 | lin318 | true | 318 | EUC_2D | 42029 | present |
| L2 | rd400 | true | 400 | EUC_2D | 15281 | present |
| L2 | pcb442 | true | 442 | EUC_2D | 50778 | present |
| L2 | d493 | true | 493 | EUC_2D | 35002 | present |
| L2 | att532 | true | 532 | ATT | 27686 | present |
| L2 | u574 | true | 574 | EUC_2D | 36905 | present |
| L2 | rat575 | true | 575 | EUC_2D | 6773 | present |
| L2 | d657 | true | 657 | EUC_2D | 48912 | present |
| L2 | u724 | true | 724 | EUC_2D | 41910 | present |
| L2 | rat783 | true | 783 | EUC_2D | 8806 | present |
| L3 | pr1002 | false | 1002 |  | 259045 | missing |
| L3 | dsj1000 | true | 1000 | CEIL_2D | 18659688 | present_check_bks_edge_weight_type |
| L3 | si1032 | false | 1032 |  | 92650 | missing |
| L3 | u1060 | true | 1060 | EUC_2D | 224094 | present |
| L3 | vm1084 | true | 1084 | EUC_2D | 239297 | present |

## 数据准备说明

本脚本不依赖网络，也不会自动下载数据。若某个实例缺失，请从 TSPLIB95 镜像或课程允许的数据源手动下载对应 `.tsp` 文件，并放入 `data/` 目录后重新运行本脚本。

特别注意：`dsj1000` 的 BKS 与 `EDGE_WEIGHT_TYPE` 有关；如果本地文件为 `CEIL_2D` 或其它格式，分析时需要核对对应 BKS。
