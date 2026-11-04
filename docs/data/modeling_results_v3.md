# Disease–Gene Link Prediction — 모델링 결과 v3 (DistMult + 재튜닝)

상태: `MODELING_V3_DISTMULT_RETUNED`
학습·평가일: 2026-10-21 / 팀 공유 준비일: 2026-10-21
브랜치: `feature/dg-link-prediction-modeling` (PR [#11](https://github.com/fabxoe/raredx_ai/pull/11))

v2(`docs/data/modeling_results_v2.md`)에서 early stopping + weight decay + 작은 GNN(dim=32)으로 과적합을 줄였습니다. 이번엔 디코더 자체를 더 가볍게 바꾸고, 그 변화에 맞춰 하이퍼파라미터를 다시 찾았습니다.

## 전달 방식

- 별도 Drive 폴더: [05.07_ai_modeling-ai 모델링](https://drive.google.com/drive/folders/1bMJLq-aQhGafGkd2u2BjP6CtbmJ4ndpU) 아래 새 하위 폴더 [Modeling_Results_v3_2026-10-21](https://drive.google.com/drive/folders/1w8QI362SaNkzXiq7bCtEllf4z7kWnstJ)에 업로드 완료: `dg_link_prediction_report_v6_final.json`, `baseline_state_dict.pt`, `gnn_state_dict.pt`.

## v2 대비 변경 사항

- **디코더: full bilinear → DistMult**. `BaselineScorer`/`GNNLinkPredictor` 둘 다 `disease_emb @ W @ gene_emb`(`embedding_dim²`개 파라미터)를 `sum(disease_emb * relation_weight * gene_emb)`(`embedding_dim`개 파라미터, 대각 관계 벡터)로 교체. Stanford CS224W류 지식그래프 임베딩 기법 중 하나로, 파라미터를 구조적으로 줄여 과적합을 억제하려는 시도입니다.
- **GNN `hidden_dim`/`embedding_dim`: 32 → 64로 복귀**. 디코더가 가벼워지니 노드 임베딩 자체는 다시 64차원이 더 잘 맞았습니다 — 실제 데이터로 grid sweep해서 확인했습니다.
- **CLI 기본값: `--epochs 150→400`, `--patience 15→20`**. baseline이 150 epoch로는 한참 덜 학습된 상태였고(epoch 287 근방에서야 수렴), DistMult 교체 이후 더 두드러졌습니다.

## 재현 방법

```bash
git checkout feature/dg-link-prediction-modeling
uv sync --extra dev
uv run python scripts/train_dg_link_model.py \
  --graph-dir data/processed/graph \
  --seed 42 \
  --output reports/dg_link_prediction_report.json \
  --model-dir reports/models
# (--epochs/--patience/--weight-decay 등은 이제 CLI 기본값 그대로 두면 됩니다)
```

## 결과 비교 (validation/test, combined)

| 모델 | Split | 지표 | v2 | v3 | 배율 |
|---|---|---|---:|---:|---:|
| baseline | validation | MRR | 0.0018 | **0.0125** | ×7.0 |
| baseline | test | MRR | 0.0022 | **0.0118** | ×5.4 |
| gnn | validation | MRR | 0.0187 | **0.0199** | ×1.1 |
| gnn | validation | Hits@50 | 0.0904 | **0.0813** | ×0.9 |
| gnn | test | MRR | 0.0158 | **0.0182** | ×1.2 |
| gnn | test | Hits@50 | 0.0868 | **0.0842** | ×1.0 |

**학습 길이**: baseline은 epoch 307에서 멈췄습니다(최적 epoch 287) — v2의 고정 150 epoch로는 수렴 전이었다는 뜻입니다. GNN은 epoch 97에서 멈췄습니다(최적 epoch 77).

## 상세 분해: cold-start 포함 + 방향별(gene/disease) 분리

위 표는 cold-start 쌍을 제외한 `combined` 지표만 보여줘서, 두 가지가 가려져 있었습니다. PR 리뷰 중 지적받아 추가합니다 ([PR #11 코멘트](https://github.com/fabxoe/raredx_ai/pull/11#issuecomment-6345334462)).

| 모델 | Split | 집합 | 방향 | MRR | Hits@1 | Hits@10 | Hits@50 | n |
|---|---|---|---|---:|---:|---:|---:|---:|
| baseline | validation | cold-start 제외 | gene (질병→유전자) | 0.0099 | 0.0019 | 0.0253 | 0.0663 | 513 |
| baseline | validation | cold-start 제외 | disease (유전자→질병) | 0.0145 | 0.0029 | 0.0346 | 0.0938 | 693 |
| baseline | validation | **전체(cold-start 포함)** | gene | 0.0064 | 0.0012 | 0.0156 | 0.0420 | 833 |
| baseline | validation | **전체(cold-start 포함)** | disease | 0.0121 | 0.0024 | 0.0288 | 0.0780 | 833 |
| **gnn** | validation | cold-start 제외 | **gene (질병→유전자)** | **0.0013** | 0.0000 | 0.0019 | 0.0039 | 513 |
| **gnn** | validation | cold-start 제외 | disease (유전자→질병) | **0.0336** | 0.0159 | 0.0649 | 0.1385 | 693 |
| gnn | validation | **전체(cold-start 포함)** | gene | 0.0039 | 0.0000 | 0.0072 | 0.0312 | 833 |
| gnn | validation | **전체(cold-start 포함)** | disease | 0.0281 | 0.0132 | 0.0540 | 0.1152 | 833 |
| **gnn** | test | cold-start 제외 | **gene (질병→유전자)** | **0.0012** | 0.0000 | 0.0000 | 0.0000 | 502 |
| **gnn** | test | cold-start 제외 | disease (유전자→질병) | **0.0306** | 0.0102 | 0.0715 | 0.1460 | 685 |

(전체 JSON은 Drive의 `dg_link_prediction_report_v6_final.json`에 `gene_corruption`/`disease_corruption`/`all_gene_corruption`/`all_disease_corruption` 키로 들어있습니다.)

**중요한 비대칭 발견**: `combined`(gene+disease 평균)만 보면 GNN MRR 0.02 정도로 보이지만, **방향을 나눠보면 전혀 다른 그림입니다.** "유전자→질병" 방향(disease_corruption, 유전자가 주어졌을 때 질병 후보를 맞히는)은 MRR 0.03대로 baseline(0.0145)보다 확실히 낫습니다. 그런데 **"질병→유전자" 방향(gene_corruption — 임상적으로 더 중요한, 증상에서 원인 유전자를 찾는 방향)은 GNN MRR이 0.0012~0.0013으로 사실상 0에 가깝고, baseline(0.0099~0.0107)보다 오히려 더 나쁩니다.** test 셋에서는 Hits@1/10/50이 전부 0입니다.

즉 지금까지 "GNN이 baseline보다 낫다"고 한 결론(결론 2, 아래)은 **disease_corruption 방향이 평균을 끌어올려서 생긴 착시일 수 있습니다.** 실제 쓰임새(증상/표현형 → 원인 유전자 추정)에 더 가까운 gene_corruption 방향만 보면 GNN이 더 나쁩니다.

### 원인 조사 결과: 구조적 degree 비대칭

cold-start를 제외한 쌍만 비교한 건데도 이 비대칭이 나와서, "cold-start가 많아서"만으로는 설명이 안 됩니다. 대신 **메시지 패싱 그래프에서 Disease와 Gene이 받는 1-hop 신호량 자체가 크게 다릅니다**:

| 노드 타입 | 받는 관계 | 평균 degree |
|---|---|---:|
| Disease | `rev_HAS_PHENOTYPE`(19.2) + `rev_ASSOCIATED_WITH`(1.1) | **20.3** |
| Gene | `ASSOCIATED_WITH`뿐 | **1.5** |

Gene은 오직 예측 대상인 바로 그 관계(Disease-Gene)로만 정보를 받는데, Disease는 그거 말고도 표현형 정보를 19배 더 많이 직접(1-hop) 받습니다. 저장된 체크포인트(`reports/models_v6_final/gnn_state_dict.pt`)로 직접 확인한 결과:

- **임베딩 분화도**: Gene 임베딩끼리 평균 코사인 유사도 0.72(더 뭉쳐있음) vs Disease 0.66.
- **점수 분산** (validation 쌍 1개로 전체 후보 대비 점수 분포 확인): gene 후보 점수 표준편차 0.32 vs disease 후보 1.15 — gene 쪽 점수가 훨씬 좁게 뭉쳐서 순위를 가르기 어렵습니다.

**결론**: 버그가 아니라 그래프 자체의 정보 비대칭이 그대로 반영된 겁니다. 그리고 **고치는 방법이 이슈 [#19](https://github.com/fabxoe/raredx_ai/issues/19)(cold-start)와 정확히 같습니다** — Gene-HPO 엣지가 추가되면 Gene도 Disease처럼 1-hop 직접 정보를 받게 돼서, cold-start뿐 아니라 이 비대칭도 같이 해결될 가능성이 높습니다.

**솔직한 평가**: baseline이 크게 좋아졌고(v2에서 "덜 학습됐다"는 지적이 맞았던 셈), GNN은 소폭 개선됐습니다. 다만 GNN의 Hits@50은 오히려 살짝 내려갔습니다. **단, 위 표는 전부 seed 42 한 번의 결과입니다 — 바로 아래 "Seed 분산 확인" 섹션을 보면 v2→v3의 이 소폭 차이들(bilinear→DistMult, hits@50 하락 등)은 seed 분산 안에 들어가 통계적으로 확정하기 어렵습니다.** 유일하게 seed에 안정적으로 재현되는 결론은 "GNN이 baseline보다 낫다"는 것입니다.

## Seed 분산 확인 (가장 중요한 섹션)

이 문서의 숫자, 그리고 v1~v3에서 비교해온 하이퍼파라미터 조합들은 전부 **seed 42 고정 한 번**의 결과였습니다. split(`load_graph_package(..., seed=42)`)은 고정한 채 모델 자체의 seed(초기화 + 학습 중 dropout/epoch별 supervision 샘플링)만 1~5로 바꿔서 5회 반복한 결과:

| 모델 | 지표 | 평균 | 표준편차 | 상대 변동폭 |
|---|---|---:|---:|---:|
| baseline | validation MRR | 0.0110 | ±0.0034 | ~31% |
| baseline | test MRR | 0.0111 | ±0.0035 | ~32% |
| gnn | validation MRR | 0.0191 | ±0.0021 | ~11% |
| gnn | test MRR | 0.0170 | ±0.0024 | ~14% |

**결론 1 — 지금까지의 미세 튜닝은 대부분 신뢰할 수 없습니다.** bilinear vs DistMult, dim 32 vs 64, weight_decay 값 사이의 MRR 차이(0.001~0.002 단위)는 이 seed 분산보다 작거나 비슷한 크기입니다. 즉 "DistMult+dim64 조합이 제일 좋다"는 단일 seed=42 결과만으로는 통계적으로 확정할 수 없고, seed 운이었을 가능성을 배제할 수 없습니다. 이후 비슷한 수준의 미세 튜닝은 권장하지 않습니다 — 효과를 보려면 애초에 multi-seed로 비교해야 합니다.

**결론 2 — 그러나 "GNN이 baseline보다 낫다"는 신뢰할 수 있습니다.** 5개 seed 전부에서 GNN의 validation MRR이 baseline보다 높았습니다(baseline 0.0069~0.0160 vs GNN 0.0167~0.0219, 구간이 거의 안 겹침). 평균도 baseline 0.0110±0.0034, GNN 0.0191±0.0021로 명확히 분리됩니다. **이번 세션 전체에서 가장 과학적으로 신뢰할 수 있는 단일 결론은 "그래프 구조(메시지 패싱)가 실제로 예측에 도움이 된다"는 것**이며, 이는 seed 하나가 아니라 5개 seed에 걸쳐 일관되게 관찰됐습니다.

**⚠️ 단서(아래 "상세 분해" 섹션 참고)**: 이 결론 2는 gene+disease 양방향 평균(`combined`) 기준입니다. 방향을 나눠보면 GNN이 나은 건 disease_corruption(유전자→질병) 방향뿐이고, **임상적으로 더 중요한 gene_corruption(질병→유전자) 방향에서는 GNN이 오히려 baseline보다 나쁩니다.** 이 seed 분산 분석은 gene/disease를 나눠서 다시 해봐야 "GNN이 낫다"는 결론이 방향별로도 유지되는지 확인할 수 있습니다 — 아직 안 했습니다.

**실무 권장**: 앞으로 이 파이프라인으로 성능을 보고할 때는 seed 하나의 숫자 대신 최소 3~5 seed의 평균±표준편차를 리포트하세요. `load_graph_package`의 split seed(42, 고정)와 모델 seed(가변)는 구분해서 각각의 역할을 명확히 하는 게 좋습니다.

## 시도했다가 되돌린 것

- **Cold-start 공유 fallback 임베딩**: 그래프에 엣지가 전혀 없는 노드에 학습 가능한 공유 벡터를 대신 쓰는 방식을 시도했으나, 실제 데이터에서 cold-start 지표가 거의 안 움직였고 전체 지표는 오히려 나빠져서 되돌렸습니다. Cold-start는 여전히 구조적 한계로 남아있습니다(gene 쪽 약 40%).

## 읽는 법 / 주의사항

v1(`docs/data/modeling_results_v1.md`), v2 문서의 "읽는 법" 섹션이 그대로 적용됩니다. 체크포인트의 파라미터 이름이 `bilinear_weight` → `relation_weight`로 바뀌었으니, 과거 체크포인트(v1/v2)와 이번 체크포인트는 서로 `load_state_dict` 호환이 안 됩니다.

재현 코드·정책: 이 브랜치의 `app/modeling/`, `docs/superpowers/specs/2026-10-21-dg-link-prediction-modeling-design.md`.
