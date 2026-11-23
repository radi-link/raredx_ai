# Disease–Gene Link Prediction — 모델링 결과 v2 (early stopping + regularization)

상태: `MODELING_V2_REGULARIZED`
학습·평가일: 2026-10-21 / 팀 공유 준비일: 2026-10-21
브랜치: `feature/dg-link-prediction-modeling` (PR [#11](https://github.com/fabxoe/raredx_ai/pull/11))

v1(`docs/data/modeling_results_v1.md`)에서 지적한 과적합 문제를 다룬 두 번째 실행입니다. v1은 고정 50 epoch, early stopping·정규화 없이 돌렸는데, 진단해보니 validation loss가 epoch 20~40대에 최저점을 찍고 이후 계속 나빠지는 전형적인 과적합 곡선이었습니다 (train positive가 6,673개뿐이라 모델 파라미터 대비 데이터가 적은 게 근본 원인). 이번엔 세 가지를 추가했습니다.

## v1 대비 변경 사항 (`app/modeling/train.py`, `scripts/train_dg_link_model.py`)

- **Early stopping**: `patience` 파라미터 추가. Validation loss 기준으로 `patience` epoch 동안 개선이 없으면 멈추고, 가장 좋았던 epoch의 가중치로 복원합니다. `TrainingHistory.best_epoch`로 어느 epoch이 선택됐는지 항상 기록됩니다. CLI 기본값: `--epochs 150 --patience 15` (`--patience 0`으로 끄면 v1과 동일하게 전체 epoch을 다 돌고 마지막 가중치를 반환합니다).
- **Weight decay (L2)**: `Adam`에 `weight_decay` 인자 추가. CLI 기본값 `--weight-decay 1e-4`.
- **GNN 모델 축소**: `hidden_dim`/`embedding_dim` 기본값 64 → 32. 파라미터 수를 데이터 규모에 더 맞췄습니다. (baseline은 원래도 32라 변화 없음.)

## 전달 방식

- GitHub 브랜치: 위 코드 변경 (`ddab09b`), 신규 테스트 8개, 전체 258개 테스트 통과.
- 별도 Drive 폴더: [05.07_ai_modeling-ai 모델링](https://drive.google.com/drive/folders/1bMJLq-aQhGafGkd2u2BjP6CtbmJ4ndpU) 아래 새 하위 폴더 [Modeling_Results_v2_2026-10-21](https://drive.google.com/drive/folders/1bmiY1NXMeUZwcmtCRLmSEvzkK8D9mXY3)에 업로드 완료: `dg_link_prediction_report_v2.json`, `baseline_state_dict.pt`, `gnn_state_dict.pt`.

## 재현 방법

```bash
git checkout feature/dg-link-prediction-modeling   # 또는 main에 머지된 뒤 main
uv sync --extra dev
uv run python scripts/train_dg_link_model.py \
  --graph-dir data/processed/graph \
  --seed 42 \
  --epochs 150 \
  --patience 15 \
  --weight-decay 1e-4 \
  --output reports/dg_link_prediction_report.json \
  --model-dir reports/models
```

## 결과 비교 (validation/test, combined)

| 모델 | Split | 지표 | v1 (과적합) | v2 (early stop + L2 + 작은 모델) | 배율 |
|---|---|---|---:|---:|---:|
| gnn | validation | MRR | 0.0037 | **0.0187** | ×5.1 |
| gnn | validation | Hits@10 | 0.0108 | **0.0406** | ×3.8 |
| gnn | validation | Hits@50 | 0.0158 | **0.0904** | ×5.7 |
| gnn | test | MRR | 0.0024 | **0.0158** | ×6.6 |
| gnn | test | Hits@50 | 0.0093 | **0.0868** | ×9.3 |
| baseline | validation | MRR | 0.0016 | 0.0018 | ×1.1 |
| baseline | test | MRR | 0.0018 | 0.0022 | ×1.2 |

**학습 길이**: baseline은 epoch 45에서 멈췄고(최적 epoch 30), GNN은 epoch 102에서 멈췄습니다(최적 epoch 87) — v1의 고정 50 epoch보다 GNN이 실제로는 더 오래, 더 유용하게 학습했습니다. weight decay가 과적합 시점을 늦춰줬기 때문입니다.

**솔직한 평가**: 절대 수치는 여전히 낮습니다(후보 유전자 ~4,500개 중 1개를 맞히는 과제라 원래 어렵습니다). 하지만 이제 **GNN이 baseline보다 MRR 기준 약 8~9배 높아서, 그래프 구조에서 실제로 뭔가 배우고 있다는 신호가 처음으로 명확히 보입니다.** v1에서는 이 차이가 거의 없었습니다(과적합 때문에 GNN의 이점이 묻혀 있었음).

## 읽는 법 / 주의사항

v1 문서(`docs/data/modeling_results_v1.md`)의 "읽는 법" 섹션이 그대로 적용됩니다 (`combined` vs `all_combined`, 양방향 filtered ranking, cold-start 처리 등). 추가로:

- 리포트 JSON에 `training` 블록이 새로 생겼습니다: `{"baseline": {"epochs_run", "best_epoch", "best_validation_loss"}, "gnn": {...}}` — 몇 epoch에서 멈췄고 어느 epoch 가중치가 저장됐는지 투명하게 보여줍니다.
- `--patience 0`을 주면 v1과 동일하게(early stopping 없이) 돌아갑니다 — 비교 실험이 필요하면 이 옵션을 쓰세요.

## 지금도 남아있는 것 (v1에서 동일)

- idf/ablation 가중치 엣지 feature 미사용.
- cold-start 완화책 미적용 — 여전히 분리해서 "보여주는" 선에서 그침 (gene 쪽 cold-start 비율 약 40%).

재현 코드·정책: 이 브랜치의 `app/modeling/`, `docs/superpowers/specs/2026-10-21-dg-link-prediction-modeling-design.md`.
