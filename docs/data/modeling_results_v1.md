# Disease–Gene Link Prediction — 모델링 결과 v1 공유

상태: `MODELING_V1_FIRST_RUN`
학습·평가일: 2026-10-21 / 팀 공유 준비일: 2026-10-21
브랜치: `fabxoe/squirrelfish` (main에 로컬 머지 완료, 2026-10-21)

`docs/data/modeling_handoff.md`에서 인수받은 Data + Knowledge Graph v1 패키지로 Disease–Gene (`ASSOCIATED_WITH`) 링크 예측 베이스라인과 GNN을 처음으로 학습·평가했습니다. 설계/계획 문서는 `docs/superpowers/specs/2026-10-21-dg-link-prediction-modeling-design.md`, `docs/superpowers/plans/2026-10-21-dg-link-prediction-modeling.md`를 참고하세요.

## 전달 방식

- GitHub 브랜치(`main`): `app/modeling/` 코드, 스펙·플랜 문서, tests 및 synthetic fixtures.
- 별도 Drive 폴더: 실제 학습된 모델 가중치(`.pt`)와 평가 리포트(JSON).
- 팀 모델링 폴더: [05.07_ai_modeling-ai 모델링](https://drive.google.com/drive/folders/1bMJLq-aQhGafGkd2u2BjP6CtbmJ4ndpU) 아래 새 하위 폴더 [Modeling_Results_v1_2026-10-21](https://drive.google.com/drive/folders/1vH9n9pCHnpDIzN68xmjlZQ7TgSUeBF9C)에 업로드 완료 (기존 파일과 이름 겹침 없음).

로컬 전달 파일(위 새 Drive 폴더에 그대로 업로드):

```
reports/dg_link_prediction_report.json      # Hits@1/10/50, MRR (baseline/gnn × validation/test)
reports/models/baseline_state_dict.pt       # BaselineScorer state_dict
reports/models/gnn_state_dict.pt            # GNNLinkPredictor state_dict
```

이 파일들은 `.gitignore`(`reports/*.json`, 모델 가중치는 커밋 대상 아님)에 의해 Git에는 포함되지 않습니다 — Drive로만 전달합니다.

## 재현 방법

```bash
git checkout main
uv sync --extra dev
uv run python scripts/train_dg_link_model.py \
  --graph-dir data/processed/graph \
  --seed 42 \
  --epochs 50 \
  --output reports/dg_link_prediction_report.json \
  --model-dir reports/models
```

- `data/processed/graph/`는 `docs/data/modeling_handoff.md`의 패키지를 그대로 복사해 둔 것이어야 합니다 (Git에 포함 안 됨, 팀 Drive에서 받아야 함).
- 기존 리포트/체크포인트를 덮어쓰지 않습니다 — 같은 경로에 파일이 있으면 에러를 내고 멈춥니다. 다시 돌리려면 `--overwrite`를 명시하거나 `--output`/`--model-dir`을 다른 경로로 지정하세요.
- seed 42 고정, CPU 전용.

## 결과 요약

seed 42, epoch 50 (기본값) 기준 첫 실행 결과입니다. 유전자 후보 풀이 약 4,500개 규모라 완전 랜덤 기준선은 MRR ≈ 0.0002, Hits@50 ≈ 0.011 정도입니다.

| 모델 | Split | 집합 | Hits@1 | Hits@10 | Hits@50 | MRR | n |
|---|---|---|---:|---:|---:|---:|---:|
| baseline | validation | combined | 0.0000 | 0.0017 | 0.0141 | 0.0016 | 1206 |
| baseline | validation | all (cold-start 포함) | 0.0000 | 0.0012 | 0.0120 | 0.0015 | 1666 |
| baseline | test | combined | 0.0000 | 0.0025 | 0.0152 | 0.0018 | 1187 |
| baseline | test | all (cold-start 포함) | 0.0006 | 0.0024 | 0.0126 | 0.0021 | 1666 |
| gnn | validation | combined | 0.0008 | 0.0108 | 0.0158 | 0.0037 | 1206 |
| gnn | validation | all (cold-start 포함) | 0.0006 | 0.0090 | 0.0174 | 0.0033 | 1666 |
| gnn | test | combined | 0.0000 | 0.0067 | 0.0093 | 0.0024 | 1187 |
| gnn | test | all (cold-start 포함) | 0.0006 | 0.0066 | 0.0132 | 0.0028 | 1666 |

**솔직한 평가: 두 모델 다 랜덤 기준선 대비 소폭 개선 수준이고, 실용적인 성능은 아닙니다.** GNN이 baseline보다 MRR 기준 약 2배 높긴 하지만(0.0037 vs 0.0016, validation), 둘 다 절대 수치가 낮습니다. "아래 알려진 리스크" 참고 — 기본 설정(50 epoch, lr=0.01)에서 과소학습됐을 가능성이 높습니다. 이 결과를 "모델이 안 된다"는 결론보다는 **파이프라인이 끝까지 정상 작동한다는 확인(첫 구현 목표)**으로 읽어주시고, 하이퍼파라미터 재실험이 다음 단계로 필요합니다.

## 읽는 법 / 주의사항

- **Task(= 예측 대상)**: `(Disease, ASSOCIATED_WITH, Gene)` 링크 존재 여부. 환자 진단 랭킹이 아니라 그래프 엣지 복원 과제입니다.
- **평가 방식**: 양방향(Gene 쪽 corrupt / Disease 쪽 corrupt) filtered ranking. 후보 풀은 `rankable=true`인 해석된 노드만 포함하고, 다른 모든 split의 known positive는 후보에서 제외합니다.
- **`combined` vs `all_combined`**: `combined`는 메시지 패싱 그래프에 엣지가 전혀 없는 cold-start 쌍(그래디언트를 한 번도 못 받은 노드)을 제외한 수치입니다. `all_*` 필드가 cold-start까지 포함한 전체 수치입니다. 두 숫자를 같이 봐야 모델 성능을 오해하지 않습니다.
- **GNN 학습 방법**: 매 epoch마다 train positive 중 일부를 "supervision"으로 떼어내 메시지 패싱 그래프에서 제외한 뒤 학습합니다(자기 정답 엣지를 보고 지름길을 배우는 것을 방지). 그 결과 GNN의 train loss는 baseline의 train loss와 직접 비교할 수 없습니다 — validation/test 쪽 숫자로 비교하세요.
- **알려진 리스크**: 기본 50 epoch / lr=0.01 설정에서 epoch당 보는 positive 예시가 줄어들어(약 30%) GNN이 과소학습됐을 가능성이 있습니다. 실제 데이터로 돌린 결과 초반 loss가 불안정했습니다. 더 많은 epoch이나 다른 lr로 재실험이 필요할 수 있습니다.
- 모델 가중치를 다시 불러오려면: `model.load_state_dict(torch.load(path, weights_only=True))` (baseline/gnn 생성자에 넘긴 `disease_ids`/`gene_ids`/`gene_prior`(baseline) 또는 `metadata`/`num_nodes`(gnn)는 `app/modeling/loader.load_graph_package`로 만든 `GraphPackage`에서 그대로 복원해야 합니다).

## 지금은 포함하지 않은 것

- idf/ablation 가중치 엣지 feature 미사용 (v2에서 검토).
- early stopping 없음, epoch 수는 CLI 인자로 수동 튜닝.
- cold-start 완화책(예: 타입 레벨 fallback 임베딩) 미적용 — v1은 cold-start를 분리해서 "보여주는" 선에서 그침.

재현 코드·정책: 이 브랜치의 `app/modeling/`, `docs/superpowers/specs/2026-10-21-dg-link-prediction-modeling-design.md`.
