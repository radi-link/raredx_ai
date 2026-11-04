# Data + Knowledge Graph v1 — 모델링 인수인계

상태: `DATA_KNOWLEDGE_GRAPH_GOAL_REACHED`
Graph 구축·검증일: 2026-10-07 / 팀 공유 준비일: 2026-10-12
브랜치: `feature/data-knowledge-graph`

실제 source 기반 데이터와 Disease–Gene link split을 준비했습니다. 모델 학습·성능 평가는 아직 수행하지 않았습니다. 기존 runtime ranking과 Neo4j pipeline은 유지합니다.

## 전달 방식

- GitHub 브랜치: builder, 정책, source manifest, tests 및 synthetic fixtures.
- 별도 압축 파일: 실제 processed graph, split, QC, source manifest 및 인수인계 문서.
- 팀 데이터 폴더: [Data_Knowledge_Graph_v1_2026-10-12](https://drive.google.com/drive/folders/1tkUnELdJktpQYmoFMG75BawPNR4xn9Y_).
- 실제 raw/processed 데이터는 Git에 포함하지 않습니다. 사용자 지정 Drive의 새 하위 폴더로 데이터 패키지를 전달하며 기존 공유 권한을 유지합니다.

로컬 전달 파일:

`data/processed/handoff/raredx-data-knowledge-graph-v1-20261012.tar.gz`

압축 파일에는 `graph/`, `manifests/sources.json`, `docs/data/`, `package_manifest.json`이 들어갑니다. Raw source, 환자 데이터, credentials, model weights는 포함하지 않습니다. `package_manifest.json`은 builder commit, graph/split 버전, source manifest hash, 파일별 SHA-256을 기록합니다. 함께 생성하는 `.sha256` 파일은 압축 파일 자체를 검증합니다.

## 모델링에 사용할 파일

| 파일 | 사용 목적 |
|---|---|
| `graph/hpo_nodes.csv`, `disease_nodes.csv`, `gene_nodes.csv` | Canonical node universe와 attributes |
| `graph/training_message_edges.csv` | **학습용 graph**: IS_A + positive Disease-HPO + train Disease-Gene |
| `graph/splits/train.csv` | 학습 target Disease-Gene pairs |
| `graph/splits/validation.csv` | 모델·설정 선택용 target pairs |
| `graph/splits/test.csv` | 최종 평가용 target pairs |
| `graph/splits/*_with_reverse.csv` | 반대 방향을 사용하는 모델용; 원본과 동일 split |
| `graph/negative_pairs.csv` | 전체 known positive를 제외한 예시 negative pool |
| `graph/splits/idf_fit.json` | Train-only gene DF/IDF/normalization fit |
| `graph/ablations/A_train.csv` | Evidence score를 Disease-Gene weight로 사용 |
| `graph/ablations/B_train.csv` | Train-only IDF penalty를 곱한 Disease-Gene weight |
| `graph/ablations/C_train.csv` | Unweighted Disease-Gene; edge_weight 생략 |
| `graph/graph_metadata.json`, `qc_report.json`, `artifact_hashes.json` | Counts, 계약, QC, 파일 해시 |

**`graph_edges.csv`와 `disease_gene_edges.csv`는 full-reference view입니다. Heldout 정답과 full-reference IDF가 포함되므로 그대로 학습 graph/feature로 사용하지 마세요.** A/B/C 파일은 train Disease-Gene relation의 weight 입력이며, ontology/Disease-HPO context는 공통 training graph에서 가져옵니다. Audit/provenance·정답에서 유래한 필드를 모델 feature로 넣지 않습니다.

Negative pool은 split별로 나뉘어 있지 않습니다. 이를 사용하는 실험은 학습/평가용 negative 분리와 seed를 별도 고정하고 split 간 재사용을 점검해야 합니다. Negative는 확인되지 않은 pair의 표본이며 검증된 생물학적 negative가 아닙니다.

## CSV 해석

- ID는 문자열입니다. Disease=`ORPHA:<integer>`, Gene=`HGNC:<integer>`, HPO=`HP:<7 digits>`.
- 배열/dict 셀은 JSON 문자열입니다. `json.loads`로 해석합니다.
- Boolean 셀은 `true`/`false`, NULL은 빈 셀입니다. 문자열 `false`의 truthiness로 판정하지 않습니다.
- Unresolved gene은 `ORPHANET_LOCUS:<source_key>`, `rankable=false`입니다. Default target에서 제외되어 있습니다.
- IS_A 방향은 child → parent입니다. 다른 방향이 필요한 모델은 relation 정의를 명시합니다.
- Gene-HPO canonical edge는 0입니다. NCBI–HGNC identity mapping이 미확보이며 source rows는 audit에 보존했습니다.

## Counts와 평가 범위

| 항목 | 수 |
|---|---:|
| HPO / Disease / Gene nodes | 19,810 / 5,996 / 4,552 |
| 전체 nodes / edges | 30,358 / 147,768 |
| IS_A / HAS_PHENOTYPE / ASSOCIATED_WITH | 24,329 / 115,098 / 8,341 |
| Train / validation / test DG pairs | 6,673 / 833 / 833 |
| Negative pool | 8,339 |

Task는 **transductive Disease–Gene link prediction**입니다. Canonical pair를 집계한 뒤 seed 42로 분할합니다. 동일 node는 여러 split에 등장할 수 있지만 동일 pair는 겹치지 않습니다. Nonrankable target edge 2개는 split에서 제외합니다.

IS_A는 structural prior, Disease-HPO는 input context입니다. Disease-HPO를 prediction target으로 바꾸는 실험에는 별도 split/leakage 정책이 필요합니다. Patient-case 평가는 graph-edge 평가와 별개이며, 동일 source annotation으로 case와 정답 graph를 만드는 source leakage를 피해야 합니다.

## 패키지를 받은 뒤

```bash
shasum -a 256 -c raredx-data-knowledge-graph-v1-20261012.tar.gz.sha256
tar -xzf raredx-data-knowledge-graph-v1-20261012.tar.gz
cd raredx-data-knowledge-graph-v1
```

다음은 Python 표준 라이브러리만 사용하는 패키지 검증입니다.

```python
import hashlib
import json
from pathlib import Path

root = Path(".")
manifest = json.loads((root / "package_manifest.json").read_text())
for relative, expected in manifest["files"].items():
    with (root / relative).open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == expected, relative
qc = json.loads((root / "graph/qc_report.json").read_text())
assert qc["passed"] and qc["split"]["passed"]
print("Package hashes and QC verified")
```

## Source부터 재생성하기

브랜치를 checkout하고 Python 3.12 프로젝트 환경을 준비합니다. Manifest에 기록된 **동일 source bytes 5개**를 다음 ignored 경로에 배치합니다. 최신 다운로드가 같은 release/bytes라는 보장은 없습니다.

```text
data/raw/hpo/hp.obo
data/raw/hpo/phenotype.hpoa
data/raw/hpo/genes_to_phenotype.txt
data/raw/orphanet/Clinical_signs_and_symptoms_in_rare_disease.xml
data/raw/orphanet/genes_associated_with_rare_disease.xml
```

저장소 root에서:

```bash
uv sync --extra dev
uv run scripts/build_processed.py --canonical-graph --seed 42
uv run pytest -q
```

출력은 `data/processed/graph/`입니다. 원본 source SHA-256을 기존 `manifests/sources.json`과 먼저 대조하고, 산출물은 전달 패키지의 `graph/artifact_hashes.json`과 비교합니다. Builder는 manifest를 다시 생성하므로 원본 bytes 검증 전에 기존 manifest를 덮어쓰지 않습니다.

## 검증과 남은 항목

- 기존 검증: 전체 tests 154 passed, graph/split QC error 0, 두 독립 real build의 36 artifact hash 일치.
- 공유 시 현재 코드의 전체 tests와 source/artifact hash를 다시 검증합니다.
- Missing source: alignment XML, HGNC–NCBI mapping, direct OMIM/DisGeNET raw.
- HPO gene TSV의 release/date 및 일부 license/acquisition metadata는 unknown입니다. 확인되지 않은 license나 외부 재배포 권한을 가정하지 않습니다.
- 압축 파일은 전달용 로컬 패키지입니다. 생성만으로 모델링 담당자에게 업로드/전달된 것은 아닙니다.

상세 계약: [Data dictionary](data_dictionary.md), [Graph schema](graph_schema.md), [Split policy](split_policy.md), [ID mapping](id_mapping_policy.md), [Source manifest](data_source_manifest.md), [QC](qc_spec.md).
