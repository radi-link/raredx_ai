# HGNC–NCBI registry mapping — 2026-10-16

사용자 합의: **공식 ID의 양방향 고유 일치만 채택**합니다. 이 단계는 identity mapping 구축이며 canonical graph v2 생성이나 Gene–HPO edge 추가를 수행하지 않습니다.

## GitHub 반영 범위 — v1.1 매핑 보완판

2026-10-18 합의에 따라 **매핑 코드·정책·테스트만** 공개합니다. 여기서 v1.1은 v1에 덧붙이는 매핑 보완판의 범위 이름이며, 원래 v1 데이터나 애플리케이션 패키지 버전을 변경하지 않습니다.

- 포함 파일: `app/etl/gene_registry.py`, `scripts/build_gene_mapping.py`, `tests/test_gene_registry.py`, 이 문서.
- 입력 snapshot 및 생성된 CSV/JSON 데이터는 로컬에 보존하며 이번 Git 변경에 포함하지 않습니다.
- HPO·Orphanet 최신화, withdrawn ID의 대체·병합 정책 적용, v2 후보 및 비교 평가 코드는 별도 작업으로 유지합니다.
- 매핑 빌더는 기존 graph 모듈에 의존하지 않습니다. 아래 v1 graph 경로는 기존 로컬 v1 산출물 기준이며, coverage 입력은 선택 사항입니다. 기존 그래프·노드 ID·train/validation/test는 수정하지 않습니다.

## 범위와 보존

- 기존 `data/raw/hpo/`, `data/raw/orphanet/`, `data/processed/graph/`, `data/processed/handoff/`, `manifests/sources.json`을 보존합니다.
- 새 원본은 `data/raw/gene_registry/2026-10-16/`, 결과는 `data/processed/gene_mapping/2026-10-16_reciprocal_v1/`에 저장합니다.
- `scripts/build_gene_mapping.py`는 로컬 snapshot만 읽습니다. 기존 graph builder를 호출하지 않으며 새 output directory만 허용합니다.
- 별도 manifest를 사용합니다. 기존 v1 manifest의 `HGNC_NCBI_mapping=NOT_AVAILABLE`은 **v1 생성 당시**의 기록으로 그대로 남깁니다.
- 실제 데이터는 기존 `.gitignore` 규칙에 따라 Git에서 제외됩니다. 새 의존성, 모델 학습, 서비스 배포는 없습니다.

## 공식 입력과 snapshot

| 파일 | 공식 출처 | 이번 응답의 Last-Modified (UTC) |
|---|---|---|
| `hgnc_complete_set.txt` | [HGNC complete TSV](https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt) | 2026-10-13 12:32:17 |
| `withdrawn.txt` | [HGNC withdrawn TSV](https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/withdrawn.txt) | 2026-10-13 12:32:21 |
| `Homo_sapiens.gene_info.gz` | [NCBI human gene_info](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz) | 2026-10-16 06:49:14 |
| `gene_history.gz` | [NCBI all-species gene_history](https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_history.gz) | 2026-10-16 06:38:56 |

이 값은 HTTP 메타데이터이며 공식 release 번호로 간주하지 않습니다. `release_version=null`을 유지합니다. 실제 취득 UTC 시각, URL, 응답 ETag/Last-Modified, 파일 크기, SHA-256은 snapshot의 `source_manifest.json`에 기록했습니다. HGNC는 [CC0](https://hgnc.genenames.org/about/license/); NCBI 파일 자체로 확인되지 않은 license는 null입니다. 공개 자료의 제3자 권리까지 임의로 단정하지 않습니다.

원본 history는 모든 종을 포함합니다. 매핑에서는 tax_id `9606`만 검사하며, 다른 종의 폐기 기록이 사람 ID를 차단하지 않도록 분리합니다. [NCBI 파일 형식](https://ftp.ncbi.nlm.nih.gov/gene/README), [HGNC 필드 설명](https://hgnc.genenames.org/help/statistics-and-downloads/).

## 채택 규칙

1. HGNC ID와 NCBI GeneID를 문자열 식별자로 파싱합니다. symbol·alias가 같다는 이유로 연결하지 않습니다. 두 DB의 symbol이 달라도 공식 ID 쌍이 일치하면 채택할 수 있습니다.
2. HGNC complete의 `status=Approved`, 유효한 `entrez_id`, 중복 없는 HGNC record를 요구합니다. 같은 NCBI ID에 여러 승인 HGNC가 연결되면 모두 audit에 남깁니다.
3. HGNC withdrawn 목록에 있는 ID는 채택하지 않습니다. 원본 merger/split 힌트는 audit에 그대로 보존하며 대체 ID로 자동 이관하지 않습니다.
4. NCBI human gene_info에 현재 GeneID가 존재하고, human gene_history의 `Discontinued_GeneID`에 등장하지 않아야 합니다. History의 replacement는 audit용이며 따라가지 않습니다.
5. NCBI `dbXrefs`의 HGNC 토큰을 공백 제거 후 전부 검사합니다. `HGNC:HGNC:5`의 첫 prefix는 DB 이름입니다. 잘못된 HGNC token, 역참조 누락·불일치·다중 참조를 audit에 남깁니다.
6. 양쪽 방향에서 각각 하나의 상대 ID만 가리켜야 합니다. 해당 HGNC를 참조하는 NCBI GeneID가 여러 개여도 채택하지 않습니다.

이 고유성 규칙은 프로젝트의 보수적 QC 정책이며, 공식 DB 전체가 일대일 관계임을 주장하지 않습니다. HGNC withdrawn export의 `WITHDRAWN_SYMBOL` 및 설명을 포함한 merger header도 명시적으로 처리합니다. 예상 필수 컬럼 누락, 불완전한 필수 row, 중복 human GeneID는 실패로 처리합니다.

## 산출물

| 파일 | 내용 |
|---|---|
| `gene_id_xref.csv` | 검증된 `hgnc_id`, `ncbi_gene_id`, `status=validated_reciprocal`, 양쪽 symbol, source, source_version |
| `gene_mapping_audit.csv` | 거부된 HGNC complete record 및 withdrawn record, source row, 사유, 원본 record, 원본 replacement 힌트 |
| `gene_mapping_coverage.csv` | v1 gene node별 및 HPO 원본 NCBI ID별 mapped/unresolved 상태와 입력 행 수 |
| `gene_mapping_qc.json` | source 수, 채택·제외 수, audit 사유, coverage 분모·분자 및 coverage 입력 해시 |
| `source_manifest.json` | 검증된 입력 manifest의 원본 bytes 복사 |
| `artifact_hashes.json` | 위 다섯 산출물의 SHA-256 |

`source_version`은 입력 source manifest bytes의 SHA-256입니다. 알고리즘 정책 버전은 QC의 `policy_version=hgnc_ncbi_reciprocal_v1`에 별도로 기록합니다. QC pass는 채택 pair가 존재하고 ID가 양방향으로 고유하다는 의미이며, 100% coverage나 graph 학습 적합성을 뜻하지 않습니다.

Coverage의 `HPO_genes_to_phenotype`는 **Gene ID identity만** 검사합니다. HPO ID의 active/obsolete 상태, `HP:0000118` descendant 여부, negative annotation, disease provenance, graph node universe 적합성 및 leakage는 후속 단계입니다. 이 결과의 331,363행을 positive Gene–HPO edge 수로 해석하면 안 됩니다. Gene–HPO의 기본 training context 제외 정책은 유지합니다.

## 재실행

저장소 root에서 같은 snapshot과 입력을 사용해 **새 출력 경로**로 실행합니다.

```bash
.venv/bin/python scripts/build_gene_mapping.py \
  --snapshot-dir data/raw/gene_registry/2026-10-16 \
  --output-dir data/processed/gene_mapping/2026-10-16_reciprocal_v1 \
  --gene-nodes data/processed/graph/gene_nodes.csv \
  --gene-annotations data/raw/hpo/genes_to_phenotype.txt
```

이미 생성된 결과를 검증용으로 다시 만들 때는 예를 들어 `--output-dir /tmp/raredx-gene-mapping-recheck`처럼 존재하지 않는 별도 경로를 사용합니다. 입력 파일 네 개의 SHA-256이 manifest와 하나라도 다르면 결과를 쓰기 전에 실패합니다. 같은 bytes와 설정은 같은 출력 bytes를 생성합니다. 최신 파일을 다시 다운로드해 기존 snapshot 위에 덮어쓰지 않습니다.

## 이번 snapshot 결과

| 항목 | 결과 |
|---|---:|
| HGNC complete records (Approved) | 45,277 |
| 현재 human NCBI records | 193,953 |
| human history records | 166,633 |
| 양방향 검증된 쌍 | 44,471 |
| HGNC complete에서 제외된 records | 806 |
| 별도 withdrawn audit records | 5,291 |
| v1 gene nodes의 mapped / 전체 | 4,518 / 4,552 (99.2531%) |
| HPO 원본 unique GeneID의 mapped / 전체 | 5,266 / 5,273 |
| HPO 원본 행의 ID mapped / 전체 | 331,363 / 331,738 (99.8870%) |

기존 v1 node의 미해결 34개는 withdrawn HGNC 32개, NCBI ID 없는 HGNC 1개(`HGNC:54925`), 기존 noncanonical locus 1개(`ORPHANET_LOCUS:23671`)입니다. v1의 기존 resolved/rankable 값을 이 결과로 변경하지 않았습니다.

HPO 원본에서 미해결 7개 ID / 375행은 retired ID 2개 / 102행과 승인 HGNC forward mapping 없는 ID 5개 / 273행입니다. 세부 목록과 source 근거는 coverage 및 snapshot에 보존했습니다.

후속 결정: 이 예외들의 대체 ID를 검토할 경우에도 별도의 검토·승인된 mapping policy와 provenance가 필요합니다. 이번 strict snapshot에 예외를 자동 삽입하지 않습니다.

## 2026-10-16 snapshot 검증 기록

- 실제 저장소 Python 3.12 전체 tests: **177 passed**, 기존 외부 Starlette/httpx deprecation warning 1개. 신규 registry tests 23개.
- 새 Python 파일 Ruff E4/E7/E9/F/I, format 및 AST 검사 통과. 독립 코드 리뷰에서 확인한 참조 공백·manifest byte hash·빈 매핑 QC 문제는 회귀 테스트 후 수정했습니다.
- 실제 저장소의 설치된 코드로 새 경로에 재생성한 **6개 파일의 bytes/hash가 모두 일치**했습니다.
- 작업 전 보존 대상으로 고정한 원본·graph·split·manifest·handoff **54개 파일의 SHA-256이 작업 후에도 모두 동일**합니다.
- 상세 검증 기록: `data/processed/gene_mapping/2026-10-16_verification.json`. 모델 학습·성능 평가는 수행하지 않았습니다.

## 2026-10-18 GitHub 반영 검증

- 최신 `main` (`5b8fc729a664e250026d92392f2372e144af1058`)에 매핑 관련 네 파일만 추가한 격리 checkout에서 Python 3.12 전체 tests **74 passed** (매핑 tests 23개 포함), 기존 Starlette/httpx deprecation warning 1개.
- 새 Python 세 파일의 Ruff E4/E7/E9/F/I 및 format 검사 통과. 독립 코드 리뷰에서 반영을 막는 문제 없음.
- 로컬 v1 보호 대상 54개 파일, 기존 매핑 산출물 6개 파일 및 매핑 코드 3개 파일이 이전 검증 기록의 SHA-256과 일치함을 다시 확인했습니다.
- 이번 PR 코드로 실제 로컬 snapshot을 새 경로에 재생성했으며 산출물 **6개 파일의 bytes가 모두 기존 결과와 일치**했습니다.
- 기존 원본 snapshot은 로컬 입력이며 GitHub에 포함하지 않습니다. 위 2026-10-16의 177개 tests 결과는 v1 그래프 개발 브랜치에서 수집한 별도 기록으로, 이번 최신 main의 테스트 수와 구분합니다.
