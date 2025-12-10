# Furiosa QA Automation Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/pytest-8.3-green.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Furiosa LLM OpenAI-Compatible API 테스트 자동화 프레임워크**
>
> FuriosaAI의 NPU 기반 LLM 서빙 솔루션을 위한 포괄적인 QA 자동화 프레임워크입니다.

## 📋 프로젝트 개요

이 프로젝트는 [Furiosa LLM OpenAI-Compatible Server](https://developer.furiosa.ai/latest/en/furiosa_llm/furiosa-llm-serve.html) API를 테스트하기 위한 자동화 프레임워크입니다.

### 주요 기능

- **API 테스트**: Chat Completions, Completions, Models, Version, Metrics API 테스트
- **SDK 시뮬레이션**: 디바이스 감지, SamplingParams 검증 테스트
- **에러 핸들링**: 잘못된 요청, 파라미터 검증 테스트
- **CI/CD 통합**: GitHub Actions 워크플로우 포함

## 🏗️ 프로젝트 구조

```
furiosa-qa-automation/
├── mock_server/                    # Furiosa API Mock 서버
│   └── main.py                     # FastAPI 서버 구현
├── tests/
│   ├── api/                        # API 엔드포인트 테스트
│   │   ├── test_chat_completion.py # /v1/chat/completions
│   │   ├── test_completions.py     # /v1/completions
│   │   ├── test_models_api.py      # /v1/models
│   │   ├── test_version_api.py     # /version
│   │   └── test_metrics.py         # /metrics
│   ├── sdk/                        # SDK 시뮬레이션 테스트
│   │   ├── test_device_detection.py
│   │   └── test_sampling_params.py
│   └── error/                      # 에러 핸들링 테스트
│       └── test_error_handling.py
├── .github/workflows/
│   └── test.yml                    # CI/CD 파이프라인
├── conftest.py                     # pytest fixtures
├── pytest.ini                      # pytest 설정
├── requirements.txt
└── README.md
```

## 🚀 시작하기

### 사전 요구사항

- Python 3.10+
- pip

### 설치

```bash
# 저장소 클론
git clone https://github.com/masha0465/furiosa-qa-automation.git
cd furiosa-qa-automation

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### Mock 서버 실행

```bash
python -m uvicorn mock_server.main:app --host 127.0.0.1 --port 8000
```

### 테스트 실행

```bash
# 전체 테스트 실행
pytest tests/ -v

# API 테스트만 실행
pytest tests/api/ -v

# SDK 테스트만 실행
pytest tests/sdk/ -v

# 특정 마커로 실행
pytest -m smoke  # smoke 테스트만
pytest -m api    # API 테스트만

# 커버리지 리포트
pytest tests/ --cov=mock_server --cov-report=html
```

## 📊 테스트 범위

### API 테스트

| 엔드포인트 | 테스트 항목 | 테스트 수 |
|-----------|------------|----------|
| `/v1/chat/completions` | 기본 요청, 스트리밍, 파라미터, 멀티턴 | 12 |
| `/v1/completions` | 기본 요청, 스트리밍, 파라미터 | 8 |
| `/v1/models` | 모델 목록, 개별 모델, Furiosa 확장 필드 | 8 |
| `/version` | 버전 정보, 포맷 검증 | 5 |
| `/metrics` | Prometheus 포맷, 메트릭 항목 | 12 |

### SDK 시뮬레이션 테스트

| 영역 | 테스트 항목 | 테스트 수 |
|-----|------------|----------|
| 디바이스 감지 | 단일/다중 디바이스, 가용성 | 12 |
| SamplingParams | 기본값, 유효성 검증, 파라미터 조합 | 19 |

### 에러 핸들링 테스트

| 영역 | 테스트 항목 | 테스트 수 |
|-----|------------|----------|
| 잘못된 요청 | 필수 필드 누락, 빈 값 | 5 |
| 잘못된 타입 | 파라미터 타입 오류 | 4 |
| HTTP 메서드 | 잘못된 메서드 | 2 |
| JSON 오류 | 잘못된 JSON | 2 |

## 🔧 기술 스택

- **테스트 프레임워크**: pytest 8.3
- **Mock 서버**: FastAPI + Uvicorn
- **HTTP 클라이언트**: requests
- **CI/CD**: GitHub Actions

## 📖 Furiosa API 참고 문서

- [Furiosa LLM 문서](https://developer.furiosa.ai/latest/en/furiosa_llm/intro.html)
- [OpenAI-Compatible Server](https://developer.furiosa.ai/latest/en/furiosa_llm/furiosa-llm-serve.html)
- [SamplingParams](https://developer.furiosa.ai/latest/en/furiosa_llm/reference/sampling_params.html)

## 🎯 QA Automation Engineer 포지션 관련

이 프로젝트는 다음 역량을 증명합니다:

- ✅ **Python 기반 테스트 자동화** (pytest)
- ✅ **API 테스트 자동화** (REST API)
- ✅ **테스트 케이스 설계 및 구현**
- ✅ **CI/CD 파이프라인 구축** (GitHub Actions)
- ✅ **Furiosa SDK/API 이해**

## 📝 라이선스

MIT License

## 👤 작성자

김선아 (Sunah Kim)
- QA Engineer with 9+ years of experience
- Specializing in Test Automation (Playwright, pytest)
