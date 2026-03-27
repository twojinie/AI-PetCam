# 🐶 강아지 행동 모니터링 파이프라인 (VLM Pipeline)

Gemini 2.0 Flash(Vertex AI)와 OpenCV 모션 감지를 이용해 강아지의 행동(먹기, 마시기, 배변, 수면 등)을 24시간 실시간으로 모니터링하고 기록하는 지능형 파이프라인입니다.

## 📁 컴포넌트 구조
```
dog_monitor/
├── merge_and_upload.py  # (옵션) 로컬 클립들을 하나의 영상으로 병합 후 GCS 업로드
├── pipeline.py          # 메인 파이프라인 (GCS 다운로드 → 모션 분석 → Gemini 호출)
├── motion_detector.py   # STEP 1: 모션 감지 (OpenCV 기반, 1FPS 샘플링)
├── gemini_analyzer.py   # STEP 2: Gemini 2.0 Flash 분석 (오디오+비디오 동시 분석)
├── routine_tracker.py   # STEP 3: SQLite 로그 저장 (루틴 통계, 누적 수면시간 추적)
└── clips/               # 분석 대상 원본 영상 (.mp4)
```

## 🛠 설치 및 사전 준비

1. **Python 라이브러리 설치**
   ```bash
   pip install google-genai opencv-python numpy moviepy google-cloud-storage python-dotenv
   ```
2. **GCP(Google Cloud Project) 인증**
   이 프로젝트는 **Vertex AI API**를 사용합니다.
   ```bash
   gcloud auth application-default login
   ```
3. **환경변수 설정 (`.env` 생성)**

## 🚀 실행 방법

### 1단계: (선택) 영상 병합 및 GCS 업로드
여러 개의 짧은 모바일 촬영 영상을 하나의 긴 스트림 영상으로 만들고 구글 클라우드에 올립니다.
```bash
python merge_and_upload.py
```
> 성공 시 풀 영상(`temp_continuous_feed.mp4`)이 `GCS_BUCKET_NAME` 버킷에 업로드됩니다.

### 2단계: 메인 파이프라인 실행
스트림 영상을 10초 단위로 자르면서, 모션이 감지된 구간만 발췌해 Gemini AI에게 행동을 분석시킵니다.
```bash
python pipeline.py
```

## 🧠 핵심 로직 흐름

1. **연속 스트림 분석 (10초 청크)**
   - 파이프라인은 전체 영상을 10초씩 순회합니다.
2. **비용 최적화 (모션 감지)**
   - OpenCV가 영상 내 픽셀 변화량(모션)을 감지합니다. 움직임이 전혀 없다면,Gemini를 호출하지 않고 `idle` 상태로 DB에 기록하며 다음 시간대로 점프합니다.
3. **스마트 추출 (클리핑)**
   - 모션이 감지되면, **움직임이 시작된 시점 5초 전부터 10초 길이**의 영상을 정밀하게 잘라냅니다. (`temp_10s_gemini.mp4`)
4. **VLM 상세 분석 (Gemini 2.0 Flash)**
   - 잘라낸 영상과 오디오를 Gemini에 전송합니다. 
5. **DB 통계 누적 (SQLite)**
   - 분석 결과는 `dog_monitor.db`에 저장됩니다. 
   - 강아지가 낮잠(lying)을 자고 일어나면 총 수면 시간(분 단위)을 계산하여 하루 통계에 자동으로 합산합니다.

## 🗄 SQLite DB 테이블 (`dog_monitor.db`)

| 테이블 명 | 역할 | 주요 컬럼 |
|--------|------|------|
| `events` | 개별 행동 이벤트 기록 | `timestamp`, `video_time_formatted`, `action`, `detail` |
| `daily_summary` | 하루 통계 (대시보드용) | `eating_count`, `drinking_count`, `lying_minutes` |
| `state` | 현재/임시 상태 저장 | `lying_since` (수면 시작 시간 추적용) |

## 💡 주요 Try-Except 처리
- **빈 영상 오류**: 영상 병합이나 자르기 중 0바이트 빈 파일이 생기면, 파이프라인이 멈추지 않도록 스킵 로직과 1KB 이하 필터링을 도입
