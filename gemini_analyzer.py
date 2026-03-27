"""
gemini_analyzer.py
영상 클립을 Gemini Flash에 전송해서 행동을 분석하는 모듈
Gemini Native Video API 사용 → 음성 + 영상 동시 분석

연결 방식: Vertex AI (GCP 프로젝트)
필요 환경변수:
    GOOGLE_CLOUD_PROJECT  : GCP 프로젝트 ID
    GOOGLE_CLOUD_LOCATION : 리전 (기본값 us-central1)
"""
import os
import json
from dataclasses import dataclass
from typing import Optional
from google import genai
from google.genai import types


@dataclass
class AnalysisResult:
    action: str                        # eating | drinking | urinating | lying | playing | scratching | barking | pacing | idle
    confidence: float                  # 0.0 ~ 1.0
    detail: str                        # 구체적인 행동 설명
    is_issue: bool                     # issue 여부
    issue_type: Optional[str]          # sudden_move | abnormal_health | anxiety | null
    alert_message: Optional[str]       # 사용자에게 보낼 메시지
    posture: str                       # sitting | standing | lying | unknown
    emotion: str                       # relaxed | anxious | excited | unknown
    input_tokens: int = 0              # 사용된 입력 토큰 수 (동영상 + 프롬프트)
    output_tokens: int = 0             # 사용된 출력 토큰 수


# 현재 상태 컨텍스트를 프롬프트에 포함시키기 위한 상태 설명 (lying, normal)
STATE_CONTEXT = {
    "lying": "강아지가 현재 누워있는 상태입니다. 이상 징후(경련, 낑낑거림)에 집중해주세요.",
    "normal": "일반적인 모니터링 상태입니다. 강아지의 현재 행동을 파악해주세요."
}


def analyze_clip(video_path: str,
                 current_state: str = "normal",
                 is_sudden_motion: bool = False,
                 time_context: str = "") -> AnalysisResult:
    """
    영상 클립을 Gemini Flash로 분석

    Args:
        video_path: 분석할 영상 경로
        current_state: 현재 상태 컨텍스트 ("lying" or "normal")
        is_sudden_motion: 모션 감지기에서 갑작스러운 움직임 감지됐는지 여부
        time_context: 이전 영상들로부터 누적된 행동 맥락 (예: "몇 청크 단위째 연속 누워있음")
    """
    project  = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    # Vertex AI 모드로 클라이언트 초기화
    # gcloud auth application-default login 또는 서비스 계정 키로 인증
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )

    # 1. 파일 용량이 너무 작은(1000 Bytes 미만) 빈 파일일 경우 스킵 (moviepy 렌더링 오류 방어)
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        print(f"    [경고] {video_path} 영상이 1KB 미만으로 비정상적입니다. 파이프라인 보호를 위해 이번 분석은 건너뜁니다.")
        return AnalysisResult(
            action="idle", confidence=0.0, detail="영상 렌더링 오류 (분석 스킵)", is_issue=False, 
            issue_type=None, alert_message=None, posture="unknown", emotion="unknown"
        )

    with open(video_path, "rb") as f:
        video_data = f.read()

    state_context = STATE_CONTEXT.get(current_state, STATE_CONTEXT["normal"])
    sudden_hint = (
        "※ 모션 감지기에서 갑작스러운 움직임이 감지되었습니다. 경련 여부를 확인해주세요."
        if is_sudden_motion else ""
    )

    context_hint = f"※ 누적 상황 기록: {time_context}" if time_context else ""

######### prompt #########
# context_hint 에는 lying 상태가 얼마나 지속되었는지에 대한 정보만 있음. prompt에서 언급한 누적 상황이 주어짐에 관한 부분 애매
# 배뇨 (소변 / 대변) 도 따로 action 에 저장 / DB 에 count 되게끔 해야함
# 이상행동들이 지속되는 것을 알려면 따로 서버에 저장하거나 기록해두어야 하지 않나 ? (특정 부위 긁는 행동 등)
    prompt = f"""
{state_context}
{sudden_hint}
{context_hint}

강아지의 행동을 면밀히 분석하고 가장 두드러진 행동을 선택해주세요. 이전 상황(누적 상황)이 주어진다면 이를 참고하여 불안 징후를 판별하세요.
특히 다음 사항들을 주의 깊게 구별하세요:
1. **식사(eating) vs 음수(drinking) 구분 주의**: 고개 숙인 행동만으로 섣불리 eating으로 단정하지 마세요. 영상 내 오디오에서 핥는 소리(촵촵)가 들리거나 투명한 액체(물)를 마신다면 반드시 'drinking'으로 분류하세요. 반면 씹는 소리가 나거나 고형물(사료/간식)을 먹는다면 'eating'입니다.
2. **건강 이상 행동 (abnormal_health)**: 과도한 긁기, 절뚝거림, 구역질 행동 패턴 등. 이상이 감지되면 detail에 관찰된 행동을 구체적으로 서술해줘.
3. **배뇨 행동 (urinating)**: 강아지가 소변/대변을 보는 행동 (다뇨 확인용)
4. **불안 징후 (anxiety)**: 문 앞 배회 지속, 빙빙 돌기, 파괴 행동 등. 이상이 감지되면 detail에 관찰된 행동을 구체적으로 서술해줘.

위 기준을 바탕으로 아래 JSON 형식으로만 답해줘. JSON 외 다른 텍스트는 절대 출력하지 마.

{{
    "action": "eating | drinking | urinating | playing | scratching | barking | walking | pacing | idle 중 하나",
    "posture": "sitting | standing | lying | unknown 중 하나",
    "emotion": "relaxed | anxious | excited | unknown 중 하나",
    "confidence": 0.0에서 1.0 사이 숫자,
    "detail": "행동 설명 1~2문장. 이상 징후가 있으면 관찰된 행동을 구체적으로 포함할 것.",
    "is_issue": true 또는 false,
    "issue_type": "abnormal_health | anxiety | sudden_move | null 중 하나",
    "alert_message": "사용자에게 보낼 자연스러운 한국어 알림 메시지 (정상 행동이면 null)"
}}

is_issue 판단 기준:
- 과도한 긁기/절뚝거림/구역질 등: true, issue_type = "abnormal_health"
- 배회/반복 행동/초조함: true, issue_type = "anxiety"
- 경련/갑작스러운 움직임: true, issue_type = "sudden_move"
- 그 외: false, issue_type = null
"""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=video_data, mime_type="video/mp4"), # 분석할 비디오 클립
                types.Part.from_text(text=prompt) # 프롬프트
            ]
        )
    ]

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",   # Vertex AI는 버전 suffix 필요
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1, # temp 0.1 ? 0 ?
                max_output_tokens=500,
            )
        )
    except Exception as e:
        # 2. 구글 서버 터짐, 타임아웃, 잘못된 응답 등 API 400/500 에러 전체 방어
        print(f"    [API 오류] Gemini 통신 실패: {str(e)}")
        return AnalysisResult(
            action="idle", confidence=0.0, detail=f"API 통신 오류로 분석 스킵", is_issue=False, 
            issue_type=None, alert_message=None, posture="unknown", emotion="unknown"
        )
    # response 객체의 원본 데이터 확인 (디버깅용)
    print("=== [RAW RESPONSE] ===")
    print(response)
    
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    # 파싱된 파이썬 딕셔너리 형태 확인
    print("=== [PARSED DATA] ===")
    print(data)
    
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

    return AnalysisResult(
        action=data.get("action", "idle"),
        confidence=float(data.get("confidence", 0.0)),
        detail=data.get("detail", ""),
        is_issue=bool(data.get("is_issue", False)),
        issue_type=data.get("issue_type"),
        alert_message=data.get("alert_message"),
        posture=data.get("posture", "unknown"),
        emotion=data.get("emotion", "unknown"),
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )
