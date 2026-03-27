"""
pipeline.py
메인 파이프라인

클립 파일들을 ./clips/ 폴더에 넣거나
CLIP_FILES 리스트에 직접 경로를 지정.
"""

import os
import glob
from datetime import datetime
from motion_detector import analyze_motion, analyze_motion_chunk
from gemini_analyzer import analyze_clip
from routine_tracker import RoutineTracker
from dotenv import load_dotenv
load_dotenv()   # .env 파일 읽기
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────
CLIP_FOLDER = "./clips"          # 클립 파일들이 있는 폴더
DB_PATH     = "dog_monitor.db"   # 현재 sqlite 로 구현. 이후 mongoDB로 보내짐
ALERT_LOG   = "alerts.log"       # 알림 텍스트 파일 저장 경로

# 수동으로 클립 순서를 지정하고 싶을 때 (비워두면 CLIP_FOLDER에서 자동 탐색)
CLIP_FILES = [
    # "./clips/lying.mp4",
    # "./clips/eating.mp4",
    # "./clips/scratching_ear.mp4",
    # "./clips/lookingaround.mp4",
]

# Lying 상태에서 Gemini 재분석 주기 (10초 단위 청크 기준) 10초단위 청크 ? 
LYING_RECHECK_INTERVAL = 3      # lying 상태에서 3개의 10초 청크(약 30초)마다 1번 Gemini 분석

# 로그 삭제 (터미널)
# cat /dev/null > alerts.log
# rm dog_monitor.db

# ──────────────────────────────────────────────────────────
# 알림 출력
# ──────────────────────────────────────────────────────────
def send_alert(message: str, level: str = "INFO"):
    """
    알림을 콘솔과 텍스트 파일 둘 다에 출력

    level: INFO | WARN | ISSUE
    """
    icons  = {"INFO": "ℹ️ ", "WARN": "⚠️ ", "ISSUE": "🚨"}
    icon   = icons.get(level, "")
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line   = "=" * 55
    body   = f"{icon}[{level}] {message}"

    # ── 콘솔 출력 ─────────────────────────────────────────
    print(f"\n{line}")
    print(f"  {body}")
    print(f"  {ts}")
    print(f"{line}\n")

    # ── 파일 저장 ─────────────────────────────────────────
    with open(ALERT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] [{level}] {message}\n")


# ──────────────────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print(f"🐶 강아지 모니터링 파이프라인 시작 (GCS 스트림 분석)")
    print(f"{'='*55}")

    tracker = RoutineTracker(db_path=DB_PATH)
    
    try:
        from moviepy import VideoFileClip
    except ImportError:
        print("\n[오류] moviepy 라이브러리가 설치되어 있지 않습니다.")
        print("연속적인 영상에서 10초 잘라내어 분석하기 위해 필요합니다.")
        return

    try:
        from google.cloud import storage
    except ImportError:
        print("\n[오류] google-cloud-storage 패키지가 설치되지 않았습니다.")
        print("터미널에서 다음 명령어로 설치해주세요: pip install google-cloud-storage")
        return

    # --- GCS에서 파일 로드 ---
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    GCS_BLOB_NAME = "merged_feed/temp_continuous_feed.mp4"
    temp_feed_path = "temp_continuous_feed.mp4"

    if not GCS_BUCKET_NAME:
        print("[오류] .env 파일 상에 GCS_BUCKET_NAME 변수가 설정되어 있지 않습니다.")
        print("예: GCS_BUCKET_NAME=my-dog-monitor-bucket")
        return

    print(f"→ GCS 버킷({GCS_BUCKET_NAME})에서 병합된 영상({GCS_BLOB_NAME})을 로드합니다...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_BLOB_NAME)

    if not blob.exists():
        print(f"[오류] GCS에 {GCS_BLOB_NAME} 파일이 존재하지 않습니다. 먼저 merge_and_upload.py 를 실행하세요!")
        return

    # 로컬에 동일 파일이 있으면 재사용하거나 덮어쓰기 (단순함을 위해 매번 덮어씀)
    print("→ 로컬로 영상을 다운로드 중입니다... (잠시 대기)")
    blob.download_to_filename(temp_feed_path)
    print(f"✅ GCS 영상 다운로드 완료 (경로: {temp_feed_path})")

    merged_clip = VideoFileClip(temp_feed_path)
    total_duration = merged_clip.duration

    dog_state = "normal"
    lying_chunk_counter = 0

    current_time = 0.0
    chunk_duration = 10.0

    while current_time < total_duration:
        print(f"\n{'─'*55}")
        end_time_clamped = min(current_time + chunk_duration, total_duration)
        print(f"[처리 구간] {current_time:.1f}초 ~ {end_time_clamped:.1f}초")

        # ── STEP 1: 모션 감지 ──────────────────────────────────
        print("  → 모션 분석 중...")
        motion = analyze_motion_chunk(
            video_path=temp_feed_path,
            start_sec=current_time,
            duration_sec=chunk_duration
        )

        should_analyze = False
        skip_reason = ""
        analysis_start_sec = current_time

        # ── STEP 2: 상태 분기 ──────────────────────────────────
        if motion.has_motion: # 모션있으면 분석
            should_analyze = True 
            analysis_start_sec = max(0.0, motion.first_motion_time - 5.0) # 움직임 시작 5초 전부터 자르기
            if dog_state == "lying":
                print(f"  → lying 상태에서 모션 감지({motion.first_motion_time:.1f}초 시점) → Gemini 분석 실행")
            else:
                print(f"  → 일반 상태에서 모션 감지({motion.first_motion_time:.1f}초 시점) → 추출 실행")
        else: # 모션없으면
            if dog_state == "lying": # 누워있는 상태는 30초마다 한번씩 상태확인 - 왜 30초?
                lying_chunk_counter += 1
                if lying_chunk_counter >= LYING_RECHECK_INTERVAL:
                    should_analyze = True
                    lying_chunk_counter = 0
                    analysis_start_sec = current_time
                    print(f"  → lying 중 주기적 재확인 ({LYING_RECHECK_INTERVAL}단위 청크마다)")
                else:
                    skip_reason = f"lying 상태 유지 중 ({lying_chunk_counter}/{LYING_RECHECK_INTERVAL})"
            else: # 누워있지도 않고 움직임도 없으면
                skip_reason = "모션 없음 → 분석 스킵"

        if not should_analyze:
            print(f"  → [{skip_reason}] Gemini 호출 없음")
            
            # MM:SS 시간 포맷 변환
            mm = int(current_time) // 60
            ss = int(current_time) % 60
            time_formatted = f"{mm:02d}:{ss:02d}"

            tracker.log_event(
                action="idle" if dog_state != "lying" else "lying", # 움직임 없어서 분석뛰어넘는경우 = 'idle'
                detail=skip_reason,
                confidence=1.0,
                clip_name=f"stream_{current_time:.1f}s",
                video_time_formatted=time_formatted
            )
            current_time += chunk_duration
            continue

        # 앞 분기에서 분석하기로 결정된 부분만
        # ── STEP 3: 10초 분량 부분 추출 및 Gemini 분석 ────────────────
        extraction_end = min(analysis_start_sec + 10.0, total_duration)
        print(f"  → {analysis_start_sec:.1f}~{extraction_end:.1f}초 구간 영상 추출 (Gemini용)...")
        
        temp_10s_path = "temp_10s_gemini.mp4"
        subclip = merged_clip.subclipped(analysis_start_sec, extraction_end) # moviepy 라이브러리
        subclip.write_videofile(temp_10s_path, fps=merged_clip.fps, codec="libx264", audio_codec="aac", logger=None) # 임시파일로 렌더링

        print("  → Gemini Flash 분석 중... (영상+음성)")
        time_context = f"{lying_chunk_counter}단위째 연속 누워있음" if dog_state == "lying" else ""
        
        result = analyze_clip( # gemini_analyzer.py 분석 결과
            video_path=temp_10s_path,
            current_state=dog_state,
            is_sudden_motion=motion.is_sudden,
            time_context=time_context
        )
        
        cost = (result.input_tokens / 1_000_000 * 0.10) + (result.output_tokens / 1_000_000 * 0.40)
        print(f"  → 결과: [{result.action}] 비고: {result.posture}/{result.emotion} | 신뢰도={result.confidence:.0%} | {result.detail}")
        print(f"  → 비용: 입력 {result.input_tokens} | 출력 {result.output_tokens} (예상 ${cost:.6f})")

        # ── STEP 4: SQLite 저장 ────────────────────────────────
        alert_was_sent = result.alert_message is not None or result.is_issue
        tier2 = result.issue_type if result.issue_type in ["abnormal_health", "anxiety", "sudden_move"] else None
        
        # MM:SS 시간 포맷 변환
        mm = int(analysis_start_sec) // 60
        ss = int(analysis_start_sec) % 60
        time_formatted = f"{mm:02d}:{ss:02d}"

        # gemini JSON output -> SQLite 저장
        tracker.log_event(
            action=result.action,
            detail=result.detail,
            confidence=result.confidence, # 필요한지 모르겠음
            clip_name=f"stream_{analysis_start_sec:.1f}s",
            video_time_formatted=time_formatted,
            is_issue=result.is_issue,
            issue_type=result.issue_type,
            alert_sent=alert_was_sent,
            posture=result.posture,
            emotion=result.emotion,
            tier2_issue=tier2
        )

        # ── STEP 5: 알림 전송 ─────────────────────────────────
        if result.is_issue:
            send_alert(result.alert_message or "이상 행동이 감지됐어요!", level="ISSUE")
        elif result.alert_message:
            send_alert(result.alert_message, level="INFO")

        # ── STEP 6: 상태 전이 ─────────────────────────────────
        if result.action == "lying" and dog_state != "lying": # 누워있지 않다가 누운 경우
            dog_state = "lying"
            lying_chunk_counter = 0
            tracker.set_state("lying_since", datetime.now().isoformat())
            send_alert("강아지가 누웠어요 🐾 잠시 쉬는 것 같아요.", level="INFO")

        elif result.action != "lying" and dog_state == "lying": # 누워있다가 누워있지 않은 경우
            # 누워있던 총 시간 계산해서 DB(daily_summary)에 누적
            lying_since_str = tracker.get_state("lying_since")
            lying_minutes = 0
            if lying_since_str:
                lying_since_date = datetime.fromisoformat(lying_since_str)
                lying_minutes = int((datetime.now() - lying_since_date).total_seconds() // 60)
                if lying_minutes > 0:
                    tracker.add_lying_minutes(lying_minutes)

            dog_state = "normal"
            tracker.set_state("lying_since", "")
            print(f"  → lying 상태 종료 → [{result.action}] 으로 전환 (총 {lying_minutes}분 휴식 누적)")

        # ── STEP 7: 루틴 체크 ─────────────────────────────────
        routine_alerts = tracker.check_routine_alerts()
        for alert in routine_alerts:
            send_alert(alert, level="WARN")

        # Gemini 분석 후, 추출된 구간 다음부터 이어서 확인하기 위해 현 시각 갱신
        if extraction_end > current_time:
            current_time = extraction_end
        else:
            current_time += chunk_duration

    # 최종 하루 요약 출력
    summary = tracker.get_today_summary()
    print(f"\n{'='*55}")
    print(f"📋 오늘의 요약")
    print(f"   🍚 식사: {summary['eating']}번  "
          f"💧 음수: {summary['drinking']}번  "
          f"🎾 놀이: {summary['playing']}번")
    print(f"   🚨 이슈: {summary['issue_count']}건")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
