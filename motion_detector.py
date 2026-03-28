"""
motion_detector.py
각 클립에서 모션 여부 + 강도를 판단하는 모듈
해상도: 320x240 리사이즈 후 분석
FPS:   target_fps(기본 1fps)로 샘플링 후 분석
"""
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class MotionResult:
    has_motion: bool
    motion_ratio: float      # 모션 샘플 프레임 비율 (0.0 ~ 1.0)
    avg_motion_area: float   # 평균 움직임 면적
    max_motion_area: float   # 최대 움직임 면적
    is_sudden: bool          # 갑작스러운 움직임 여부 (경련 감지용)
    sampled_frames: int      # 실제 분석한 프레임 수
    first_motion_time: float # 처음 모션이 감지된 시점 (초)
    annotated_frame_base64: str = "" # 시각화용 이미지

# 영상의 start_sec부터 duration_sec(예: 10초) 동안만 잘라서 모션을 확인하는 함수
def analyze_motion_chunk(video_path: str, start_sec: float, duration_sec: float,
                         machine_code: str = "",
                         motion_threshold: int = 3000,
                         sudden_threshold: int = 15000,
                         target_fps: int = 1) -> MotionResult:
    """
    영상 클립의 특정 구간(start_sec ~ start_sec + duration_sec) 모션을 분석
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    # 지정된 시작 시간으로 이동
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    hop = max(1, int(original_fps / target_fps)) # 1초에 1장씩만 추출하도록 (1FPS)

    motion_areas = []
    motion_frame_count = 0
    raw_frame_idx = 0
    sampled_frames = 0
    prev_area = 0
    sudden_detected = False
    first_motion_time = -1.0
    
    prev_gray = None
    max_seen_area = 0
    best_frame_small = None

    # duration_sec 초 분량의 프레임 수
    max_frames = int(duration_sec * original_fps)
    frames_read = 0

    while cap.isOpened() and frames_read < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_read += 1

        if raw_frame_idx % hop != 0:
            raw_frame_idx += 1
            continue

        small = cv2.resize(frame, (320, 240)) # 320x240로 리사이즈
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) # 흑백으로 변환 - 모션감지에 컬러 굳이 X
        
        if prev_gray is None:
            prev_gray = gray
            raw_frame_idx += 1
            continue

        frame_diff = cv2.absdiff(prev_gray, gray) # 직전 프레임 - 현재 프레임 
        _, fg_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY) # 25 이상 차이나면 변화로 판단
        motion_area = cv2.countNonZero(fg_mask)
        motion_areas.append(motion_area) # 하얀 픽셀 수 = 움직임의 양

        if motion_area > motion_threshold: # 변한 픽셀 면적이 3000 이상이면 모션이 있다고 판단
            motion_frame_count += 1
            if first_motion_time < 0: # 처음 모션이 감지된 시간 (앞뒤 5초 자르기 기준값)
                first_motion_time = start_sec + (frames_read / original_fps)
            
        if motion_area > max_seen_area:
            max_seen_area = motion_area
            best_frame_small = small.copy()
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(best_frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if prev_area < motion_threshold and motion_area > sudden_threshold: # 경련 감지 -> 고민 필요
            sudden_detected = True

        prev_area = motion_area
        raw_frame_idx += 1
        sampled_frames += 1

    cap.release()
    
    import base64
    annotated_frame_base64 = ""
    if best_frame_small is not None:
        _, buffer = cv2.imencode('.jpg', best_frame_small)
        annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    if sampled_frames == 0 or len(motion_areas) == 0:
        return MotionResult(False, 0.0, 0.0, 0.0, False, 0, -1.0, "")

    import numpy as np
    motion_ratio = motion_frame_count / sampled_frames
    avg_area = float(np.mean(motion_areas))
    max_area = float(np.max(motion_areas))

    has_motion = motion_ratio >= 0.10 # 분석한 프레임 중 모션이 감지된 프레임의 비율이 10% 이상일때 모션이 있음으로 감지됨

    return MotionResult(
        has_motion=has_motion,
        motion_ratio=motion_ratio,
        avg_motion_area=avg_area,
        max_motion_area=max_area,
        is_sudden=sudden_detected, # 경련이나 매우 급격한 움직임 여부
        sampled_frames=sampled_frames,
        first_motion_time=first_motion_time, # 움직임이 처음시작된 시간(초) - gemini 에 보낼 기준점
        annotated_frame_base64=annotated_frame_base64
    )

def analyze_motion(video_path: str,
                   machine_code: str = "",
                   motion_threshold: int = 3000,
                   sudden_threshold: int = 15000,
                   target_fps: int = 1) -> MotionResult:
    # 하위 호환성을 위해 분석 시간을 매우 길게 지정해서 전체 클립 분석
    return analyze_motion_chunk(video_path, 0.0, 99999.0, machine_code, motion_threshold, sudden_threshold, target_fps)

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("사용법: python motion_detector.py <기계코드> [비디오경로]")
        sys.exit(1)
        
    machine_code = sys.argv[1]
    test_video = sys.argv[2] if len(sys.argv) > 2 else "test.mp4"
    
    print(f"[{machine_code}] 기계 코드 적용 - 모션 분석 시작: {test_video}")
    result = analyze_motion(test_video, machine_code=machine_code)
    
    print(f"모션 감지 여부: {result.has_motion}")
    
    if result.has_motion:
        try:
            from gemini_analyzer import analyze_clip
            print(f"[{machine_code}] Gemini 분석으로 넘깁니다...")
            # gemini_analyzer 쪽으로 기계 코드(인자) 함께 전달
            gemini_result = analyze_clip(test_video, machine_code=machine_code)
            print(f"Gemini 분석 결과: {gemini_result.action} / {gemini_result.detail}")
        except ImportError:
            print("gemini_analyzer.py 를 찾을 수 없거나 임포트 에러가 발생했습니다.")


