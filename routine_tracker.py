"""
routine_tracker.py
행동 로그를 SQLite에 저장하고, 하루 루틴 이상을 감지하는 모듈

SQLite 역할 요약:
TABLE
  events         → Gemini 분석 결과를 클립 단위로 저장
  daily_summary  → 오늘 하루 행동 횟수 집계
  state          → 현재 강아지 상태 (lying 진입 시간 등)
"""
import sqlite3
from datetime import datetime, date
from typing import Optional
from dataclasses import dataclass


@dataclass
class DailyGoals: # 임의 설정
    eating: int = 3
    drinking: int = 5


class RoutineTracker:
    def __init__(self, db_path: str = "dog_monitor.db"):
        self.db_path = db_path
        self.goals = DailyGoals()
        self._init_db()

    # ──────────────────────────────────────────
    # DB 초기화
    # ──────────────────────────────────────────
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    date         TEXT NOT NULL,
                    timestamp    TEXT NOT NULL,
                    clip_name    TEXT,
                    video_time_formatted TEXT,
                    action       TEXT NOT NULL,
                    confidence   REAL,
                    detail       TEXT,
                    is_issue     INTEGER DEFAULT 0,
                    issue_type   TEXT,
                    alert_sent   INTEGER DEFAULT 0,
                    posture      TEXT,
                    emotion      TEXT,
                    barking_reason TEXT,
                    tier2_issue  TEXT
                );

                CREATE TABLE IF NOT EXISTS daily_summary (
                    date             TEXT PRIMARY KEY,
                    eating_count     INTEGER DEFAULT 0,
                    drinking_count   INTEGER DEFAULT 0,
                    urinating_count  INTEGER DEFAULT 0,
                    playing_count    INTEGER DEFAULT 0,
                    scratching_count INTEGER DEFAULT 0,
                    lying_minutes    INTEGER DEFAULT 0,
                    issue_count      INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS state (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

            # 기존 만들어진 DB에 새 컬럼을 자동으로 추가하기 위한 안전장치
            try:
                conn.execute("ALTER TABLE events ADD COLUMN video_time_formatted TEXT;")
            except sqlite3.OperationalError:
                pass  # 이미 컬럼이 존재하면 무시됨

    # ──────────────────────────────────────────
    # 이벤트 저장
    # ──────────────────────────────────────────
    def log_event(self, action: str, detail: str, confidence: float,
                  clip_name: str = "", video_time_formatted: str = "", is_issue: bool = False,
                  issue_type: Optional[str] = None, alert_sent: bool = False,
                  posture: Optional[str] = None, emotion: Optional[str] = None,
                  tier2_issue: Optional[str] = None):
        today = date.today().isoformat()
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO events
                    (date, timestamp, clip_name, video_time_formatted, action, confidence,
                     detail, is_issue, issue_type, alert_sent,
                     posture, emotion, tier2_issue)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (today, now, clip_name, video_time_formatted, action, confidence,
                  detail, int(is_issue), issue_type, int(alert_sent),
                  posture, emotion, tier2_issue))

            # daily_summary 카운트 업데이트
            count_col = f"{action}_count"
            valid_actions = ["eating", "drinking", "urinating", "playing", "scratching"]
            # gemini returns action 중 daily_summary 횟수로 카운팅할 액션들만 

            if action in valid_actions:
                conn.execute(f"""
                    INSERT INTO daily_summary (date, {count_col})
                    VALUES (?, 1)
                    ON CONFLICT(date) DO UPDATE SET
                    {count_col} = {count_col} + 1
                """, (today,))
            else:
                conn.execute("""
                    INSERT OR IGNORE INTO daily_summary (date) VALUES (?)
                """, (today,))

            if is_issue:
                conn.execute("""
                    UPDATE daily_summary SET issue_count = issue_count + 1
                    WHERE date = ?
                """, (today,))

    # ──────────────────────────────────────────
    # 상태 관리 (lying 진입/탈출 시간 추적) -> 지금 필요 ? (dashboard 현재는 안쓸예정)
    # ──────────────────────────────────────────
    def set_state(self, key: str, value: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?
            """, (key, value, value))

    def get_state(self, key: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM state WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def add_lying_minutes(self, minutes: int):
        today = date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO daily_summary (date, lying_minutes)
                VALUES (?, ?)
                ON CONFLICT(date) DO UPDATE SET
                lying_minutes = lying_minutes + ?
            """, (today, minutes, minutes))

    # ──────────────────────────────────────────
    # 오늘 요약 조회
    # ──────────────────────────────────────────
    def get_today_summary(self) -> dict:
        today = date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM daily_summary WHERE date = ?", (today,)
            ).fetchone()

        if not row:
            return {"date": today, "eating": 0, "drinking": 0,
                    "playing": 0, "scratching": 0, "issue_count": 0}

        return { # 이것도 잘 맵핑되게 하기
            "date":      row[0],
            "eating":    row[1],
            "drinking":  row[2],
            "urinating": row[3],
            "playing":   row[4],
            "scratching": row[5],
            "lying_minutes": row[6],
            "issue_count": row[7],
        }
        
    # ──────────────────────────────────────────
    # 루틴 이상 감지
    # ──────────────────────────────────────────
    def check_routine_alerts(self) -> list[str]:
        now = datetime.now()
        summary = self.get_today_summary()
        alerts = []

        # 저녁 18시 이후 → 식사 횟수 부족 체크
        if now.hour >= 18:
            eaten = summary.get("eating", 0)
            if eaten < self.goals.eating:
                alerts.append(
                    f"오늘 강아지가 밥을 {eaten}번 먹었어요 "
                    f"(목표 {self.goals.eating}번). 저녁 식사를 챙겨줄까요? 🍚"
                )

        # ── 음수 횟수 체크 (시간대별 누적 기준) ──────────────
        # 오전 10시 이후 → 아직 0번이면 알림
        # 오후 15시 이후 → 2번 미만이면 알림
        # 저녁 20시 이후 → 목표(5번) 미만이면 알림
        drunk = summary.get("drinking", 0)

        if now.hour >= 20 and drunk < self.goals.drinking:
            alerts.append(
                f"오늘 물을 {drunk}번 마셨어요 (목표 {self.goals.drinking}번). "
                f"자기 전에 물을 챙겨주세요! 💧"
            )
        elif now.hour >= 15 and drunk < 2:
            alerts.append(
                f"오후인데 아직 물을 {drunk}번밖에 마시지 않았어요. "
                f"탈수 증상이 없는지 확인해보세요 💧"
            )
        elif now.hour >= 10 and drunk == 0:
            alerts.append(
                "오전인데 아직 물을 한 모금도 마시지 않았어요. "
                "물그릇을 확인해주세요 🚰"
            )

        # ── lying 지속 시간 체크 ──────────────────────────────
        lying_since_str = self.get_state("lying_since")
        if lying_since_str:
            lying_since = datetime.fromisoformat(lying_since_str)
            lying_minutes = int((now - lying_since).total_seconds() // 60)
            if lying_minutes >= 240:  # 4시간 이상
                alerts.append(
                    f"강아지가 {lying_minutes // 60}시간째 누워있어요. "
                    f"괜찮은지 확인해보세요 🐾"
                )

        return alerts
