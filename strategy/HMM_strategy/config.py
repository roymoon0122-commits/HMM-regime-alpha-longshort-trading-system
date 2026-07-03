"""
HMM_strategy 패키지의 모든 튜닝 가능 변수 모음 (중앙 설정 파일).

────────────────────────────────────────────────────────────────────
사용 원칙 (반드시 지킬 것)
────────────────────────────────────────────────────────────────────
1. 이 파일은 "기본값 모음집" 역할만 한다.
   → 함수는 이 파일의 변수를 인자의 기본값으로만 사용한다.

2. 함수 내부에서 직접 config를 import 하지 말 것.
   → 단위 테스트 시 인자로 다른 값을 주입할 수 없게 됨.
   → 같은 함수를 여러 파라미터로 부를 수 없게 됨.

3. 호출하는 쪽 (verify_features.py, strategy.py 등)에서 config 값을
   인자로 명시적으로 전달한다.

────────────────────────────────────────────────────────────────────
좋은 예시 (Pattern B)
────────────────────────────────────────────────────────────────────
  # config.py
  ADX_PERIOD = 12

  # indicators.py
  def compute_adx(df, period=12):     # 인자로 받음
      ...

  # 호출하는 쪽
  from strategy.HMM_strategy import config
  adx = compute_adx(df, period=config.ADX_PERIOD)

────────────────────────────────────────────────────────────────────
나쁜 예시 (Pattern A — 절대 금지)
────────────────────────────────────────────────────────────────────
  # indicators.py
  from .config import ADX_PERIOD     # ← 함수 내부에서 직접 import
  def compute_adx(df):
      period = ADX_PERIOD            # ← 숨은 의존성
      ...
"""

# ─── 데이터 ───────────────────────────────────────────────────────
# 주식은 분봉을 미리 정규장 30분봉으로 리샘플한 parquet을 쓴다.
# (data/2_resample_bars.py 로 생성 → data/30min/ 폴더)
# 로드는 stock_loader.load_resampled_bars() 사용.
DATA_PATH = "data/30min/AAPL_20210101_20260523_30min.parquet"   # 기본 종목
TIMEFRAME = "30min"     # 정규장 30분봉 (하루 13봉: 09:30~15:30 ET).
                        # 리샘플은 미리 끝나 있으므로, 이 값은 워밍업·
                        # 연율화 계산의 라벨로만 쓰인다.
FORCE_RETRAIN = True    # True: joblib 캐시 무시하고 매번 새로 학습 후 저장
                        # False: 캐시 파일이 있으면 재사용


# ─── 윈도우 ──────────────────────────────────────────────────────
WINDOW_SIZE  = 130      # 윈도우 내 봉 수 (30분봉 130봉 = 10거래일 ≈ 2주)
PREDICT_SIZE = 60       # 예측 대상 윈도우 크기 (Phase 2~3에서 사용)
STEP_SIZE    = 1        # 롤링 스텝 (1 = 1봉씩 이동)


# ─── ADX ────────────────────────────────────────────────────────
ADX_PERIOD    = 26      # ADX 계산 기간 (30분봉 26봉 ≈ 2거래일)
ADX_THRESHOLD = 25      # 추세/횡보 경계값 (Phase 3 분류기에서 사용)


# ─── R² ─────────────────────────────────────────────────────────
R2_PERIOD     = 88      # 선형회귀 기간 (30분봉 88봉 ≈ 6.8거래일)
R2_THRESHOLD  = 0.55    # R² 임계값 (Phase 3 분류기에서 사용)


# ─── 분류기 시그모이드 기울기 (Phase 3에서 활성) ────────────────
# Soft probability 변환을 위한 sigmoid 곡선 기울기.
# 작을수록 부드럽게(완만), 클수록 임계값 근처에서 급격히 변함.
#
# ADX_CLF_STEEPNESS = 0.2 → ADX가 임계값 ±5 범위에서 0.27~0.73 분배
# R2_CLF_STEEPNESS  = 8.0 → R²가 임계값 ±0.1 범위에서 0.31~0.69 분배
# DIRECTION_STEEPNESS = 50 → cum_return이 ±0.03(±3%) 범위에서 0.18~0.82 분배
ADX_CLF_STEEPNESS    = 0.2     # ADX 추세 강도 시그모이드 기울기
R2_CLF_STEEPNESS     = 8.0     # R² 직선성 시그모이드 기울기 (R²는 0~1 범위)
DIRECTION_STEEPNESS  = 50.0    # cum_return 방향 시그모이드 (ADX 분류기용, Bull vs Bear)
R2_DIRECTION_STEEPNESS = 1.0   # 정규화된 slope 방향 시그모이드 (R² 분류기용)
                                # slope을 RollingStandardScaler로 z-score화한 값을 입력으로 사용.
                                # z-score 분포가 std≈1이므로 steepness=1로 sigmoid(±1.5)≈0.18~0.82 분배.
SLOPE_NORM_COL = 'slope_norm'  # 정규화된 slope 컬럼명 (호출자가 미리 계산해서 넣어둘 것)


# ─── HMM (Phase 2에서 활성) ─────────────────────────────────────
N_STATES           = 3        # 국면 수 (Bull / Side / Bear)
HMM_N_ITER         = 200      # Baum-Welch 최대 반복 횟수
HMM_RANDOM_RESTART = 30       # Random Restart 횟수, 30이 기본
HMM_COVARIANCE_TYPE = 'diag'  # 'spherical' | 'diag' | 'full'
                              # 현재 'diag' 사용 — 피처 간 독립 가정.
                              # 상관 높은 피처는 HMM_FEATURE_COLS에서 제외 권장.
                              # 나중에 'full'로 전환 가능: 전체 공분산 학습 →
                              # 상관성 자체를 모델이 처리, 더 많은 피처 활용 가능.
                              # (단 데이터가 충분하고 |r| < 0.9일 때만 'full' 안전)


# ─── 정규화 (Scaler) — Phase 2 결정 사항 ───────────────────────
# 세 가지 모드를 비교 학습한다 (Phase 2 검증 단계):
#   'none'    — 정규화 없음 (raw 피처 그대로)
#   'global'  — 전체 학습 데이터로 fit한 sklearn StandardScaler
#   'rolling' — 시점별 과거 ROLLING_SCALER_WINDOW봉의 mean/std로 normalize
#
# Rolling 사용 이유: 자산 가격 스케일이 시간에 따라 크게 변할 때
# 과거 데이터가 현재 정규화에 끼어드는 문제를 방지.
# (예: 2018년 BTC=3,000불, 2024년 BTC=70,000불의 차이가 정규화에 포함되면
#  현재 시점의 상대적 위치 해석을 왜곡함)
SCALER_MODE = 'rolling'        # 'none' | 'global' | 'rolling'
                                # 검증 스크립트에서는 세 모드 모두 학습/비교
                                # 운영 단계에서는 베스트 모드 하나로 고정 예정 (Phase 4)

ROLLING_SCALER_WINDOW = 3300    # 과거 N봉의 mean/std 사용
                                # 30분봉 기준 약 1년 (13봉/일 × 252거래일 ≈ 3,276)
                                # 너무 짧으면 → 장기 국면(Bear 시장 등)이 정규화로 사라짐
                                # 너무 길면 → 시간 드리프트 해결 효과 감소
                                # 주의: 첫 3,300봉은 cold start(정규화 불가).
                                #   종목당 17,589봉 중 ~14,300봉이 실사용 가능.


# ─── 피처 분리 (HMM vs Meta Model) ──────────────────────────────
# HMM과 Meta Model에 서로 다른 피처 부분집합을 입력하고 싶을 때 사용.
# None이면 compute_window_features의 9개 피처를 모두 사용.
#
# 결정 기준:
#   - HMM: 국면을 직접 정의하는 피처만. 상관 높은 피처는 한쪽만.
#          covariance_type='diag'와 짝을 맞춰 신중히 선정.
#   - Meta: HMM 확률 + 보조/노이즈 필터 피처까지 포함 가능.
#
# 추가/제외 방법:
#   1. 아래 리스트에 피처명을 더하거나 빼기만 하면 됨
#   2. 가능한 피처명은 strategy/HMM_strategy/features/window_features.py의
#      FEATURE_COLUMNS 참조 (현재 9개)
#   3. 모든 피처를 사용하려면 None으로 설정
#
# Phase 1 종료 시점 결정 (VIF 분석 결과 기반):
#   - max_drawdown(VIF 12.09) 제외 — 다변량 중복 심각
#   - slope 제외 — cum_return과 페어와이즈 |r|=0.77로 중복
#   - adx_end, r2_end 제외 — "현재 시점" 정보는 Meta가 전이 예측에 활용
HMM_FEATURE_COLS = [
    'cum_return',       # 방향 + 크기 (VIF 6.16, 회색지대지만 핵심 피처)
    'volatility',       # 양방향 변동성 (VIF 6.62)
    'adx_mean',         # 추세 강도 평균 (VIF 2.29)
    'r2_mean',          # 추세 직선성 평균 (VIF 2.30)
    'up_candle_ratio',  # 양봉 비율 (VIF 1.72)
]

META_FEATURE_COLS = None  # Phase 3에서 확정 (현재 None = 9개 전체 + HMM 확률)


# ─── 메타 모델 (Phase 3에서 활성) ───────────────────────────────
META_MODEL_TYPE = 'logistic'   # 'logistic' | 'xgboost' | 'nn'
TS_SPLIT_N      = 5            # TimeSeriesSplit 분할 수
# TS_SPLIT_GAP은 WINDOW_SIZE에 의존하므로 사용처에서 직접 참조


# ─── 모델 캐시 경로 (Phase 3에서 활성) ──────────────────────────
# DATA_PATH 파일명의 첫 토큰(종목 심볼)에서 자동 유도.
# e.g. AAPL_20210101_20260523_30min.parquet → models/hmm_aapl.joblib
# 캐시 파일이 없으면 첫 실행 시 자동 학습 후 저장됨.
from pathlib import Path as _Path
_asset_slug = _Path(DATA_PATH).stem.split('_')[0].lower()
HMM_MODEL_PATH = f"models/hmm_{_asset_slug}.joblib"


# ─── Retrospective Label Smoother (Phase 3 후반 추가) ─────────
# HMM Viterbi 라벨이 급격한 국면 전환(폭락/폭등)을 늦게 따라가는
# 문제를 보정하기 위해, 학습용 정답지를 사후적으로 backdate한다.
#
# ★ 룩어헤드 안전성:
#   - 정답지(y) 개선 용도. 예측 모델 입력(X)에는 영향 없음.
#   - Phase 1 보고서의 룩어헤드 규칙 위반 아님.
#
# 알고리즘:
#   1. HMM Viterbi 라벨에서 전환점 (label[t-1] != label[t]) 찾기
#   2. 후속 N봉(persistence_check) 모두 새 국면이면 진짜 전환으로 인정
#   3. 전환점에서 K봉(lookback) 뒤로 돌아가, |마지막 1봉 수익률|이
#      threshold 초과한 봉이 있으면 그 시점으로 라벨을 backdate
#   4. SIDE 전환은 점진적이므로 backdate 제외 (LABEL_SMOOTHER_INCLUDE_SIDE)
LABEL_SMOOTHER_LOOKBACK         = 10     # 전환점에서 backdate할 최대 봉 수
LABEL_SMOOTHER_THRESHOLD        = 0.03   # |1봉 수익률| 임계값 (3%)
LABEL_SMOOTHER_PERSISTENCE      = 3      # 전환 인정에 필요한 후속 일관성 봉 수
LABEL_SMOOTHER_INCLUDE_SIDE     = False  # SIDE 전환도 backdate할지 (기본 False)


# ─── 포지션 사이저 (Phase 4에서 활성) ───────────────────────────
POSITION_MODE          = 'net'   # 'net' | 'dual'
MIN_POSITION_THRESHOLD = 0.1
REBALANCE_THRESHOLD    = 0.15


# ─── HMMStrategy variant 스위치 (Phase 4에서 활성) ──────────────
# 메타 모델 학습/추론 방식의 변형 옵션. Phase 3 보고서 4-3 참조.
#
# INCLUDE_HMM_PROBA:
#   True  → 메타 입력 X에 HMM 사후확률 3개 + 전이 사전확률 3개 포함 (총 16개 피처)
#   False → 메타 입력에서 HMM 관련 6개 피처 제외 (총 10개 피처:
#                                                  분류기 6 + 윈도우 4)
#   Phase 3에서 메타가 HMM 사후확률에만 의존해 persistence baseline 수준의
#   정확도를 보였으므로, False로 설정해 분류기/피처만으로 학습한 결과와 비교.
#
# USE_SMOOTHED_LABELS:
#   True  → RetrospectiveLabelSmoother로 보정된 라벨을 학습 라벨 y로 사용
#   False → 원본 HMM Viterbi 라벨을 그대로 학습 라벨 y로 사용
#   Phase 3 5장 검증 결과 smoother가 전환시점 정확도를 9.7%→12.9%로 개선.
#
# 룩어헤드 안전성:
#   - INCLUDE_HMM_PROBA: HMM 사후확률은 같은 윈도우 데이터로 계산되므로 룩어헤드 X
#   - USE_SMOOTHED_LABELS: 라벨 보정은 정답지(y)에만 영향, 입력(X)에 영향 없음
INCLUDE_HMM_PROBA   = True
USE_SMOOTHED_LABELS = True

# 메타 모델 하이퍼파라미터 (Phase 3 verify_meta_model.py 기본값과 동일)
META_C            = 1.0
META_CLASS_WEIGHT = 'balanced'   # 'balanced' | None

# ─── 거래량(RVOL) 피처 — meta 입력 전용 variant ─────────────────
# INCLUDE_VOLUME:
#   탈시즌화 RVOL 기반 거래량 피처 3종(rvol_mean/rvol_slope/vol_price_corr)을
#   meta-model 입력에 추가한다. HMM 클러스터링(국면 정의)에는 넣지 않으므로
#   기존 HMM 캐시는 그대로 유효. 룩어헤드 X (슬롯별 rolling median + shift(1)).
#   A/B 비교용 토글 — 기본 False(기존 알파 그대로).
# VOLUME_LOOKBACK_DAYS:
#   RVOL 시즌 기준선 거래일 수 (같은 시간대 과거 N일 중앙값).
# VOLUME_CLIP:
#   log(RVOL) winsorize 범위 [-clip,+clip]. 반일장 등 데이터 아티팩트 절단용.
INCLUDE_VOLUME       = True
VOLUME_LOOKBACK_DAYS = 20
VOLUME_CLIP          = 3.0


# ─── 백테스트 기간 ──────────────────────────────────────────────
# run_backtest_hmm.py의 기본값으로 사용됨.
# train_end는 test_start 하루 전으로 자동 설정되므로 별도 지정 불필요.
TRAIN_START = "2021-01-01"      # 학습 시작일
TEST_START  = "2024-01-01"      # OOS 백테스트 시작일 (train_end = 하루 전 자동)
TEST_END    = "2026-05-22"      # 백테스트 종료일


# ─── 데이터 기간 (검증 스크립트에서 사용) ───────────────────────
VERIFY_START = "2021-01-01"     # 검증 스크립트 시작일 (주식 데이터 시작)
VERIFY_END   = "2026-05-23"     # 검증 스크립트 종료일


# ─── 봉 밀도 / 백테스트 워밍업 (주식 30분봉 기준) ─────────────────
# 미국 정규장은 하루 6.5시간 → 30분봉이면 하루 13봉.
# 코인(24시간 연속)과 달리 "캘린더 일수"와 "거래일·봉 수"가 다르므로,
# 워밍업은 봉 개수(WARMUP_BARS)로 직접 계산하고, 날짜 기반 슬라이싱이
# 필요한 곳을 위해 캘린더 일수(WARMUP_DAYS)는 거래일 밀도로 환산한다.
import math as _math

BARS_PER_DAY          = 13     # 정규장 30분봉 (09:30~15:30)
TRADING_DAYS_PER_YEAR = 252    # 미국 증시 연간 거래일

# generate_signals()가 유효한 출력을 내려면 df_test 앞쪽에 다음 봉 수만큼
# 워밍업이 필요하다 (순차 워밍업의 합):
#   ROLLING_SCALER_WINDOW        : 롤링 정규화 워밍업
#   + WINDOW_SIZE                : compute_window_features 워밍업
#   + max(ADX_PERIOD, R2_PERIOD) : ADX/R² 지표 워밍업
WARMUP_BARS = ROLLING_SCALER_WINDOW + WINDOW_SIZE + max(ADX_PERIOD, R2_PERIOD)

# 봉 수 → 캘린더 일수 (run_backtest_hmm.py가 날짜로 슬라이싱할 때 사용).
# 봉 수 → 거래일 → 캘린더일 환산 후 10일 버퍼.
# ROLLING_SCALER_WINDOW·WINDOW_SIZE·ADX/R2_PERIOD 변경 시 자동 재계산.
_warmup_trading_days = _math.ceil(WARMUP_BARS / BARS_PER_DAY)
WARMUP_DAYS: int = _math.ceil(
    _warmup_trading_days * 365 / TRADING_DAYS_PER_YEAR
) + 10


# ─── 라이브 실행 / 주문 집행 (live_trade.py) ─────────────────────
# live_trade.py 의 모든 튜닝 값을 이 섹션에 모은다.
# (배분비율 자체는 allocations.py 가 source of truth → 여기 중복 안 둠)

# 거래 대상 종목 / 데이터 위치
# plans/candidate_universe_v2.csv 의 50종목 중 GEV 제외. SPY는 거래 대상이
# 아니라 beta benchmark로만 사용한다(LIVE_BETA_BENCHMARK_SYMBOL).
LIVE_SYMBOLS = [
    "META", "GOOGL", "NFLX", "TMUS", "TSLA", "AMZN", "HD", "BKNG",
    "CVNA", "WMT", "COST", "PG", "KO", "PEP", "XOM", "CVX", "COP",
    "SLB", "HOOD", "COIN", "JPM", "BRK.B", "V", "UNH", "LLY", "JNJ",
    "ABBV", "PFE", "BA", "UBER", "CAT", "GE", "NVDA", "AAPL", "MSFT",
    "MU", "AMD", "LIN", "NEM", "FCX", "SHW", "WELL", "AMT", "EQIX",
    "PLD", "CEG", "VST", "NEE", "SO",
]
LIVE_DATA_DIR = "data/30min"     # 종목별 *_30min.parquet 폴더
MARKET_TZ     = "America/New_York"
BAR_MINUTES   = 30               # 봉 간격(분). 30분봉.

# ── 라이브 순베타 캡 ──
# 각 사이클 ET 날짜 기준으로 "오늘 이전" 일별 close-to-close 수익률만 사용해
# SPY 대비 CAPM beta를 추정하고, raw portfolio weight의 순베타를 제한한다.
NET_BETA_CAP = 0.25
LIVE_BETA_LOOKBACK_DAYS = 252
LIVE_BETA_MIN_OBS = 126
LIVE_BETA_BENCHMARK_SYMBOL = "SPY"

# 30분봉 마감 후 IEX 데이터가 정착할 때까지의 대기(초).
SETTLE_BUFFER_SEC = 75

# ── 승인 모델 / EOD 학습 ──
LIVE_MODEL_DIR = "models/live"
LIVE_REQUIRE_APPROVED_MODEL = True
LIVE_MAX_MODEL_STALENESS_TRADING_DAYS = 1
LIVE_MAX_BETA_STALENESS_TRADING_DAYS = 1
LIVE_TRAIN_AFTER_CLOSE_BUFFER_MIN = 30

# ── 루프 재학습 ──
# Phase 2부터 장중 loop는 학습하지 않는다. `--train-eod`가 장 마감 후
# model/beta/manifest를 만들고, 장중 loop는 latest approved artifact를 로드한다.
LIVE_DISABLE_INTRADAY_RETRAIN = True
RETRAIN_EVERY_DAYS  = 0   # 재학습 주기(거래일). 0이면 재학습 안 함.
LIVE_LOOKBACK_YEARS = 5   # EOD 학습에 쓸 최근 연수. 0이면 전체 과거.

# ── 주문 집행 안전장치 ──
# execute 모드는 안전 우선이다. 아래 guard 중 하나라도 실패하면 그 cycle의
# 주문표 전체를 폐기한다(dry-run은 진단 출력만 수행).
LIVE_MAX_CYCLE_AGE_SEC = 300          # 시작~주문 직전 최대 허용 시간
LIVE_EXPECTED_BAR_TOLERANCE_MIN = 5   # 마지막 완성봉 시각 허용 오차
LIVE_REQUIRE_FRESH_ALPACA_DATA = True # 최신 데이터 요청 실패/stale이면 no-trade
LIVE_REQUIRE_ALL_SYMBOLS = True       # 49종목 중 하나라도 실패하면 no-trade
LIVE_MIN_BETA_COVERAGE = 1.0          # execute 기본은 49/49 beta 필요
LIVE_REQUIRE_SHORTABLE = True         # 숏 증가 주문 전 shortable 확인
LIVE_REQUIRE_EASY_TO_BORROW = True    # shortable과 easy_to_borrow 모두 요구
LIVE_CANCEL_MUST_SETTLE = True        # open order 취소 실패/timeout이면 no-trade
LIVE_NO_NEW_ORDERS_BEFORE_CLOSE_MIN = 5

# 사이클 시작 시 '우리 심볼'의 미체결 주문만 취소하고, 모두 정리될 때까지
# 폴링한다(계좌 전체 취소 아님 — 다른 알파와 공존 대비).
CANCEL_SETTLE_TIMEOUT_SEC = 5.0   # 미체결 취소 후 정리 대기 한도(초)
# 부호 반전(롱↔숏) 시 청산 주문이 체결될 때까지 대기 후 신규 진입.
FILL_WAIT_TIMEOUT_SEC     = 10.0  # 청산 체결 대기 한도(초). 초과 시 진입 보류.
POLL_INTERVAL_SEC         = 0.5   # 취소/체결 폴링 간격(초)

# ── 주문 제출 재시도 (연결오류 한정) ──
# Alpaca 서버와의 일시적 전송오류(Connection reset / timeout)에만 재시도.
# wash trade·수량부족 같은 API 거부는 재시도하지 않는다.
# 멱등성은 client_order_id로 보장하므로 중복 체결 위험 없음.
ORDER_MAX_RETRIES       = 3      # 총 시도 횟수 (최초 1회 + 재시도 2회)
ORDER_RETRY_BACKOFF_SEC = 0.5    # 첫 백오프(초). 시도마다 2배: 0.5 → 1.0 → 2.0

# ── 단일 인스턴스 PID 가드 ──
# True면 동일 프로그램 중복 실행을 막는다(이중 주문 방지).
SINGLE_INSTANCE_LOCK = True
LIVE_PID_PATH        = "logs/live_trade.pid"
