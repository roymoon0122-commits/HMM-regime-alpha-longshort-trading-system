"""
extract_walkforward_signals.py — 하락장 포함 walk-forward(확장윈도우) OOS 시그널 추출기.

extract_oos_signals.py 의 walk-forward 버전. 단일 (TRAIN/TEST) 대신 여러 윈도우를
돌며, 각 윈도우에서 '그 train_end 이전 데이터로만 학습 → 다음 test 구간 시그널 추출'
을 반복해 look-ahead 를 제거한다. test 세그먼트를 이어붙이면 2020·2022 하락장을 포함한
연속 OOS 가 된다. 라이브 설정(매일 재학습·확장윈도우)과도 일치.

확장윈도우 (train_start 고정 = 2016-01-01):
    train 2016–2019 → test 2020      (코로나 급락)
    train 2016–2021 → test 2022      (약세장)
    train 2016–2023 → test 2024–2025 (강세장; 기존 OOS 와 비교용)

시점별 가변 유니버스: 각 윈도우에서 학습봉(train_end 이전)이 MIN_TRAIN_BARS 미만인
종목은 그 윈도우에서 자동 제외(상장 늦은 COIN/HOOD/CEG/UBER/CVNA 등 자연 처리).

★ 함정: HMMStrategy.from_config() 는 config.HMM_MODEL_PATH 캐시를 무조건 로드하므로
   종목별 재학습에는 반드시 hmm_model_path=None 을 넘긴다(원본 extract 와 동일).

재개가능: 이미 만든 (sym, window) part 는 건너뜀. 시간예산(MAX_SECONDS) 도달 시 종료.
분석용 restart 는 N_RESTART(기본 10). 최종 결론 전 30 재확인.

환경변수:
  N_RESTART     HMM 랜덤 재시작 수 (기본 10)
  MAX_SECONDS   1회 호출 시간예산 (기본 38)
  WF_SMOKE=1    스모크테스트: 현재 2021+ 데이터로 단일 윈도우(train2021-23→test2024)만.
                2016 백필 도착 전 파이프라인 검증용.

산출물: analysis/sig_parts_wf/{sym}__{wtag}.parquet
        (전부 완료 시) analysis/walkforward_signals.parquet
        analysis/extract_wf_progress.log
"""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy.HMM_strategy import config
from strategy.HMM_strategy.features.stock_loader import load_resampled_bars
from strategy.HMM_strategy.strategy import HMMStrategy

OUT = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(OUT, "sig_parts_wf"); os.makedirs(PARTS, exist_ok=True)
LOG = os.path.join(OUT, "extract_wf_progress.log")
N_RESTART = int(os.environ.get("N_RESTART", "10"))
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "38"))
MIN_TRAIN_BARS = int(os.environ.get("MIN_TRAIN_BARS", "4000"))  # ≈300거래일, 원본과 동일 기준

# (wtag, train_start, train_end, test_start, test_end)
WINDOWS_FULL = [
    ("2020", "2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2022", "2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2024", "2016-01-01", "2023-12-31", "2024-01-01", "2026-05-22"),
]
# 스모크: 현재 보유한 2021+ 데이터만으로 파이프라인 검증 (단일 윈도우)
WINDOWS_SMOKE = [
    ("2024s", "2021-01-01", "2023-12-31", "2024-01-01", "2026-05-22"),
]
WINDOWS = WINDOWS_SMOKE if os.environ.get("WF_SMOKE") == "1" else WINDOWS_FULL

UNIVERSE = pd.read_csv(os.path.join(OUT, "..", "plans", "candidate_universe_v2.csv"))["sym"].tolist()
UNIVERSE = [s for s in UNIVERSE if s != "GEV"]
SYMBOLS = UNIVERSE + ["SPY"]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def find_file(sym):
    """가장 긴 백필(2016~) 파일 우선. 없으면 가장 최근 파일."""
    fs = sorted(glob.glob(f"data/30min/{sym}_*_30min.parquet"))
    if not fs:
        return None
    # 시작일이 가장 이른 파일을 고름(2016 백필 우선)
    def start_of(p):
        base = os.path.basename(p)
        try:
            return base.split("_")[1]  # YYYYMMDD
        except Exception:
            return "99999999"
    return sorted(fs, key=start_of)[0]


def part_path(sym, wtag):
    return os.path.join(PARTS, f"{sym}__{wtag}.parquet")


def todo_jobs():
    jobs = []
    for w in WINDOWS:
        wtag = w[0]
        for s in SYMBOLS:
            if not os.path.exists(part_path(s, wtag)):
                jobs.append((s, w))
    return jobs


def extract_one(sym, window):
    wtag, tr_start, tr_end, ts, te = window
    tr_start = pd.Timestamp(tr_start); tr_end = pd.Timestamp(tr_end)
    ts = pd.Timestamp(ts); te = pd.Timestamp(te)
    path = find_file(sym)
    if not path:
        log(f"{sym}[{wtag}]: 파일없음"); return None
    df = load_resampled_bars(path, start=tr_start, end=te)
    if df.empty:
        log(f"{sym}[{wtag}]: 빈 데이터"); return None
    dtr = df[df["datetime"] <= tr_end].reset_index(drop=True)
    if len(dtr) < MIN_TRAIN_BARS:
        # 시점별 가변 유니버스: 이 윈도우에선 학습데이터 부족 → 제외 (빈 part로 표식)
        log(f"{sym}[{wtag}]: 학습부족({len(dtr)}) → 윈도우 제외")
        pd.DataFrame(columns=["datetime", "symbol", "signal", "close", "ret_fwd", "window"]
                     ).to_parquet(part_path(sym, wtag), index=False)
        return 0
    if sym == "SPY":
        sig = np.zeros(len(df))
    else:
        s = HMMStrategy.from_config(hmm_model_path=None, hmm_n_random_restart=N_RESTART)
        s.fit(dtr)
        sig = np.asarray(s.generate_signals(df), dtype=float)
    d = pd.DataFrame({
        "datetime": pd.to_datetime(df["datetime"].values),
        "symbol": sym,
        "signal": sig,
        "close": df["close"].astype(float).values,
    })
    d = d[d["datetime"] >= ts].reset_index(drop=True)
    d["ret_fwd"] = d["close"].pct_change().shift(-1)
    d["window"] = wtag
    d.to_parquet(part_path(sym, wtag), index=False)
    nz = int((d["signal"].abs() > 1e-9).sum())
    log(f"{sym}[{wtag}]: OOS {len(d)}봉, sig!=0 {nz}")
    return len(d)


def maybe_combine():
    expected = [(s, w[0]) for w in WINDOWS for s in SYMBOLS]
    have = [(s, wtag) for (s, wtag) in expected if os.path.exists(part_path(s, wtag))]
    if len(have) < len(expected):
        log(f"진행 {len(have)}/{len(expected)} part 완료")
        return
    frames = []
    for (s, wtag) in expected:
        p = part_path(s, wtag)
        try:
            f = pd.read_parquet(p)
            if len(f):
                frames.append(f)
        except Exception as e:
            log(f"part 읽기 실패 {s}[{wtag}]: {e}")
    allp = pd.concat(frames, ignore_index=True).sort_values(["symbol", "datetime"]).reset_index(drop=True)
    out = os.path.join(OUT, "walkforward_signals.parquet")
    allp.to_parquet(out, index=False)
    log(f"전체 결합 저장: walkforward_signals.parquet ({len(allp):,}행, "
        f"{allp['symbol'].nunique()}종목, 윈도우 {sorted(allp['window'].unique())})")


def main():
    mode = "SMOKE(2021+ 단일윈도우)" if os.environ.get("WF_SMOKE") == "1" else "FULL(2016~ 3윈도우)"
    jobs = todo_jobs()
    log(f"=== {mode} | 남은 {len(jobs)} job, restart={N_RESTART}, 예산={MAX_SECONDS}s ===")
    t_start = time.time()
    for sym, window in jobs:
        if time.time() - t_start > MAX_SECONDS:
            log("시간예산 도달 — 종료(재호출로 이어서)"); break
        t0 = time.time()
        try:
            extract_one(sym, window)
        except Exception as e:
            log(f"{sym}[{window[0]}]: 오류 {type(e).__name__}: {str(e)[:90]}")
    maybe_combine()


if __name__ == "__main__":
    main()
