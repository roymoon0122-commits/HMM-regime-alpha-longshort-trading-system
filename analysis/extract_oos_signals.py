"""
extract_oos_signals.py — 포트폴리오 배분 분석용 OOS 시그널 추출기 (재개가능).

각 종목을 production variant로 종목별 강제 재학습(hmm_model_path=None) 후
OOS 구간 시그널·종가를 per-symbol parquet로 저장. 이미 저장된 종목은 건너뜀.
시간예산(MAX_SECONDS) 도달 시 깔끔히 종료 → 여러 번 호출해 누적 완료.

분석용이라 HMM restart는 N_RESTART(기본 10)로 낮춤(production 30의 근사).

산출물: analysis/sig_parts/{sym}.parquet  →  (combine 시) analysis/oos_signals.parquet
        analysis/extract_progress.log
"""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy.HMM_strategy import config
from strategy.HMM_strategy.features.stock_loader import load_resampled_bars
from strategy.HMM_strategy.strategy import HMMStrategy

OUT = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(OUT, "sig_parts"); os.makedirs(PARTS, exist_ok=True)
LOG = os.path.join(OUT, "extract_progress.log")
N_RESTART = int(os.environ.get("N_RESTART", "10"))
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "38"))

UNIVERSE = pd.read_csv(os.path.join(OUT, "..", "plans", "candidate_universe_v2.csv"))["sym"].tolist()
UNIVERSE = [s for s in UNIVERSE if s != "GEV"]
SYMBOLS = UNIVERSE + ["SPY"]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def find_file(sym):
    fs = [f for f in sorted(glob.glob(f"data/30min/{sym}_*_30min.parquet")) if "20260527" in f] \
         or sorted(glob.glob(f"data/30min/{sym}_*_30min.parquet"))
    return fs[-1] if fs else None


def main():
    ts = pd.Timestamp(config.TEST_START); te = pd.Timestamp(config.TEST_END)
    tr_end = ts - pd.Timedelta(days=1); warmup = config.WARMUP_BARS
    todo = [s for s in SYMBOLS if not os.path.exists(os.path.join(PARTS, f"{s}.parquet"))]
    log(f"남은 {len(todo)}/{len(SYMBOLS)}종목, restart={N_RESTART}, 예산={MAX_SECONDS}s")
    t_start = time.time()
    for sym in todo:
        if time.time() - t_start > MAX_SECONDS:
            log("시간예산 도달 — 종료(재호출로 이어서)"); break
        t0 = time.time(); path = find_file(sym)
        if not path: log(f"{sym}: 파일없음"); continue
        try:
            df = load_resampled_bars(path, start=pd.Timestamp(config.TRAIN_START), end=te)
            dtr = df[df["datetime"] <= tr_end].reset_index(drop=True)
            if len(dtr) < 4000: log(f"{sym}: 학습부족({len(dtr)})"); continue
            if sym == "SPY":
                sig = np.zeros(len(df))
            else:
                s = HMMStrategy.from_config(hmm_model_path=None, hmm_n_random_restart=N_RESTART)
                s.fit(dtr); sig = np.asarray(s.generate_signals(df), dtype=float)
            d = pd.DataFrame({"datetime": pd.to_datetime(df["datetime"].values),
                              "symbol": sym, "signal": sig, "close": df["close"].astype(float).values})
            d = d[d["datetime"] >= ts].reset_index(drop=True)
            d["ret_fwd"] = d["close"].pct_change().shift(-1)
            d.to_parquet(os.path.join(PARTS, f"{sym}.parquet"), index=False)
            log(f"{sym}: OOS {len(d)}봉, sig!=0 {int((d['signal'].abs()>1e-9).sum())}, {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"{sym}: 오류 {type(e).__name__}: {str(e)[:90]}")
    done = len(glob.glob(os.path.join(PARTS, "*.parquet")))
    log(f"진행 {done}/{len(SYMBOLS)} 완료")
    if done == len(SYMBOLS):
        allp = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(PARTS, "*.parquet"))], ignore_index=True)
        allp.to_parquet(os.path.join(OUT, "oos_signals.parquet"), index=False)
        log(f"전체 결합 저장: oos_signals.parquet ({len(allp):,}행)")


if __name__ == "__main__":
    main()
