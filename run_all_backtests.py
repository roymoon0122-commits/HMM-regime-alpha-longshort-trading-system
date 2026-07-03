# 전체 종목 HMM 백테스트 일괄 실행 + 비교 요약
#
# run_backtest_hmm.py 를 종목마다 차례로 실행하고, 각 실행의 콘솔
# 요약표를 파싱해 한 장의 비교표로 정리한다.
# (run_backtest_hmm.py 자체는 수정하지 않는다)
#
# 실행 (프로젝트 루트, venv 활성화 상태에서):
#   python run_all_backtests.py                       # data/30min/ 전 종목 자동 감지
#   python run_all_backtests.py --symbols AAPL TSLA   # 특정 종목만
#
# 산출물:
#   results/backtest_hmm_{종목}.html   — 종목별 상세 차트
#   results/all_stocks_summary.md      — 전 종목 비교표
# ─────────────────────────────────────────────────────────────
import argparse
import re
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("data/30min")
OUT_DIR  = Path("results")


def discover_symbols() -> list[str]:
    """data/30min/ 에 있는 *_30min.parquet 파일에서 종목 심볼을 자동 추출.

    파일명 규칙: {SYMBOL}_{날짜범위}_30min.parquet
    새 종목 파일을 추가하면 별도 코드 수정 없이 자동으로 인식됨.
    """
    files = sorted(DATA_DIR.glob("*_30min.parquet"))
    symbols = [f.name.split('_')[0] for f in files]
    return symbols

# run_backtest_hmm.py 요약표 한 줄 파싱:
#   "  전략라벨 ...   +12.34%   -0.52   -34.39%   95   40.00%"
ROW_RE = re.compile(
    r'^\s+(.+?)\s+([+-][\d.]+)%\s+([+-]?[\d.]+)\s+'
    r'([+-][\d.]+)%\s+([\d,]+)\s+([\d.]+)%\s*$'
)


def run_one(symbol: str):
    """한 종목 백테스트 실행 → {전략라벨: stats} dict 반환 (실패 시 None)."""
    files = sorted(DATA_DIR.glob(f"{symbol}_*_30min.parquet"))
    if not files:
        print(f"  [{symbol}] 데이터 파일 없음 — 건너뜀")
        return None

    cmd = [
        sys.executable, "run_backtest_hmm.py",
        "--csv-path",    str(files[0]),
        "--hmm-cache",   f"models/hmm_{symbol.lower()}.joblib",
        "--output-html", str(OUT_DIR / f"backtest_hmm_{symbol}.html"),
    ]
    print(f"  [{symbol}] 실행 중 ...", flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"  [{symbol}] 시간 초과 (15분)")
        return None
    if proc.returncode != 0:
        print(f"  [{symbol}] 실패 (returncode {proc.returncode})")
        print("  --- stderr 끝부분 ---")
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-12:]))
        return None

    # 요약표 블록만 스캔 ("백테스트 결과 비교" 이후)
    lines = proc.stdout.splitlines()
    try:
        scan_from = next(i for i, l in enumerate(lines) if "결과 비교" in l)
    except StopIteration:
        scan_from = 0

    rows = {}
    for line in lines[scan_from:]:
        m = ROW_RE.match(line)
        if not m:
            continue
        label, cagr, sharpe, mdd, trades, win = m.groups()
        rows[label.strip()] = {
            'cagr':   float(cagr),
            'sharpe': float(sharpe),
            'mdd':    float(mdd),
            'trades': int(trades.replace(',', '')),
            'win':    float(win),
        }
    if not rows:
        print(f"  [{symbol}] 요약표 파싱 실패")
        return None
    print(f"  [{symbol}] 완료 — {len(rows)}개 전략 파싱")
    return rows


def best_hmm(rows: dict):
    """4개 HMM variant 중 Sharpe 최고를 (라벨, stats)로 반환."""
    hmm = {k: v for k, v in rows.items() if k.startswith('HMM')}
    if not hmm:
        return None, None
    label = max(hmm, key=lambda k: hmm[k]['sharpe'])
    return label, hmm[label]


def pick(rows: dict, keyword: str):
    """라벨에 keyword가 들어간 전략 stats 반환."""
    return next((v for k, v in rows.items() if keyword in k), None)


def parse_args():
    p = argparse.ArgumentParser(description="전체(또는 지정) 종목 HMM 백테스트 일괄 실행")
    p.add_argument(
        '--symbols', nargs='+', default=None, metavar='SYM',
        help=(
            "백테스트할 종목 심볼 목록 (예: --symbols AAPL TSLA NVDA). "
            "생략하면 data/30min/ 에 있는 모든 종목을 자동으로 실행."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
        print("=" * 70)
        print(f"  지정 종목 HMM 백테스트: {', '.join(symbols)}")
        print("=" * 70)
    else:
        symbols = discover_symbols()
        if not symbols:
            print(f"[오류] {DATA_DIR}/ 에 *_30min.parquet 파일이 없습니다.")
            sys.exit(1)
        print("=" * 70)
        print(f"  전체 종목 HMM 백테스트 ({len(symbols)}종목 자동 감지)")
        print(f"  종목: {', '.join(symbols)}")
        print("=" * 70)

    results = {}
    for sym in symbols:
        r = run_one(sym)
        if r:
            results[sym] = r

    if not results:
        print("\n결과 없음 — 종료")
        return

    # ── 비교표 작성 ──────────────────────────────────────────
    out = []
    out.append("# 전체 종목 HMM 백테스트 비교\n")
    out.append("OOS 2025-01-01 ~ 2026-05-22, 30분봉. "
               "HMM은 4개 variant 중 Sharpe 최고를 표기.\n")
    out.append("| 종목  | HMM 최고 (variant)        | Donchian          | Buy & Hold        |")
    out.append("|-------|---------------------------|-------------------|-------------------|")

    n = beats_bh = beats_don = 0
    for sym, rows in results.items():
        blabel, b = best_hmm(rows)
        don = pick(rows, 'Donchian')
        bh  = pick(rows, 'Buy')
        if b is None:
            continue
        n += 1
        if bh and b['cagr'] > bh['cagr']:
            beats_bh += 1
        if don and b['cagr'] > don['cagr']:
            beats_don += 1
        variant = blabel.replace('HMM ', '').strip()
        hmm_s = f"{b['cagr']:+6.1f}% Sh{b['sharpe']:+.2f}"
        don_s = f"{don['cagr']:+6.1f}% Sh{don['sharpe']:+.2f}" if don else "—"
        bh_s  = f"{bh['cagr']:+6.1f}% Sh{bh['sharpe']:+.2f}"   if bh  else "—"
        out.append(f"| {sym:<5} | {variant:<13} {hmm_s} | {don_s:<17} | {bh_s:<17} |")

    out.append("")
    out.append(f"**HMM 최고 variant 기준 ({n}종목):** "
               f"Buy&Hold 상회 {beats_bh}/{n}, Donchian 상회 {beats_don}/{n}")
    out.append("")
    # 전체 상세 (4 variant 포함)
    out.append("## 종목별 전체 전략 상세\n")
    for sym, rows in results.items():
        out.append(f"### {sym}")
        out.append("| 전략 | CAGR | Sharpe | MDD | Trades | Win% |")
        out.append("|------|------|--------|-----|--------|------|")
        for label, s in rows.items():
            out.append(f"| {label} | {s['cagr']:+.2f}% | {s['sharpe']:+.2f} | "
                       f"{s['mdd']:+.2f}% | {s['trades']:,} | {s['win']:.1f}% |")
        out.append("")

    report = "\n".join(out)
    summary_path = OUT_DIR / "all_stocks_summary.md"
    summary_path.write_text(report, encoding="utf-8")

    # 콘솔 출력 (비교표 부분만)
    print("\n" + "=" * 70)
    print("\n".join(out[:n + 6]))
    print("=" * 70)
    print(f"\n전체 비교표 저장: {summary_path}")
    print(f"종목별 차트:      {OUT_DIR}/backtest_hmm_*.html")


if __name__ == "__main__":
    main()
