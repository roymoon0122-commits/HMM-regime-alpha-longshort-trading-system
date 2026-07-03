# 분봉 → 정규장 리샘플 (배치 변환)
#
# data/minute/*_1min.parquet  →  data/30min/*_30min.parquet
#
# ★ 원본 data/minute/ 파일은 절대 건드리지 않는다.
#   리샘플 결과만 새 폴더(data/30min/)에 저장한다.
#
# ★ 아래 설정 블록만 수정하면 타임프레임/폴더를 바꿀 수 있습니다.
#
# 실행 (프로젝트 루트에서):
#   python data/2_resample_bars.py
# ─────────────────────────────────────────────────────────────
import os
import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가 (strategy 패키지 접근용)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy.HMM_strategy.features.stock_loader import load_stock_bars


# ══════════════════════════════════════════════════
# ★ 설정 — 여기만 바꾸세요
# ══════════════════════════════════════════════════
INPUT_DIR  = "data/minute"     # 1분봉 원본 폴더
OUTPUT_DIR = "data/30min"      # 리샘플 결과 폴더 (없으면 자동 생성)
TIMEFRAME  = "30min"           # 리샘플 단위
RTH_ONLY   = True              # True: 정규장(09:30~16:00 ET)만
# ══════════════════════════════════════════════════


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_1min.parquet"))
    if not files:
        print(f"[오류] 입력 파일 없음: {in_dir}/*_1min.parquet")
        return

    print(f"입력: {in_dir}/  ({len(files)}개 파일)")
    print(f"출력: {out_dir}/  (타임프레임={TIMEFRAME}, 정규장만={RTH_ONLY})\n")

    summary = []
    for f in files:
        # AAPL_20210101_20260523_1min.parquet → 'AAPL'
        symbol = f.stem.split("_")[0]
        print(f"  ▶ {symbol:<6} ... ", end="", flush=True)

        try:
            bars = load_stock_bars(str(f), timeframe=TIMEFRAME,
                                   rth_only=RTH_ONLY)
        except Exception as exc:
            print(f"실패: {exc}")
            summary.append((symbol, None))
            continue

        # 파일명: _1min → _30min
        out_name = f.stem.replace("_1min", f"_{TIMEFRAME}") + ".parquet"
        out_path = out_dir / out_name
        bars.to_parquet(out_path, index=False)

        # 검증용 요약
        n = len(bars)
        d0 = bars["datetime"].iloc[0].date()
        d1 = bars["datetime"].iloc[-1].date()
        per_day = bars.groupby(bars["datetime"].dt.date).size()
        print(f"OK  {n:>7,}봉  ({d0} ~ {d1})  "
              f"하루 {int(per_day.median())}봉(중앙값)")
        summary.append((symbol, out_path.name))

    print("\n" + "=" * 60)
    print("완료 요약")
    print("=" * 60)
    for sym, name in summary:
        print(f"  {sym:<6} → {name if name else '실패'}")
    print(f"\n저장 위치: {out_dir}/")


if __name__ == "__main__":
    main()
