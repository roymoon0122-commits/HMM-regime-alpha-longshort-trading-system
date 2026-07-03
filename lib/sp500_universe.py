"""특정 날짜 시점의 S&P 500 멤버십을 재구성하는 모듈."""

import pandas as pd


def load_data(data_dir="data"):
    current = pd.read_csv(f"{data_dir}/sp500_current.csv")
    current_symbols = set(current["Symbol"].tolist())

    changes = pd.read_csv(f"{data_dir}/sp500_changes.csv", header=[0, 1])
    changes.columns = [
        "_".join(col).strip() if col[0] != col[1] else col[0]
        for col in changes.columns
    ]
    changes["Effective Date"] = pd.to_datetime(changes["Effective Date"])

    return current_symbols, changes


def get_sp500_members_at(date, current_symbols, changes):
    target_date = pd.to_datetime(date)
    members = current_symbols.copy()

    future_changes = changes[changes["Effective Date"] > target_date]

    for _, row in future_changes.iterrows():
        added = row["Added_Ticker"]
        removed = row["Removed_Ticker"]

        if pd.notna(added) and added in members:
            members.discard(added)
        if pd.notna(removed):
            members.add(removed)

    return members
