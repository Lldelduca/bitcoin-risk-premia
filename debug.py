import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent
df = pd.read_parquet(project_root / 'data' / 'raw' / 'deribit' / 'btc_options_trades.parquet',
                     columns=['date', 'callput'])
df = df[(df['date'] >= '2020-01-12') & (df['date'] <= '2023-08-31')]
print(f"Raw trades in friend's window: {len(df):,}")
print(f"Unique dates: {df['date'].nunique()}")