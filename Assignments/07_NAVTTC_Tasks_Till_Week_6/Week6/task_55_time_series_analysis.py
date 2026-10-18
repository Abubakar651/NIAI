# Task 55 - Time Series Analysis with Pandas
# Reference: https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas

import pandas as pd

dates = pd.date_range(start="2024-01-01", periods=30, freq="D")

daily_sales = [120, 135, 110, 145, 160, 155, 170, 165, 180, 175, 190, 185, 200, 210, 195,
               220, 215, 230, 240, 225, 250, 245, 260, 270, 255, 280, 275, 290, 305, 300]

df = pd.DataFrame({"Date": dates, "Sales": daily_sales})
df.set_index("Date", inplace=True)

df["Day_Name"] = df.index.day_name()
df["7_Day_Moving_Avg"] = df["Sales"].rolling(window=7).mean().round(2)

weekly_summary = df["Sales"].resample("W").agg(["sum", "mean", "min", "max"]).round(2)

print("Abubakar's Retail Time Series Data (First 10 Days):")
print(df.head(10))

print("\nWeekly Sales Summary:")
print(weekly_summary)

print("\nTotal Monthly Sales:", df["Sales"].sum())
print("Average Daily Sales:", round(df["Sales"].mean(), 2))
print("Highest Sales Day:", df["Sales"].idxmax().strftime('%Y-%m-%d'), f"({df['Sales'].max()})")
