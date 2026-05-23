# f1_race_analyzer.py
"""
F1 Race Performance & Strategy Analyst
Data Analyst–style project using FastF1 + Pandas + Matplotlib.
"""

import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: change cache path
# fastf1.Cache.enable_cache("cache")

# --- Step A: Load a specific race session ---
YEAR = 2025
GP = "Monaco"      # e.g. "Monaco", "Silverstone", "Spa", etc.
SESSION_TYPE = "R" # "R" = Race, "Q" = Qualifying, "S" = Sprint

session = fastf1.get_session(YEAR, GP, SESSION_TYPE)
session.load()

laps = session.laps
results = session.results

print(f"Loaded {len(laps)} laps for {len(session.drivers)} drivers in {YEAR} {GP} {SESSION_TYPE}.")

# --- Step B: Clean and enrich laps data ---
df = laps.copy()

# Keep only “on‑track” completed laps
df = df[df["LapNumber"] > 0]
df = df[df["PitOutTime"].isna() | df["PitInTime"].isna()]

# Convert Timedelta to float seconds
df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

# Map driver abbreviations → full names
drv_map = {row["Abbreviation"]: row["FullName"] for _, row in results.iterrows()}
df["DriverName"] = df["Driver"].map(drv_map)

# Keep only the columns you want
df = df[[
    "Driver", "DriverName", "LapNumber",
    "LapTimeSeconds", "Compound", "TyreLife", "Stint"
]].dropna(subset=["LapTimeSeconds"])

print(f"After cleaning: {len(df)} laps.")

# --- Step C: Per‑driver performance stats ---
stats = (
    df
    .groupby("Driver", as_index=False)
    .agg(
        n_laps=("LapTimeSeconds", "count"),
        avg_lap=("LapTimeSeconds", "mean"),
        min_lap=("LapTimeSeconds", "min"),
        max_lap=("LapTimeSeconds", "max"),
    )
)

# Merge driver names
stats = stats.merge(
    df[["Driver", "DriverName"]].drop_duplicates(subset=["Driver"]),
    on="Driver"
)

# Estimate pit‑stop count from Stint
stint_max = df.groupby("Driver")["Stint"].max() - 1
stint_max.name = "n_stops"
stats = stats.merge(stint_max.reset_index(), on="Driver", how="left")

# --- Time formatting helper ---
# --- Time formatting helper ---
def time_str(tsec):
    m = int(tsec // 60)
    s = tsec % 60
    return f"{m:02.0f}:{s:06.3f}"

stats["avg_lap_str"] = stats["avg_lap"].apply(time_str)
stats["min_lap_str"] = stats["min_lap"].apply(time_str)

print("\nStats columns:", list(stats.columns))

# --- Sort and print summary ---
summary_df = (
    stats[["Driver", "DriverName", "n_laps", "avg_lap", "avg_lap_str", "min_lap_str", "n_stops"]]
    .sort_values("avg_lap")
)

print("\n=== Driver performance summary ===")
print(summary_df[["Driver", "DriverName", "n_laps", "avg_lap_str", "min_lap_str", "n_stops"]].to_string(index=False))

# --- Step D: Tire‑stint strategy analysis ---
stint_summary = (
    df.groupby(["Driver", "Stint", "Compound"], as_index=False)
    .agg(
        n_laps=("LapNumber", "count"),
        first_lap=("LapNumber", "min"),
        last_lap=("LapNumber", "max"),
        mean_lap=("LapTimeSeconds", "mean"),
    )
)

stint_summary = stint_summary.merge(
    df[["Driver", "DriverName"]].drop_duplicates(subset=["Driver"]),
    on="Driver"
)

print("\n=== Tire‑stint strategy (per driver) ===")
print(
    stint_summary[["Driver", "DriverName", "Compound", "n_laps", "first_lap", "last_lap", "mean_lap"]]
    .sort_values(["Driver", "first_lap"])
    .head(20)
    .to_string(index=False)
)

# --- Step E: Plot 1 – Lap‑time trend for top 5 drivers ---
top5 = stats.nsmallest(5, "avg_lap")["Driver"].tolist()
top5_laps = df[df["Driver"].isin(top5)].copy()

colors = sns.color_palette("husl", len(top5))
colormap = dict(zip(top5, colors))

plt.figure(figsize=(12, 6))
for drv in top5:
    drv_data = top5_laps[top5_laps["Driver"] == drv]
    plt.plot(
        drv_data["LapNumber"],
        drv_data["LapTimeSeconds"],
        label=f"{drv} ({drv_data['DriverName'].iloc[0]})",
        color=colormap[drv],
        alpha=0.8,
        linewidth=1.2
    )

plt.xlabel("Lap Number")
plt.ylabel("Lap Time (seconds)")
plt.title(f"Lap‑time evolution – Top 5 drivers – {YEAR} {GP} {SESSION_TYPE}")
plt.legend(loc="upper left", fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Step F: Bar plot – Avg lap by tyre compound (per driver) ---
plt.figure(figsize=(10, 6))
tmp = df[df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
tmp = tmp.groupby(["Driver", "Compound"], as_index=False)["LapTimeSeconds"].mean()

sns.barplot(
    data=tmp,
    x="Driver",
    y="LapTimeSeconds",
    hue="Compound",
    palette="Set1",
    errorbar=None
)

plt.title(f"Average lap time by tyre compound – {YEAR} {GP} {SESSION_TYPE}")
plt.xlabel("Driver")
plt.ylabel("Avg lap time (seconds)")
plt.legend(title="Compound")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nF1 data analyst project.")
# --- Step G: Lap‑time consistency (per driver) ---
print("\n=== Lap‑time consistency (per driver) ===")

consistency = (
    df
    .groupby("Driver", as_index=False)
    .agg(
        mean_lap=("LapTimeSeconds", "mean"),
        std_lap=("LapTimeSeconds", "std"),          # consistency
        cv=("LapTimeSeconds", lambda x: x.std() / x.mean()),  # coefficient of variation
        n_laps=("LapTimeSeconds", "count")
    )
    .merge(df[["Driver", "DriverName"]].drop_duplicates(), on="Driver")
    .sort_values("mean_lap")
)

# Add readable time columns
consistency["mean_lap_str"] = consistency["mean_lap"].apply(time_str)
consistency["std_lap_str"] = consistency["std_lap"].apply(lambda s: f"{s:.3f}")

print(
    consistency[["Driver", "DriverName", "n_laps", "mean_lap_str", "std_lap_str", "cv"]]
    .round({"cv": 4})
    .to_string(index=False)
)

# Plot: mean lap vs variability (error bars)
plt.figure(figsize=(10, 6))
plt.errorbar(
    x=consistency["Driver"],
    y=consistency["mean_lap"],
    yerr=consistency["std_lap"],
    fmt="o",
    markersize=6,
    alpha=0.8
)

plt.title("Avg lap vs variability (error bars = std)")
plt.xlabel("Driver")
plt.ylabel("Avg lap time (seconds)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- Step H: Tyre degradation within each stint (per driver) ---
print("\n=== Tyre degradation within stint (top 3 drivers) ===")

# Limit to top‑performing drivers from stats
top3_drivers = stats.nsmallest(3, "avg_lap")["Driver"].tolist()
top3_df = df[df["Driver"].isin(top3_drivers)].copy()

# Add lap‑number within stint
top3_df["LapInStint"] = (
    top3_df.groupby(["Driver", "Stint"])["LapNumber"]
    .rank(method="first")
    .astype(int)
)

# Compute per‑stint degradation trend
degradation = (
    top3_df
    .groupby(["Driver", "Stint", "Compound", "LapInStint"], as_index=False)
    .agg(mean_lap=("LapTimeSeconds", "mean"))
    .merge(df[["Driver", "DriverName"]].drop_duplicates(), on="Driver")
)

plt.figure(figsize=(12, 6))
for drv in top3_drivers:
    drv_data = degradation[degradation["Driver"] == drv]
    for comp in drv_data["Compound"].unique():
        comp_data = drv_data[drv_data["Compound"] == comp]
        plt.plot(
            comp_data["LapInStint"],
            comp_data["mean_lap"],
            label=f"{drv} - {comp}",
            marker="o",
            markersize=4,
            alpha=0.8
        )

plt.title("Tyre degradation within stint (top 3 drivers)")
plt.xlabel("Lap in stint")
plt.ylabel("Avg lap time (seconds)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- Step I: Positional progression (if position data is available) ---
print("\n=== Positional progression (approximate) ===")

try:
    positions = session.pos_data
    pos_df = []

    for drv_idx in range(len(positions)):
        drv_abbr = session.drivers[drv_idx]
        drv_pos = positions[drv_idx]
        pos_df.append(
            pd.DataFrame({
                "Driver": drv_abbr,
                "LapNumber": range(1, len(drv_pos) + 1),
                "Position": drv_pos
            })
        )

    pos_df = pd.concat(pos_df, ignore_index=True)
    pos_df = pos_df.merge(
        df[["Driver", "DriverName"]].drop_duplicates(),
        on="Driver"
    )

    # Compute positions at start vs end
    start_pos = (
        pos_df[pos_df["LapNumber"] == 1]
        [["Driver", "Position"]]
        .rename(columns={"Position": "StartPos"})
    )
    end_pos = (
        pos_df.groupby("Driver")["Position"]
        .last()
        .reset_index()
        .rename(columns={"Position": "EndPos"})
    )

    pos_diff = (
        start_pos.merge(end_pos, on="Driver")
        .merge(df[["Driver", "DriverName"]].drop_duplicates(), on="Driver")
    )

    pos_diff["PosChange"] = pos_diff["StartPos"] - pos_diff["EndPos"]
    pos_diff = pos_diff.sort_values("PosChange", ascending=False)

    print(
        pos_diff[["Driver", "DriverName", "StartPos", "EndPos", "PosChange"]]
        .to_string(index=False)
    )

    # Plot: top 5 position gainers
    top_gainers = pos_diff.nlargest(5, "PosChange")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        top_gainers["DriverName"],
        top_gainers["PosChange"],
        color=sns.color_palette("husl", 5)
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.title("Position gainers vs start position")
    plt.xlabel("Driver")
    plt.ylabel("Positions gained")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Could not load position data (maybe not available for this session):", str(e))


# --- Step J: Weather vs performance (air/wet conditions) ---
print("\n=== Weather vs performance (air temp vs lap time) ===")

try:
    weather = session.weather_data
    wt_df = (
        weather[["Time", "AirTemp", "TrackTemp"]]
        .dropna()
        .copy()
    )

    # Map each lap in df to its Time (nearest weather sample)
    # Simplification: use the same Time for all laps
    wt_df["Time"] = pd.to_datetime(wt_df["Time"], utc=True)
    wt_df["air_temp"] = wt_df["AirTemp"].astype(float)
    wt_df["track_temp"] = wt_df["TrackTemp"].astype(float)

    # For portfolio, we’ll just plot overall trend
    plt.figure(figsize=(10, 6))
    plt.scatter(
        wt_df["TrackTemp"],
        wt_df["AirTemp"],
        alpha=0.6,
        c="skyblue",
        edgecolors="black"
    )
    plt.title("Track vs Air temperature during the race")
    plt.xlabel("Track Temperature (°C)")
    plt.ylabel("Air Temperature (°C)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Could not load weather data (maybe not available for this session):", str(e))


print("\n F1 data analyst project finished with 4 new analyses.")