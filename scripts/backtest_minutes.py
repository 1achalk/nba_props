"""Backtest the positional minutes projection model.

Historically simulates the injury report by looking at who actually
sat out the game, preventing Point-in-Time data leakage.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd

from src.nba_pipeline.config import DB_PATH
from src.nba_pipeline.minutes_model import get_team_minutes_projection

# We will intercept pandas' SQL reader to inject historical injuries
_original_read_sql = pd.read_sql
_retro_injuries = {}

def _mock_read_sql(sql, con, *args, **kwargs):
    """Intercepts the player_injuries query to use point-in-time historical scratches."""
    if "FROM player_injuries" in sql:
        df = pd.DataFrame(list(_retro_injuries.items()), columns=["player_id", "status_normalized"])
        return df
    return _original_read_sql(sql, con, *args, **kwargs)

pd.read_sql = _mock_read_sql


def main():
    conn = sqlite3.connect(DB_PATH)

    # Test window: Let's use a solid chunk of the late season
    test_start = "2025-03-01"
    test_end = "2025-04-13"

    print(f"Loading games for backtest window: {test_start} to {test_end}...")

    # 1. Get all team-games in the test window
    games = pd.read_sql("""
        SELECT game_id, game_date, home_team_id, away_team_id
        FROM games 
        WHERE game_date >= ? AND game_date <= ?
    """, conn, params=(test_start, test_end))

    # 2. Get the actual box scores for the test window to evaluate against
    actuals = pd.read_sql("""
        SELECT pb.game_id, pb.team_id, pb.player_id, pb.minutes
        FROM player_box pb 
        JOIN games g ON pb.game_id = g.game_id
        WHERE g.game_date >= ? AND g.game_date <= ?
    """, conn, params=(test_start, test_end))

    results = []

    # Iterate through unique team-games
    team_games = actuals[['game_id', 'team_id']].drop_duplicates()
    
    print(f"Running positional redistribution on {len(team_games)} team-games. This may take a minute...")
    
    for idx, row in team_games.iterrows():
        gid = row['game_id']
        tid = row['team_id']
        
        # Find the game date
        game_date = games[games['game_id'] == gid]['game_date'].iloc[0]
        
        # Find who actually played in this specific game
        actual_roster = actuals[(actuals['game_id'] == gid) & (actuals['team_id'] == tid)]
        active_pids = set(actual_roster[actual_roster['minutes'] > 0]['player_id'])
        
        # Find who played for the team historically but sat out today (Simulated Injuries)
        historical_roster = pd.read_sql("""
            SELECT DISTINCT player_id FROM player_box 
            WHERE team_id = ? AND game_id IN (
                SELECT game_id FROM games WHERE game_date < ?
            )
        """, conn, params=(tid, game_date))['player_id'].tolist()
        
        global _retro_injuries
        _retro_injuries = {}
        for pid in historical_roster:
            if pid not in active_pids:
                _retro_injuries[pid] = "OUT"  # Mock them as injured
            else:
                _retro_injuries[pid] = "HEALTHY"

        # Ask the model for the projection (it will use the mocked injuries and positional buckets)
        try:
            projections = get_team_minutes_projection(tid, game_date, DB_PATH)
        except Exception:
            continue
            
        # Compare to actuals
        for pid, proj in projections.items():
            # If the model expected them to play 0 (because we told it they were OUT), we ignore them 
            # for the error calculation, as we only care about how well it distributed the active minutes.
            if proj.expected == 0.0:
                continue
                
            actual_mins = actual_roster[actual_roster['player_id'] == pid]['minutes'].sum() if pid in active_pids else 0.0
            
            results.append({
                "game_date": game_date,
                "team_id": tid,
                "player_id": pid,
                "bucket": proj.debug.get("bucket", "UNK"),
                "predicted": proj.expected,
                "actual": actual_mins,
                "error": proj.expected - actual_mins,
                "abs_error": abs(proj.expected - actual_mins)
            })

    # Evaluate Results
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 65)
    print("BACKTEST RESULTS — POSITIONAL INJURY MINUTES MODEL")
    print("=" * 65)
    print(f"  Sample size:          {len(df):,} player-games")
    print(f"  Mean error (bias):    {df['error'].mean():+.2f} min (Ideally close to 0)")
    print(f"  Mean abs error (MAE): {df['abs_error'].mean():.2f} min")
    print(f"  RMSE:                 {np.sqrt((df['error']**2).mean()):.2f} min")
    print(f"  Median abs error:     {df['abs_error'].median():.2f} min")
    
    print("\n  [MAE by Positional Bucket]")
    for bucket in ['G', 'F', 'C']:
        b_df = df[df['bucket'] == bucket]
        if not b_df.empty:
            print(f"    {bucket} (Guards/Wings/Centers): {b_df['abs_error'].mean():.2f} min")

    print("\n  [Largest Vacuum Adjustments (Where the model shifted massive minutes)]")
    # Sort by error to see where it got it wrong, or just show the highest predicted
    for _, r in df.nlargest(5, "predicted")[["game_date", "player_id", "bucket", "predicted", "actual", "abs_error"]].iterrows():
        print(f"    {r['game_date']} | PID: {r['player_id']} ({r['bucket']}) | Pred: {r['predicted']:.1f} | Act: {r['actual']:.1f} | Err: {r['abs_error']:.1f}")

if __name__ == "__main__":
    main()
