import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import requests
from bs4 import BeautifulSoup

st.title("NFL +EV Fanatics Bets")
st.write("Fetches odds from Fanatics and compares with 538 Elo projections to compute +EV bets.")

# --- CONFIG ---
FANATICS_URL = "https://www.fanatics.com/sportsbook/nfl"
stake_unit = st.number_input('Stake unit ($):', value=100, step=10)

# --- FETCH ODDS ---
@st.cache_data
def fetch_fanatics_odds():
    r = requests.get(FANATICS_URL)
    soup = BeautifulSoup(r.text, 'html.parser')
    games = soup.find_all("div", {"data-testid": "event-row"})
    data = []
    for game in games:
        try:
            home = game.find("span", {"data-testid": "team-home-name"}).text.strip()
            away = game.find("span", {"data-testid": "team-away-name"}).text.strip()
            home_odds = int(game.find("div", {"data-testid": "team-home-moneyline"}).text.strip())
            away_odds = int(game.find("div", {"data-testid": "team-away-moneyline"}).text.strip())
            data.append({"home": home, "away": away, "team": home, "price": home_odds})
            data.append({"home": home, "away": away, "team": away, "price": away_odds})
        except:
            continue
    return pd.DataFrame(data)

# --- FETCH 538 ELO PROJECTIONS ---
@st.cache_data
def fetch_projections():
    url = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv'
    df = pd.read_csv(url)
    latest = df.groupby('team1').tail(1)
    projections = latest[['team1','team2','elo1_pre','elo2_pre']].copy()
    projections.rename(columns={'team1':'home','team2':'away','elo1_pre':'elo_home','elo2_pre':'elo_away'}, inplace=True)
    projections['model_prob_home'] = 1 / (1 + 10 ** ((projections['elo_away'] - projections['elo_home']) / 400))
    projections['model_prob_away'] = 1 - projections['model_prob_home']
    return projections[['home','away','model_prob_home','model_prob_away']]

# --- IMPLIED PROBABILITY ---
def implied_prob(odds):
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

# --- EV CALCULATION ---
def compute_ev(odds_df, projections):
    odds_df['implied_prob'] = odds_df['price'].apply(implied_prob)
    merged = odds_df.merge(projections, how='left', left_on=['home','away'], right_on=['home','away'])
    merged['model_prob'] = np.where(
        merged['team'] == merged['home'],
        merged['model_prob_home'],
        merged['model_prob_away']
    )
    merged['alpha'] = merged['model_prob'] - merged['implied_prob']
    merged['expected_value'] = merged['alpha'] * merged['price'] / 100.0
    return merged

# --- KELLY STAKE ---
def kelly_fraction(p, b, q):
    return max(((p * (b + 1) - 1) / b), 0)

# --- MAIN ---
st.write("Fetching data...")
odds_df = fetch_fanatics_odds()
projections = fetch_projections()
ev_df = compute_ev(odds_df, projections)

results = []
for _, row in ev_df.iterrows():
    p = row['model_prob']
    dec_odds = (row['price'] / 100 + 1) if row['price'] > 0 else (100 / abs(row['price']) + 1)
    q = 1 - p
    kelly = kelly_fraction(p, dec_odds - 1, q)
    results.append({
        'match': f"{row['away']} @ {row['home']}",
        'team': row['team'],
        'odds': row['price'],
        'implied': row['implied_prob'],
        'model': row['model_prob'],
        'alpha': row['alpha'],
        'stake': round(stake_unit * kelly * 0.5,2)
    })

final_df = pd.DataFrame(results).sort_values('alpha', ascending=False)

st.subheader("Top +EV Bets")
st.dataframe(final_df.head(20))

st.download_button(
    label="Download full +EV CSV",
    data=final_df.to_csv(index=False).encode('utf-8'),
    file_name='nfl_ev_ranked_fanatics.csv',
    mime='text/csv'
)

# --- CALIBRATION PLOT (demo) ---
y_true = np.random.binomial(1,0.5,size=100)
y_prob = np.random.uniform(0.4,0.6,size=100)
prob_true, prob_pred = calibration_curve(y_true,y_prob,n_bins=10)
fig, ax = plt.subplots()
ax.plot(prob_pred,prob_true,marker='o')
ax.plot([0,1],[0,1],linestyle='--')
ax.set_title('Calibration Plot (Demo)')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('True Probability')
st.pyplot(fig)
