# 📈 StockCast: Forecasting with Sentiment Intelligence

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stockcast-forecasting-with-sentiment-intel.streamlit.app/)

What if the stock market wasn’t just driven by data... but by people?

What if emotions, opinions, and hashtags had the power to shape financial decisions — and we could predict that?

**StockCast** is my answer to that question.  
It’s not just a machine learning project — it’s a journey through chaos, curiosity, and code.

---

## 🌟 Why I Built This

I’ve always believed that the market is a story — not just of numbers, but of *narratives*.  
And Twitter? It’s where those narratives are born, argued, amplified, and sometimes... exploded.

So I asked myself:
> Can I blend cold, historical stock data with the fiery emotion of live tweets — and make it forecast the future?

This project is my attempt.  
It’s built from scratch, one phase at a time, through wins, retries, and long nights.

---

## 🚀 Live Dashboard

Forecast prices, compare models, explore sentiment overlays — all in your browser.

---

## 🔧 The Journey (Phase-by-Phase)

### 🧩 Phase 1: Turning Tweets into Signals

I began by scraping thousands of tweets related to a chosen stock using `snscrape`.  
These raw, noisy texts were then filtered through two lenses:

- **VADER** — a lexicon-based model, simple yet surprisingly effective
- **RoBERTa** — a powerful transformer model tuned for Twitter sentiment

> I didn’t just want raw sentiment. I wanted *patterns*. So I aggregated them daily, normalized them, and aligned them with actual market activity.

📁 Key Outputs:
- `vader_sentiment_daily.csv`
- `roberta_sentiment_10k.csv`
- Visual sentiment distributions

---

### 📈 Phase 2: Marrying Price with Emotion

Next, I pulled stock price data using `yfinance`.  
I synchronized it with sentiment — turning two chaotic worlds into one coherent dataset:  
`merged_sentiment_stock.csv`.

This dataset became the **foundation** of every model that followed.

---

### 🔮 Phase 3: Forecasting with Prophet

I wanted a simple baseline, so I trained **Facebook Prophet** on price-only data.  
Then I added RoBERTa sentiment as a regressor and watched the forecasts shift.

📊 I visualized:
- Forecasted price vs actual
- Model changepoints
- MAE and RMSE comparisons

> This phase showed me that sentiment isn’t noise. When handled right, it *nudges the curve* in the right direction.

📁 Outputs:
- `roberta_prophet_forecast.csv`
- `vanilla_prophet_forecast.csv`
- Comparison plots

---

### ⚙️ Phase 4: Getting Smart with LightGBM

Prophet was good — but I wanted more power and more features.

So I moved to **LightGBM**:
- Engineered lag features (1-day, 2-day, 3-day)
- Created rolling mean/std features
- Integrated sentiment (daily, 2-day avg, 3-day avg)

Trained 3 models:
1. Without sentiment
2. With sentiment
3. With all advanced features

📊 Results (RMSE ↓ = better):
| Model                 | MAE   | RMSE   |
|----------------------|-------|--------|
| No sentiment         | 0.1424| 0.2059 |
| With sentiment       | 0.1428| 0.2077 |
| Advanced features ✅ | 0.1217| 0.1735 |

> The third model told me: **patterns matter. Emotion matters. Time matters.**

📁 Outputs:
- `lgbm_advanced_preds.csv`
- Forecast plots for each variant
- Tuning results with `GridSearchCV`

---

### 🖥️ Phase 5: Letting the World See It

All of this meant nothing if it stayed on my machine.

So I built a **Streamlit dashboard** where users can:
- Toggle sentiment overlays
- View forecasts from Prophet and LightGBM
- See actual vs predicted prices
- Understand which features matter most

And then... I deployed it.

👉 [Open the app on Streamlit](https://stockcast-forecasting-with-sentiment-intel.streamlit.app/)
You can try it live. Right now.

---

## 🗂️ Project Structure

``` 
📁 data/ 
├── raw/ → Tweet + stock data 
└── processed/ → Cleaned, merged, feature-ready datasets 
📁 outputs/ 
├── forecasts/ → All model forecasts 
├── charts/ → PNG visualizations 
└── tuning/ → Grid search outputs 
📁 scripts/ 
├── sentiment/ → Sentiment analysis + Prophet with regressors 
├── lightgbm/ → Feature engineering + LGBM models 
├── prophet/ → Vanilla + sentiment Prophet models 
├── visualization/ → Comparison plots, changepoints 
└── streamlit_app.py → The interactive dashboard 
📄 requirements.txt → For reproducibility + deployment 
```

## 📊 Tools That Powered This

- **ML & Forecasting:** Prophet, LightGBM, Scikit-learn  
- **Sentiment Analysis:** VADER, Hugging Face RoBERTa  
- **Data Prep:** Pandas, NumPy  
- **Scraping:** `snscrape`  
- **Deployment:** Streamlit Cloud  

---

## 💬 What I Learned

- Sentiment is **not noise** — it’s signal buried under chaos
- Simpler models (like Prophet) offer explainability, but boosting models shine when feature engineering is done right
- Visual dashboards let your work speak for itself

---

## 🎯 Future Directions

- Integrate live tweet streaming with sentiment refresh  
- Add news-based sentiment scoring (Reddit, RSS feeds)  
- Model explainability using SHAP or LIME  
- Train on other sectors: crypto, tech, commodities

---

## 🧠 Final Words

This project taught me more than just modeling.  
It taught me resilience, patience, and the beauty of bringing chaos into clarity.  

If you’re someone who loves blending real-world messiness with elegant ML — welcome.  
Let’s forecast the future together.

---

## 🔗 Launch the App Again

👉 [Launch the app on Streamlit](https://stockcast-forecasting-with-sentiment-intel.streamlit.app/)

---

Built with ❤️ and curiosity.
