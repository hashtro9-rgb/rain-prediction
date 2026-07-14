# Predicting Rain from Weather Conditions - Findings

*A classification study on 2,500 daily weather observations. Reproducible via
`python analysis/rain_analysis.py` (random_state = 42). Data: weather_forecast_data.csv.*

---

## The headline

**A Random Forest predicts rain with a ROC-AUC of 1.000, catching
79 of 79
real rain days on unseen data - despite rain being only 12.6% of all days.**
Humidity and Cloud Cover are the two conditions that matter most.

---

## Why accuracy is a trap here

**The data is imbalanced: 314 rainy days vs 2,186 dry days (12.6% rain).**
A lazy model that *always* predicts "no rain" is right 87.4% of the time - yet it
never warns you about a single storm. **So this study never judges a model on accuracy alone.**
Instead we use ROC-AUC, plus precision, recall and F1 for the rain (positive) class, and the
confusion matrix.

*So what:* headline accuracy would make a useless model look excellent. The metrics that
matter are whether we actually catch the rain.

---

## What does a rainy day look like?

**Rainy days are markedly more humid and cloudier, with slightly lower pressure.** Comparing
group means (no-rain -> rain):

| Feature | No-rain -> Rain (delta) |
|---|---|
| Humidity | 61.5 -> 84.5 % (+23.0) |
| Cloud Cover | 46.1 -> 74.7 % (+28.7) |
| Pressure | 1014.2 -> 1014.8 hPa (+0.5) |
| Temperature | 23.3 -> 17.4 °C (-5.9) |
| Wind Speed | 9.9 -> 9.9 m/s (-0.0) |

*So what:* the physical intuition holds - humid, cloudy, low-pressure air brings rain. These
are the levers a model should lean on.

---

## A simple rule already separates most of the signal

**When humidity is low and skies are clear, the rain rate is
0%; when both humidity and cloud cover are high,
it jumps to 60%.** (See
`charts/04_rainrate_humidity_cloud.png`.)

*So what:* even before machine learning, two features carve the days into low- and high-risk
buckets - which is exactly what the models then formalize.

---

## The models

Both models handle the imbalance with `class_weight="balanced"` and are evaluated on a
stratified 25% held-out test set (625 days, 79 of them rain).

| Model | ROC-AUC | Precision (rain) | Recall (rain) | F1 (rain) | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.956 | 0.525 | 0.924 | 0.670 | 0.885 |
| Random Forest | 1.000 | 0.988 | 1.000 | 0.994 | 0.998 |

**Winner by ROC-AUC: Random Forest (1.000).** The interpretable Logistic Regression is
a strong, close baseline (0.956) - reassuring, because it means the signal is
largely learnable with a simple linear rule.

*So what:* both models rank a random rainy day above a random dry day almost every time - far
above the 50% of a coin flip.

### An honest caveat on that near-perfect Random Forest

The Random Forest scores ROC-AUC 1.000 with only 1 false
alarm and 0 missed storms on the test set. **A score this clean is a
signal in itself: it almost certainly means this dataset was generated from a near-deterministic
rule** on humidity, cloud cover and temperature, rather than sampled from messy real weather.
Real operational data is noisier, and I would expect performance closer to the Logistic
Regression's on live observations. The result is reported honestly as-is, but I would not promise
100% recall in production without validating on real, independently collected data.

---

## Which conditions drive the prediction?

**Random Forest importance ranks Humidity (0.441),
Cloud_Cover (0.328) and Temperature (0.221) on top.**
The Logistic Regression agrees on direction: the strongest push *toward* rain comes from
**Humidity** (coef +2.88, odds ratio 17.76
per +1 SD), while higher **Temperature** pushes toward no-rain
(coef -1.84).

*So what:* the two very different model families agree on the same physical story, which is a
good sign the finding is real rather than an artifact.

---

## The precision / recall tradeoff

At the default 0.50 threshold the winner runs at precision 0.988 /
recall 1.000. Moving the threshold trades one for the other:

| Threshold | Precision | Recall | F1 | Days flagged rain |
|---|---|---|---|---|
| 0.3 | 0.963 | 1.000 | 0.981 | 82 |
| 0.4 | 0.988 | 1.000 | 0.994 | 80 |
| 0.5 | 0.988 | 1.000 | 0.994 | 80 |
| 0.6 | 1.000 | 1.000 | 1.000 | 79 |
| 0.7 | 1.000 | 1.000 | 1.000 | 79 |

*So what:* if missing a storm is costly, lower the threshold to raise recall (at the price of
more false alarms); if false alarms are costly, raise it. The right operating point is a
business decision, not a modeling one.

---

## Future work

A gradient-boosted model (XGBoost or scikit-learn's `HistGradientBoostingClassifier`) would
likely squeeze out marginal gains over the Random Forest, and calibrated probabilities plus
cost-based threshold tuning would sharpen the operating point.

---

### Charts
- `charts/01_class_balance.png` - the imbalance and the accuracy trap
- `charts/02_feature_distributions.png` - conditions by outcome
- `charts/03_correlation_heatmap.png` - feature correlations
- `charts/04_rainrate_humidity_cloud.png` - rain rate by humidity x cloud
- `charts/05_roc_curves.png` - ROC curves, both models
- `charts/06_confusion_matrix.png` - confusion matrix, Random Forest
- `charts/07_precision_recall.png` - precision-recall tradeoff
- `charts/08_rf_importance.png` - Random Forest feature importance
- `charts/09_logistic_coefficients.png` - logistic coefficients & odds ratios
