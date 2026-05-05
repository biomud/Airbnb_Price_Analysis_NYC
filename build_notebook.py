"""Build the rewritten notebook. Run once, then execute the .ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ---------- Title ----------
md("""# Analyze Factors Influencing Airbnb Listing Price in NYC

**Authors:** Wendy Fu & Kerry Zhang

We use the Inside Airbnb New York City listings snapshot (November 1, 2025) to look at how nightly listing price varies with location, room type, host behavior, and review metrics, and then train a few regression models to predict price. The questions below guide the analysis.""")

# ---------- 1. Project Goals ----------
md("""## 1. Project Goals

1. How does nightly price differ across boroughs, neighborhoods, and room types?
2. Which features (capacity, reviews, host behavior, availability) line up with price?
3. Are borough-level price differences statistically meaningful?
4. Can simple regression models from class beat a naive baseline at predicting price?
5. Inside the best model, which features end up being the most useful predictors?

We focus on association, not causation. The dataset is one snapshot, so results describe listed prices rather than booked revenue.""")

# ---------- imports ----------
co("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme()
plt.rcParams["figure.figsize"] = (10, 6)""")

# ---------- 2. Load Data ----------
md("""## 2. Load Data

The file is the detailed listings table from Inside Airbnb. It contains listing characteristics, host attributes, location columns, availability, and review metrics.""")

co("""df = pd.read_csv("listings.csv", low_memory=False)
print("Rows and columns:", df.shape)
df.head()""")

co("""df.info()""")

# ---------- 3. Data Cleaning ----------
md("""## 3. Data Cleaning

The `price` column is stored as text like `$87.00`, so it has to be converted to a number. Listings without a price cannot be used for price prediction and are dropped. Airbnb prices are also heavily right-skewed; a few extreme listings can dominate averages and regression coefficients, so we trim the bottom and top 1% of prices.""")

co("""data = df.copy()

# price is stored as text with $ and commas
data["price"] = (data["price"].astype(str)
                              .str.replace("$", "", regex=False)
                              .str.replace(",", "", regex=False))
data["price"] = pd.to_numeric(data["price"], errors="coerce")

print("Original rows:", len(data))
print("Listings with a usable price:", data["price"].notna().sum())
print("Missing price share:", round(data["price"].isna().mean() * 100, 2), "%")

data = data.dropna(subset=["price"])
data = data[data["price"] > 0]

lower_price = data["price"].quantile(0.01)
upper_price = data["price"].quantile(0.99)
data = data[(data["price"] >= lower_price) & (data["price"] <= upper_price)]

print("Rows after price cleaning:", len(data))
print("Kept price range: $", round(lower_price, 2), "to $", round(upper_price, 2))""")

co("""# percentage columns are also text
for col in ["host_response_rate", "host_acceptance_rate"]:
    data[col] = data[col].astype(str).str.replace("%", "", regex=False)
    data[col] = pd.to_numeric(data[col], errors="coerce")

# t/f columns -> 1/0
for col in ["host_is_superhost", "host_has_profile_pic",
            "host_identity_verified", "instant_bookable"]:
    data[col] = data[col].map({"t": 1, "f": 0})

# years on the platform, anchored to the snapshot date
snapshot_date = pd.Timestamp("2025-11-01")
data["host_since"] = pd.to_datetime(data["host_since"], errors="coerce")
data["host_years"] = (snapshot_date - data["host_since"]).dt.days / 365.25

data["has_license"] = data["license"].notna().astype(int)
data["amenities_count"] = (data["amenities"].fillna("")
                                             .apply(lambda x: 0 if x == "" else x.count(",") + 1))

data[["price", "host_response_rate", "host_acceptance_rate",
      "host_years", "amenities_count"]].head()""")

# ---------- 4. EDA ----------
md("""## 4. Exploratory Data Analysis

We start with the overall price distribution, then move to borough, room type, neighborhood, capacity, and review patterns. Borough is a useful first cut, but each borough contains many neighborhoods with very different prices, so the analysis also drills down within boroughs.""")

co("""data["price"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])""")

co("""sns.histplot(data["price"], bins=60, kde=True)
plt.title("Distribution of Nightly Prices After Outlier Filtering")
plt.xlabel("Nightly Price ($)")
plt.ylabel("Number of Listings")
plt.show()""")

co("""# quick distribution diagnostics from the stats material
skew_val = stats.skew(data["price"])
skew_p   = stats.skewtest(data["price"]).pvalue
print(f"Skewness: {skew_val:.2f} (skewtest p-value: {skew_p:.3g})")""")

md("""Even after trimming the top and bottom 1%, the price distribution is still strongly right-skewed (skewness around 3.5, skewtest p-value far below 0.05). This is why the rest of the EDA leans on medians instead of means whenever it compares groups.""")

# Borough overview
md("""### 4.1 Borough Overview""")

co("""borough_summary = (data.groupby("neighbourhood_group_cleansed")["price"]
                     .agg(["count", "mean", "median"])
                     .sort_values("median", ascending=False))
borough_summary""")

co("""sns.boxplot(data=data, x="neighbourhood_group_cleansed", y="price",
            order=borough_summary.index)
plt.title("Price Distribution by Borough")
plt.xlabel("Borough")
plt.ylabel("Nightly Price ($)")
plt.xticks(rotation=30)
plt.show()""")

md("""Manhattan has the highest median nightly price (around $222) and Bronx the lowest (around $95), so the Manhattan median is roughly 2.3x the Bronx median. The boxplots also show heavy overlap between boroughs, so borough alone is not enough to predict whether a specific listing is expensive or cheap.""")

# Borough x Room type
md("""### 4.2 Room Type Within Borough""")

co("""room_borough = pd.pivot_table(data, values="price",
                              index="neighbourhood_group_cleansed",
                              columns="room_type", aggfunc="median")
room_borough""")

co("""room_borough.plot(kind="bar")
plt.title("Median Price by Borough and Room Type")
plt.xlabel("Borough")
plt.ylabel("Median Nightly Price ($)")
plt.xticks(rotation=30)
plt.legend(title="Room Type")
plt.show()""")

md("""The entire-home premium over a private room is far from constant: about $133 in Manhattan, $108 in Brooklyn, $78 in Queens, $64 in Bronx, and only $38 in Staten Island. So the same room-type upgrade is worth very different amounts in different boroughs. This is exactly the kind of interaction a single borough average hides.""")

# Capacity
md("""### 4.3 Guest Capacity""")

co("""capacity_summary = (data[data["accommodates"] <= 10]
                    .groupby("accommodates")["price"]
                    .agg(["count", "median", "mean"]))
capacity_summary["median_step_up"] = capacity_summary["median"].diff()
capacity_summary""")

co("""sns.lineplot(data=capacity_summary, x=capacity_summary.index, y="median", marker="o")
plt.title("Median Price by Guest Capacity")
plt.xlabel("Accommodates")
plt.ylabel("Median Nightly Price ($)")
plt.show()""")

md("""Median price rises with capacity but the per-guest step-up is uneven, so adding more capacity does not give a fixed dollar boost. Listings that accommodate 1-2 guests sit well below the 4+ tier, but the gap between 6 and 10 guests is small relative to the gap between 2 and 4.""")

# Neighborhood within borough
md("""### 4.4 Neighborhood Variation Within Boroughs

Boroughs are too coarse on their own: Manhattan contains very expensive neighborhoods like Midtown and the Financial District, but also less expensive areas like parts of Inwood. To avoid noisy comparisons, we only keep neighborhoods with at least 100 listings after the price cleaning.""")

co("""neighborhood_by_borough = (data.groupby(
                                ["neighbourhood_group_cleansed", "neighbourhood_cleansed"])["price"]
                            .agg(count="count", median_price="median", mean_price="mean")
                            .reset_index())

large_neighborhoods = neighborhood_by_borough[neighborhood_by_borough["count"] >= 100].copy()

top_neighborhoods_by_borough = (large_neighborhoods
                                .sort_values(["neighbourhood_group_cleansed", "median_price"],
                                             ascending=[True, False])
                                .groupby("neighbourhood_group_cleansed")
                                .head(5))
top_neighborhoods_by_borough""")

co("""g = sns.catplot(data=top_neighborhoods_by_borough,
                x="median_price", y="neighbourhood_cleansed",
                col="neighbourhood_group_cleansed", col_wrap=2,
                kind="bar", sharex=False, height=4, aspect=1.4)
g.set_axis_labels("Median Nightly Price ($)", "Neighborhood")
g.set_titles("{col_name}")
plt.show()""")

md("""Within Manhattan, the highest-priced neighborhoods (Midtown, Hell's Kitchen, Theater District) sit far above the borough median, while within Brooklyn, neighborhoods like Williamsburg and DUMBO drive most of the borough's high-end. Bronx and Staten Island are flatter. The takeaway: a host pricing against the "Manhattan median" is using the wrong reference point if their listing is in a much pricier or cheaper neighborhood.""")

# NEW: Neighborhood x room_type for top neighborhoods
md("""### 4.5 Neighborhood x Room Type

The recommendations below say a host should compare against the same neighborhood **and** the same room type. To support that, we look at median price for the three most-listed neighborhoods in each borough, broken out by room type.""")

co("""# pick the three biggest neighborhoods inside each borough
biggest = (data.groupby(["neighbourhood_group_cleansed", "neighbourhood_cleansed"])
                .size().reset_index(name="n")
                .sort_values(["neighbourhood_group_cleansed", "n"],
                             ascending=[True, False])
                .groupby("neighbourhood_group_cleansed").head(3))

mask = data.set_index(["neighbourhood_group_cleansed", "neighbourhood_cleansed"]).index.isin(
    list(zip(biggest["neighbourhood_group_cleansed"], biggest["neighbourhood_cleansed"]))
)

nbhd_room = (data[mask]
             .pivot_table(values="price",
                          index=["neighbourhood_group_cleansed", "neighbourhood_cleansed"],
                          columns="room_type", aggfunc="median"))
nbhd_room""")

co("""nbhd_room[["Entire home/apt", "Private room"]].plot(kind="bar")
plt.title("Median Price by Neighborhood and Room Type")
plt.xlabel("Borough, Neighborhood")
plt.ylabel("Median Nightly Price ($)")
plt.xticks(rotation=60, ha="right")
plt.legend(title="Room Type")
plt.tight_layout()
plt.show()""")

md("""Even within one borough, the entire-home / private-room gap shifts a lot from neighborhood to neighborhood. For example, an entire home in Midtown commands a much larger premium over a Midtown private room than the equivalent comparison in Astoria or Bedford-Stuyvesant. Neighborhood + room type together is a much tighter benchmark than borough alone.""")

# Reviews
md("""### 4.6 Reviews""")

co("""review_cols = ["price", "number_of_reviews", "reviews_per_month",
               "review_scores_rating", "review_scores_cleanliness",
               "review_scores_location", "review_scores_value"]

data[review_cols].corr(numeric_only=True)["price"].sort_values(ascending=False)""")

co("""sns.scatterplot(data=data.sample(min(3000, len(data)), random_state=42),
                x="review_scores_rating", y="price", alpha=0.35)
plt.title("Review Rating vs Price (sample of 3000)")
plt.xlabel("Review Score Rating")
plt.ylabel("Nightly Price ($)")
plt.show()""")

md("""The Pearson correlations between price and the review-related columns are all small: rating ≈ 0.04, cleanliness ≈ 0.07, location ≈ 0.11, value ≈ 0.00. The scatter plot makes the reason clear: most listings sit at very high ratings (4.5+), so the rating column has little spread to explain price differences. Reviews look more like a baseline trust requirement than a strong price lever in this snapshot.""")

# Host type
md("""### 4.7 Host Type""")

co("""data["host_type"] = np.where(data["calculated_host_listings_count"] >= 5,
                              "Professional host", "Small host")

host_summary = (data.groupby("host_type")
                    .agg(listings=("id", "count"),
                         median_price=("price", "median"),
                         mean_price=("price", "mean"),
                         median_availability=("availability_365", "median"),
                         median_minimum_nights=("minimum_nights", "median")))
host_summary""")

co("""sns.boxplot(data=data, x="host_type", y="price")
plt.title("Price Distribution by Host Type")
plt.xlabel("Host Type")
plt.ylabel("Nightly Price ($)")
plt.show()""")

md("""Hosts with 5+ listings ("Professional host") tend to have higher availability and slightly different minimum-night patterns than smaller hosts. The price distributions overlap heavily, so this is more a market-segment signal than a strong price predictor on its own.""")

# Heatmap
md("""### 4.8 Numeric Correlations""")

co("""numeric_for_corr = ["price", "accommodates", "bathrooms", "bedrooms", "beds",
                    "minimum_nights", "availability_365", "number_of_reviews",
                    "review_scores_rating", "reviews_per_month",
                    "host_response_rate", "host_acceptance_rate",
                    "calculated_host_listings_count", "host_years", "amenities_count"]

corr = data[numeric_for_corr].corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()""")

md("""Capacity-related variables (`accommodates`, `bedrooms`, `beds`, `bathrooms`) are highly correlated with each other and are the strongest linear correlates of price. Review variables form a tight cluster with weak correlations to price. This collinearity is one reason a regularized linear model (Ridge / Lasso) can be more stable than plain linear regression on this feature set.""")

# ---------- 5. Statistics ----------
md("""## 5. Statistical Test

The boxplots make borough differences look real, but we should back that up with an actual test. Five independent borough groups with a continuous outcome is a textbook case for one-way ANOVA.

- **H0:** mean nightly price is the same across all five boroughs.
- **H1:** at least one borough has a different mean price.
- **α = 0.05.**""")

co("""borough_groups = [g["price"].values
                  for _, g in data.groupby("neighbourhood_group_cleansed")]

f_stat, p_val = stats.f_oneway(*borough_groups)
print(f"F-statistic: {f_stat:.2f}")
print(f"p-value:     {p_val:.3g}")

if p_val < 0.05:
    print("Reject H0: borough mean prices are not all equal.")
else:
    print("Fail to reject H0.")""")

md("""The p-value is essentially zero, so we reject H0: borough mean prices are not all equal.

One important caveat: with ~21,000 listings, even tiny mean differences will produce a vanishing p-value. So statistical significance here is almost guaranteed regardless of effect size. The practical magnitudes from the EDA (Manhattan median ≈ $222 vs Bronx ≈ $95) are what actually matter. The test confirms the difference is real; it does not tell us whether it is large.""")

# ---------- 6. ML Setup ----------
md("""## 6. Machine Learning Setup

The target is `price`. We use a mix of location, room type, property type, capacity, host, availability, review, and engineered features. Models stay within course scope:

1. **Median baseline** — predicts the training median for every listing. Sets the floor.
2. **Linear regression** — simplest model in the regression family.
3. **Ridge regression** — L2 regularized linear model; better behaved when features are correlated.
4. **Lasso regression** — L1 regularized; can shrink coefficients to zero, useful with many one-hot dummies.
5. **Decision tree regressor** — non-linear, handles interactions but can overfit.
6. **Random forest regressor** — bagged trees; usually more stable than a single tree.

Reported metrics:
- **MAE** — average absolute prediction error in dollars (easiest to interpret).
- **RMSE** — penalizes larger errors more heavily.
- **R²** — share of price variance the model explains.""")

co("""features = ["neighbourhood_group_cleansed", "neighbourhood_cleansed",
            "room_type", "property_type",
            "accommodates", "bathrooms", "bedrooms", "beds",
            "minimum_nights", "maximum_nights",
            "availability_30", "availability_60", "availability_90", "availability_365",
            "number_of_reviews", "number_of_reviews_ltm",
            "review_scores_rating", "review_scores_cleanliness",
            "review_scores_location", "review_scores_value",
            "reviews_per_month",
            "host_response_rate", "host_acceptance_rate",
            "host_is_superhost", "host_identity_verified", "instant_bookable",
            "calculated_host_listings_count", "host_years",
            "has_license", "amenities_count"]

model_data = data[features + ["price"]].copy()
X = model_data[features]
y = model_data["price"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical:", categorical_features)
print("Numeric:    ", len(numeric_features), "columns")""")

co("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])""")

co("""def fit_and_score(name, model):
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {
        "Model": name,
        "MAE":  mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2":   r2_score(y_test, pred),
        "pipeline": pipe,
    }

RF_PARAMS = dict(n_estimators=100, max_depth=18, random_state=42)

results = [
    fit_and_score("Median baseline",   DummyRegressor(strategy="median")),
    fit_and_score("Linear regression", LinearRegression()),
    fit_and_score("Ridge",             Ridge(alpha=1.0)),
    fit_and_score("Lasso",             Lasso(alpha=0.5, max_iter=10000)),
    fit_and_score("Decision tree",     DecisionTreeRegressor(max_depth=12, random_state=42)),
    fit_and_score("Random forest",     RandomForestRegressor(**RF_PARAMS)),
]

metrics = (pd.DataFrame([{k: r[k] for k in ("Model", "MAE", "RMSE", "R2")}
                         for r in results])
             .sort_values("MAE")
             .reset_index(drop=True))
metrics""")

co("""sns.barplot(data=metrics, x="MAE", y="Model")
plt.title("Held-out Test MAE by Model")
plt.xlabel("MAE: Average Dollar Error")
plt.ylabel("")
plt.show()""")

md("""All four real models beat the median baseline by a clear margin, so the features carry real signal. Random forest gives the lowest held-out MAE; the linear models (Linear / Ridge / Lasso) are close to each other and clearly behind the trees. A single decision tree is in between. This is consistent with price depending on combinations of features (neighborhood + room type + capacity) rather than a clean linear sum.""")

# ---------- 7. CV ----------
md("""## 7. Cross-Validation

A single train / test split can be lucky or unlucky. K-fold cross-validation gives a more stable estimate by averaging over multiple folds. We run 5-fold CV on the **training set only** so that the held-out test set above stays untouched. The Random Forest configuration is the same as in section 6.""")

co("""cv_models = [
    ("Linear regression", LinearRegression()),
    ("Ridge",             Ridge(alpha=1.0)),
    ("Lasso",             Lasso(alpha=0.5, max_iter=10000)),
    ("Decision tree",     DecisionTreeRegressor(max_depth=12, random_state=42)),
    ("Random forest",     RandomForestRegressor(**RF_PARAMS)),
]

cv_rows = []
for name, model in cv_models:
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    mae_scores = -cross_val_score(pipe, X_train, y_train, cv=5,
                                  scoring="neg_mean_absolute_error")
    r2_scores  =  cross_val_score(pipe, X_train, y_train, cv=5,
                                  scoring="r2")
    cv_rows.append({
        "Model":       name,
        "CV MAE mean": mae_scores.mean(),
        "CV MAE std":  mae_scores.std(),
        "CV R2 mean":  r2_scores.mean(),
        "CV R2 std":   r2_scores.std(),
    })

cv_metrics = pd.DataFrame(cv_rows).sort_values("CV MAE mean").reset_index(drop=True)
cv_metrics""")

md("""The cross-validated rankings line up with the single-split results from section 6: random forest first, then decision tree, then the linear family. The CV MAE standard deviations are small relative to the means, so the model ordering is stable across folds, not an artifact of one lucky split.""")

# ---------- 8. Feature Importance ----------
md("""## 8. Feature Importance

Random forest can report which features it relied on most. These importances reflect predictive usefulness inside this model; they are not a causal claim about what determines price.""")

co("""rf_pipeline = next(r["pipeline"] for r in results if r["Model"] == "Random forest")

onehot_names = (rf_pipeline.named_steps["pre"]
                            .named_transformers_["cat"]
                            .named_steps["onehot"]
                            .get_feature_names_out(categorical_features))
all_feature_names = numeric_features + list(onehot_names)

importances = rf_pipeline.named_steps["model"].feature_importances_

importance_df = (pd.DataFrame({"feature": all_feature_names,
                               "importance": importances})
                   .sort_values("importance", ascending=False)
                   .reset_index(drop=True))
importance_df.head(20)""")

co("""top_importance = importance_df.head(15)
sns.barplot(data=top_importance, x="importance", y="feature")
plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("")
plt.show()""")

md("""The top features line up with what the EDA suggested: capacity dominates (`accommodates`, `bathrooms`, with `bedrooms` and `beds` further down), followed by a borough-level location flag (`neighbourhood_group_cleansed_Manhattan` is third, but no specific `neighbourhood_cleansed_*` column makes the top 20 — the model compresses location into 'Manhattan vs the rest'), host-scale columns (`calculated_host_listings_count`, `host_years`), and the engineered `amenities_count`. The first room-type column to appear is `room_type_Private room` at rank 11, and the highest-ranked review column, `review_scores_location`, only shows up at rank 16. The other review-score columns (`rating`, `cleanliness`, `value`) drop out of the top 20 entirely. That ranking matches the weak review-price correlations from section 4.6, which is a useful internal consistency check between the EDA and the model.""")

# ---------- 9. Insights ----------
md("""## 9. Main Insights

1. **Borough matters but is too coarse.** Manhattan's median ($222) is about 2.3x Bronx's ($95), but the boxplots overlap heavily and within-borough neighborhood medians vary by tens of dollars. Borough is a useful first cut, not a benchmark.

2. **The entire-home premium is location-dependent.** Choosing "entire home over private room" is worth about $133 in Manhattan, $108 in Brooklyn, $78 in Queens, $64 in Bronx, and only $38 in Staten Island. A flat statement like "entire homes cost more" misses this large interaction.

3. **Capacity raises price unevenly.** Bigger listings cost more, but the marginal step from 6 to 10 guests is much smaller than the step from 2 to 4. The relationship is not linear, which is part of why tree models outperform linear models here.

4. **Review scores barely move price.** Correlations with price are small (rating ≈ 0.04, location ≈ 0.11). Most listings already cluster near the top of the rating scale, so high reviews look more like a basic trust requirement than a meaningful pricing lever.

5. **Random forest wins on this feature set.** It clearly beats the median baseline and beats the linear models on both held-out MAE and 5-fold CV MAE. The most likely reason is that price depends on combinations of features (neighborhood × room type × capacity), which trees handle natively.

6. **Feature importance lines up with the EDA story.** Capacity columns and the Manhattan borough flag dominate the random forest's top features; review scores rank far below, with only `review_scores_location` making it into the top 20. Specific neighborhood dummies do not appear in the top 20 at all — the model relies on the coarser Manhattan flag rather than the 200+ neighborhood columns. Same direction as the correlations and the borough / neighborhood EDA, even though the model was free to use any feature.""")

# ---------- 10. Recommendations ----------
md("""## 10. Host-Facing Recommendations

- Benchmark against the same **neighborhood + room type**, not just borough. Section 4.5 shows the entire-home / private-room gap shifts a lot between neighborhoods inside the same borough.
- Treat borough medians as a first sanity check, not a target.
- When evaluating whether to add a bedroom or bump capacity, compare the median price *step* at the relevant capacity level (section 4.3) rather than assuming a constant per-guest premium.
- Do not expect a higher rating to translate into a meaningfully higher price. Strong reviews matter for getting booked at all, but they are not a major price differentiator in this snapshot.
- If the goal is short-stay tourist demand, room type, capacity, and neighborhood matter more than host-side variables. Host availability and minimum-nights settings are mostly market-segment signals.""")

# ---------- 11. Limitations ----------
md("""## 11. Limitations

- **Listed price, not booked revenue.** The dataset is a snapshot of advertised nightly prices, not actual bookings or occupancy.
- **Large share of missing prices.** About 41% of raw listings have no price field and are dropped. The remaining sample skews toward listings that were active and priced at the snapshot date.
- **Outlier filter.** We trim the bottom and top 1% of prices, so conclusions describe typical listings, not luxury rentals or data-entry errors.
- **No time component.** Seasonality, holidays, big events, and competitor price changes are invisible in a single snapshot.
- **Association, not causation.** A high feature importance does not mean changing that feature will move price.
- **Many neighborhoods, sparse cells.** Section 4.5 only uses the largest neighborhoods. Smaller neighborhoods are too thin for stable medians.""")

# ---------- 12. Conclusion ----------
md("""## 12. Conclusion

Borough alone is the wrong unit for pricing decisions. Once we drop down to neighborhood + room type, the structure of the market becomes much clearer: the entire-home / private-room gap, the marginal value of capacity, and the irrelevance of review scores all emerge. Among course-scope models, random forest gives the best predictions on both a held-out test set and 5-fold CV, consistent with the non-linear, interaction-heavy structure that the EDA already suggested. The model's own feature importance ranking lines up with that EDA story, which is the most useful kind of validation here: the data, the test, and the model all point the same way.""")

nb["cells"] = cells
nbf.write(nb, "Airbnb_NYC_Price_Analysis.ipynb")
print("Wrote", len(cells), "cells")
