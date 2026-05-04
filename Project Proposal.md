# Analyze Factors Influencing Airbnb Listing Price in NYC
Wendy Fu & Kerry Zhang

---

## 1. Introduction

With the rapid growth of short-term rental platforms such as Airbnb, pricing strategies have become increasingly important for hosts aiming to maximize revenue while remaining competitive. Listing prices are influenced by multiple factors, including property characteristics, location, host behavior, and customer feedback. This project aims to analyze Airbnb listing data to identify the key factors that influence listing prices. Using a real-world dataset representing a snapshot of Airbnb listings, the project will explore how various features contribute to price differences and provide data-driven insights for effective pricing strategies.

## 2. Objectives

- Perform exploratory data analysis (EDA) to understand the distribution of listing prices across boroughs, room types, and neighborhoods.
- Identify which listing-level features (e.g., room type, number of bedrooms) and host-level features (e.g., number of listings) most strongly correlate with nightly price.
- Build and evaluate machine learning regression models to predict nightly listing price.
- Interpret model feature importances to produce actionable pricing recommendations for Airbnb hosts.

## 3. Data Collection

The dataset used in this project is obtained from publicly available detailed Airbnb listings data for New York City, sourced directly from **Inside Airbnb** (http://insideairbnb.com/get-the-data/), a publicly available platform that aggregates Airbnb listing data. We will use the latest NYC `listings.csv.gz` (dated November, 1, 2025), which contains approximately 40,000+ listings. The dataset encompasses variables such as listing characteristics, location information, host attributes, availability, and guest review metrics. Proper data preprocessing techniques such as handling missing values, feature engineering, etc. will be applied to clean and prepare the data for analysis.

## 4. Deliverables

A Jupyter Notebook (`.ipynb`) containing:

- The code implemented for data preprocessing, analysis, and modeling techniques.
- EDA visualizations with written interpretation
- Trained and evaluated ML models with with side-by-side evaluation metrics
- Feature importance chart with interpretation
- Summary of insights and host-facing recommendations

## 5. Conclusion

This project aims to provide valuable insights into the factors driving Airbnb listing prices in New York City. By identifying the key features that influence pricing, the analysis will enable hosts to make data-driven decisions to optimize their pricing strategies and improve listing performance.

