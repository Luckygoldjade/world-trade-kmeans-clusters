# World Trade is Good, World Tariffs are Bad

**Applied Data Science — Final Project (Winter 2025)**  
*By Tony Chan, Bachelor of Science in Computer Science, Spring 2025*

This project explores how international trade and tariffs impact global economies, using real-world data from sources like Kaggle, U.S. Census, WTO, and WITS. We apply clustering and decision tree models to identify trade behavior trends across decades.

---

## Project Structure

- Project Root: world-trade-kmeans-clusters/
  - world-trade-kmeans-clusters.py: Main Python analysis file
  - requirements.txt: Python dependencies
  - README.md: This file
  - data/
    - raw/: Original CSVs from Kaggle and APIs
    - cleaned/: Cleaned and clustered data
  - output/
    - *.png: Model visualizations (elbow, clusters, etc.)
  - docs/
    - *.png and *.pdf: Screenshots and report visuals

---

## Overview

### Problem Statement
How do tariffs affect international trade, and what trade patterns emerge when countries interact over time?

### Research Goals
- Quantify changes in exports/imports over 30+ years
- Visualize trade clusters using KMeans
- Predict trade characteristics using Decision Trees
- Examine U.S. trade relationships using real-time data

---

## Methods

### Data Cleaning
- Used pandas to normalize and clean 3 major datasets
- Removed nulls, corrected datatypes, and standardized formats
- Created new columns such as "Trade Balance", "Export Dependence"

### Modeling
- **KMeans Clustering** (Unsupervised)
  - Explored clusters of trade behavior
  - Used elbow method to select optimal `k`
- **Decision Tree Classifier** (Supervised)
  - Predicted trade labels like "Import-Driven" or "Balanced"

---

## Data Sources

- [Kaggle World Trade Dataset](https://www.kaggle.com/datasets/muhammadtalhaawan/world-export-and-import-dataset)
- [U.S. Census Bureau](https://www.census.gov/foreign-trade/current/index.html)
- [World Bank WITS](https://wits.worldbank.org)
- [World Trade Organization](https://stats.wto.org/dashboard/merchandise_en.html)

---

## Visual Results and Figures

### Python-Generated Output (from `/output/` folder)

Figure / Description / Link

| 19 | 3D KMeans clustering of Year vs Export vs Import | [19_kmeans_3d_clusters.png](output/19_kmeans_3d_clusters.png) |<br>
| 20 | 2D scatter: Import vs Export (Linear view 1) | [20_kmeans_import_vs_export_linear1.png](output/20_kmeans_import_vs_export_linear1.png) |<br>
| 21 | 2D scatter: Import vs Export (Linear view 2) | [21_kmeans_import_vs_export_linear2.png](output/21_kmeans_import_vs_export_linear2.png) |<br>
| 22 | 2D scatter: Import vs Export (Log scale) | [22_kmeans_import_vs_export_log.png](output/22_kmeans_import_vs_export_log.png) |<br>
| 28 | Confusion matrix heatmap for Export Dependence prediction | [28_confusion_matrix_export_dependence.png](output/28_confusion_matrix_export_dependence.png) |<br>
| 29 | Decision tree — Root node and first split | [29_decision_tree_root_split.png](output/29_decision_tree_root_split.png) |<br>
| 30 | Decision tree — Mid-level branches | [30_decision_tree_mid_levels.png](output/30_decision_tree_mid_levels.png) |<br>
| 31 | Decision tree — Full view (depth 5, ≤100 nodes) | [31_decision_tree_full_view.png](output/31_decision_tree_full_view.png) |<br>

### Conceptual & Supporting Screenshots (from `/docs/` folder)

Figure / Description / Link

| 01 | Trade container photo — representing global trade flow | [01_trade_container_photo.png](docs/01_trade_container_photo.png) |<br>
| 02 | Kaggle dataset preview (raw data screenshot) | [02_kaggle_dataset_preview.png](docs/02_kaggle_dataset_preview.png) |<br>
| 06 | Census.gov trade dashboard interface | [06_census_dashboard_main.png](docs/06_census_dashboard_main.png) |<br>
| 10 | WITS (World Bank) tariff dashboard screenshot | [10_wits_dashboard_main.png](docs/10_wits_dashboard_main.png) |<br>
| 17 | WTO dashboard and web scraping target | [17_wto_dashboard_scrape_target.png](docs/17_wto_dashboard_scrape_target.png) |<br>
| 32 | Conclusion visual — “Almost everything is imported” | [32_imported_product_everyday_life.png](docs/32_imported_product_everyday_life.png) |<br>

---

## How to Run

# 1. Clone this repository
git clone https://github.com/your-username/world-trade-kmeans-clusters.git

# 2. Navigate into the folder
cd world-trade-kmeans-clusters

# 3. Install required libraries
pip install -r requirements.txt

# 4. Run the main Python file
python world-trade-kmeans-clusters.py

## Key Findings
Countries form natural trade clusters based on export/import volumes<br>
Decision tree accuracy: ~77% predicting "Export Dependence"<br>
U.S. tariffs are decreasing over time — inversely correlated with trade volume growth<br>
Free trade policies benefit both large and small economies<br>

## License
This project is open for review by instructors, recruiters, and collaborators.<br>
Feel free to reference or fork for educational and non-commercial purposes.<br>
