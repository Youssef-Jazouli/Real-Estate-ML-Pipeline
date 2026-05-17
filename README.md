# 🏠 Morocco Real Estate ML

A machine learning pipeline for predicting real estate prices in Morocco — covering both **price regression** (exact value in MAD) and **price category classification** (Économique / Moyen / Luxe).

---

## 📌 Project Overview

This project scrapes, processes, and models real estate listings from Moroccan property markets. It trains two models on engineered features:

| Task | Target | Algorithm |
|---|---|---|
| Regression | `log_prix` (log of price in MAD) | Gradient Boosting / XGBoost |
| Classification | `price_category` | Random Forest / Gradient Boosting |

---

## 📁 Project Structure

```
morocco-real-estate-ml/
├── data/
│   └── raw/
│       └── obt_data.csv          # Raw scraped listings
├── src/
│   ├── extraction/
│   │   └── db_connector.py       # PostgreSQL OBT extraction
│   ├── features/
│   │   └── engineering.py        # Feature engineering & train/test split
│   ├── preprocessing/
│   │   └── preprocessor.py       # Scaling, encoding, pipelines
│   └── models/
│       ├── train_models.py       # Model training & evaluation
│       └── *.pkl                 # Saved preprocessors
├── main.py                       # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The raw dataset (`obt_data.csv`) contains **8,193 listings** scraped from Moroccan real estate platforms, with the following columns:

| Column | Description |
|---|---|
| `id` | Listing identifier |
| `titre` | Listing title |
| `prix_mad` | Price in Moroccan Dirham (MAD) |
| `ville` | City (e.g. Casablanca, Marrakech, Rabat…) |
| `quartier` | Neighbourhood |
| `surface_m2` | Surface area in m² |
| `nb_chambres` | Number of bedrooms |
| `nb_salles_de_bain` | Number of bathrooms |
| `prix_par_m2` | Price per m² |
| `scraped_at` | Scraping timestamp |

**Price range:** 108,000 MAD → 100,000,000 MAD  
**Surface range:** 23 m² → 1,001 m²  
**Cities covered:** Casablanca, Rabat, Marrakech, Tanger, Kénitra, El Jadida, Mohammedia, Tétouan, Asilah, Saidia, and more.

---

## ⚙️ Feature Engineering

After the train/test split, the following features are derived:

| Feature | Description |
|---|---|
| `log_prix` | Natural log of `prix_mad` (regression target) |
| `price_category` | `Économique` / `Moyen` / `Luxe` (classification target) |
| `surface_par_chambre` | Surface area per bedroom |
| `mois_annonce` | Month the listing was published |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/morocco-real-estate-ml.git
cd morocco-real-estate-ml
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example env file and fill in your database credentials:

```bash
cp _env .env
```

Edit `.env`:

```
DB_HOST=your_host
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password
```

### 5. Run the pipeline

```bash
python main.py
```

The pipeline will:
1. **Extract** raw data from the PostgreSQL database (or load from `data/raw/obt_data.csv` if it already exists)
2. **Engineer features** and split into train / test sets
3. **Preprocess** data (scaling, encoding)
4. **Train** the regression model and evaluate it
5. **Train** the classification model and evaluate it

---

## 🧪 Models & Evaluation

### Regression — Predicting `log_prix`

Metrics reported: **RMSE**, **MAE**, **R²**

The log-price target reduces the effect of extreme outliers and brings the distribution closer to normal, improving model stability.

### Classification — Predicting `price_category`

Three classes:
- **Économique** — budget listings
- **Moyen** — mid-range listings
- **Luxe** — luxury listings

Metrics reported: **Accuracy**, **F1-score (weighted)**, **Classification Report**

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| Python 3.13 | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | Preprocessing, models, evaluation |
| SQLAlchemy + psycopg2 | PostgreSQL connection |
| python-dotenv | Environment variable management |
| joblib | Model serialization |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is for educational and research purposes. See `LICENSE` for details.
