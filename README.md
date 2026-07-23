# 🎬 Movie Recommendation System

## 📌 Project Overview

The **Movie Recommendation System** is a Machine Learning project that recommends movies to users based on movie descriptions and popularity.

The project uses the **TMDB 5000 Movies Dataset** and **TMDB 5000 Credits Dataset**. It combines movie information with user ratings and popularity metrics to create a ranking system and uses **Natural Language Processing (NLP)** to recommend movies that are similar to a selected movie.

The recommendation system uses **TF-IDF Vectorization** and **Sigmoid Kernel Similarity** to identify movies with similar storylines and descriptions.

---

## 🎯 Project Objective

The main objectives of this project are:

* Analyze movie data from the TMDB dataset.
* Combine movie and credits information.
* Clean and preprocess the dataset.
* Rank movies based on ratings and popularity.
* Calculate weighted movie ratings.
* Normalize rating and popularity scores.
* Create a combined movie ranking score.
* Use NLP to analyze movie overviews.
* Generate movie recommendations based on content similarity.

---

## 🛠️ Technologies Used

* **Python** – Programming language
* **Pandas** – Data manipulation and preprocessing
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Plotly** – Interactive visualization
* **Scikit-learn** – Machine Learning and NLP
* **TF-IDF Vectorizer** – Text feature extraction
* **Sigmoid Kernel** – Similarity calculation

---

## 📂 Dataset

The project uses two datasets from the TMDB 5000 Movie Dataset:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

### Movies Dataset

The movie dataset contains information such as:

* Movie ID
* Movie title
* Original title
* Overview
* Genres
* Keywords
* Popularity
* Vote average
* Vote count
* Runtime
* Release date
* Budget
* Revenue

### Credits Dataset

The credits dataset contains information related to:

* Movie ID
* Cast
* Crew

The two datasets are merged using the movie `id`.

---

## 🔄 Project Workflow

```text
Load Movie Dataset
        ↓
Load Credits Dataset
        ↓
Rename movie_id to id
        ↓
Merge Both Datasets
        ↓
Data Cleaning
        ↓
Movie Rating Analysis
        ↓
Weighted Rating Calculation
        ↓
Popularity & Rating Normalization
        ↓
Combined Ranking Score
        ↓
Movie Overview Extraction
        ↓
TF-IDF Vectorization
        ↓
Similarity Calculation
        ↓
Movie Recommendation
```

---

## 🧹 Data Preprocessing

The movie and credits datasets are first loaded using Pandas.

The `movie_id` column in the credits dataset is renamed to `id` so that both datasets can be merged.

```python
credits_updated_df = credits.rename(
    columns={"movie_id": "id"}
)
```

The datasets are then merged:

```python
movies_df_merge = movies.merge(
    credits_updated_df,
    on="id"
)
```

Several unnecessary columns are removed, including:

* `homepage`
* `title_x`
* `title_y`
* `status`
* `production_countries`

This creates a cleaner dataset for further analysis.

---

# ⭐ Movie Ranking System

The project first creates a movie ranking system using:

* `vote_average`
* `vote_count`
* Overall average rating

A weighted rating is calculated to prevent movies with very few votes from appearing unfairly high in the rankings.

The weighted rating formula is:

```text
Weighted Rating =
(R × v + C × m) / (v + m)
```

Where:

* `R` = Average rating of the movie
* `v` = Number of votes for the movie
* `C` = Mean rating across all movies
* `m` = Minimum number of votes required

Movies with vote counts in the top 10% are selected for ranking.

This helps reduce the impact of movies with a very small number of ratings.

---

## 📊 Weighted Rating

The weighted rating is used to rank highly-rated movies.

The top movies can be identified using:

```python
df_sorted_ranking = filtered_df.sort_values(
    'weighted_avg',
    ascending=False
).head(20)
```

The project also visualizes the top-rated movies using Seaborn.

---

# 🔥 Popularity-Based Ranking

In addition to ratings, the project analyzes the `popularity` score of movies.

Movies are sorted based on popularity:

```python
popularity_df = filtered_df.sort_values(
    'popularity',
    ascending=False
)
```

The most popular movies are visualized using a bar chart.

---

# ⚖️ Combined Movie Score

The project combines:

* Weighted rating
* Movie popularity

Since these two values have different scales, **MinMaxScaler** is used to normalize them between 0 and 1.

```python
scaling = MinMaxScaler()

scaling_values = scaling.fit_transform(
    filtered_df[['weighted_avg', 'popularity']]
)
```

Two new features are created:

```text
weighted_avg_scaled
popularity_scaled
```

A combined score is then calculated:

```python
filtered_df['score_mix'] = (
    filtered_df['weighted_avg_scaled'] * 0.5 +
    filtered_df['popularity_scaled'] * 0.5
)
```

The final ranking gives equal importance to:

* 50% Movie Rating
* 50% Movie Popularity

This produces a balanced ranking that considers both quality and popularity.

---

# 🧠 Content-Based Recommendation System

The second part of the project builds a **Content-Based Movie Recommendation System**.

The system uses the movie's `overview` column to understand the content and storyline of each movie.

For example:

```text
Avatar
```

The system analyzes Avatar's movie description and finds other movies with similar descriptions.

---

## 📝 TF-IDF Vectorization

The movie overview text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(
    min_df=3,
    ngram_range=(1, 3),
    stop_words="english"
)
```

The vectorizer uses:

* Unigrams
* Bigrams
* Trigrams

This helps the model capture individual words as well as short phrases.

For example:

```text
Unigram:
space

Bigram:
space travel

Trigram:
space travel adventure
```

The resulting TF-IDF matrix represents each movie as a numerical vector.

---

## 🔗 Similarity Calculation

After converting movie overviews into numerical vectors, the project calculates similarity between movies.

The **Sigmoid Kernel** is used:

```python
from sklearn.metrics.pairwise import sigmoid_kernel

sig = sigmoid_kernel(
    tfv_matrix,
    tfv_matrix
)
```

The resulting similarity matrix contains similarity scores between every pair of movies.

A higher similarity score indicates that two movies have more similar content based on their descriptions.

---

# 🎬 Movie Recommendation Function

The `give_recommendations()` function takes:

* Movie title
* Similarity model

as inputs.

```python
def give_recommendations(movie_title, model):
    
    indices = pd.Series(
        data=data.index,
        index=data['original_title']
    )
    
    idx = indices[movie_title]
    
    model_scores = list(
        enumerate(model[idx])
    )
    
    model_scores_sorted = sorted(
        model_scores,
        key=lambda x: x[1],
        reverse=True
    )
    
    model_scores_10 = model_scores_sorted[1:11]
    
    movie_indices_10 = [
        i[0] for i in model_scores_10
    ]
    
    return dataframe[
        'original_title'
    ][movie_indices_10]
```

The function finds the selected movie, compares its similarity scores with all other movies, sorts the results, and returns the **top 10 most similar movies**.

---

## 💡 Example

If the user selects:

```text
Avatar
```

The system analyzes the overview of Avatar and recommends movies with similar content.

The recommendation process is:

```text
User Selects Movie
       ↓
Find Movie Index
       ↓
Get Similarity Scores
       ↓
Sort Similar Movies
       ↓
Select Top 10
       ↓
Display Recommendations
```

---

## 📊 Key Features

* 🎬 Movie data analysis
* ⭐ Weighted movie ratings
* 🔥 Popularity-based ranking
* ⚖️ Combined rating and popularity score
* 📊 Data visualization
* 📝 Natural Language Processing
* 🔤 TF-IDF feature extraction
* 🔗 Movie similarity calculation
* 🎯 Content-based recommendations
* 🎥 Top 10 similar movie suggestions

---

## 🚀 How to Run

### 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly
```

### 2. Download the Dataset

Place the following files in the project directory:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

### 3. Load the Data

Update the file paths according to your system:

```python
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")
```

### 4. Run the Notebook or Python Script

Execute the code step by step to:

* Process the dataset.
* Generate movie rankings.
* Calculate similarity scores.
* Generate recommendations.

### 5. Get Recommendations

Use:

```python
give_recommendations(
    'Avatar',
    sig
)
```

The function will return the top 10 movies similar to **Avatar**.

---

## ⚠️ Limitations

The current system has some limitations:

* Recommendations depend only on movie overview text.
* User preferences are not considered.
* The system does not use collaborative filtering.
* The similarity model is based on textual content only.
* New movies may not have enough information for accurate recommendations.
* The model does not learn from individual user feedback.

---

## 🚀 Future Improvements

The project can be improved by:

* Adding movie genres and keywords to the recommendation features.
* Including cast and crew information.
* Using cosine similarity instead of or alongside the sigmoid kernel.
* Building a hybrid recommendation system.
* Combining content-based and collaborative filtering.
* Adding user ratings and preferences.
* Creating a Streamlit web application.
* Adding movie posters and trailers.
* Deploying the system as a web application.
* Using deep learning or transformer-based embeddings for better semantic similarity.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Python programming
* Data preprocessing
* Data merging
* Exploratory Data Analysis
* Data visualization
* Movie rating analysis
* Weighted average calculations
* Feature scaling
* Natural Language Processing
* TF-IDF Vectorization
* Similarity-based recommendation
* Content-based filtering
* Machine Learning workflows

---

## 📌 Project Summary

This project demonstrates the development of a **Movie Recommendation System** using Python and Machine Learning techniques. The project first ranks movies using a combination of **weighted ratings and popularity**. It then uses **Natural Language Processing and TF-IDF Vectorization** to analyze movie descriptions and calculate content similarity using the **Sigmoid Kernel**.

Based on the similarity scores, the system recommends the **top 10 movies similar to a selected movie**.

The project provides practical experience in **Data Science, NLP, Feature Engineering, Machine Learning, and Recommendation Systems**.

---

## 📝 Resume Description

> Developed a Content-Based Movie Recommendation System using Python, Pandas, and Scikit-learn. Implemented weighted movie ratings and popularity-based ranking, applied TF-IDF Vectorization to movie overviews, and used Sigmoid Kernel similarity to recommend the top 10 movies based on content similarity.

---

## ⭐ Short Project Description

> **Movie Recommendation System** is a Python-based content recommendation project that analyzes TMDB movie data, ranks movies using weighted ratings and popularity, and recommends similar movies using **TF-IDF Vectorization and Sigmoid Kernel similarity** based on movie overviews.
