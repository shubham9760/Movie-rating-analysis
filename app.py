import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
df_final = pd.read_pickle("dataframe.pkl")

# Convert 'release date' to datetime format
df_final["release date"] = pd.to_datetime(df_final["release date"])
df_final["Days"] = df_final["release date"].dt.day_name()

# Streamlit app configuration
st.set_page_config(page_title="Movie Ratings Analysis", page_icon="🎬", layout="wide")
st.title("Movie Ratings & Release Date Analysis")

# Add GitHub icon with a button link to the top right
st.markdown(
    """
    <style>
    .github-link {
        position: absolute;
        right: 20px;
        top: 10px;
        text-decoration: none;
    }
    .github-link:hover {
        opacity: 0.8;
    }
    .github-link img {
        width: 32px;
    }
    .github-button {
        display: inline-block;
        padding: 10px 20px;
        font-size: 16px;
        color: #fff;
        background-color: #24292e;
        border-radius: 5px;
        text-decoration: none;
        margin-top: 10px;
        margin-right: 10px;
    }
    .github-button:hover {
        background-color: #444;
    }
    </style>

    <a href="https://github.com/shubham9760/FMGC-Project" class="github-link" target="_blank">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Link">
    </a>

    <a href="https://github.com/shubham9760/FMGC-Project" class="github-button" target="_blank">
        View on GitHub
    </a>
    """,
    unsafe_allow_html=True
)

# Sidebar options
st.sidebar.header("Navigation")
st.sidebar.markdown("Select a view to analyze movie ratings and release dates.")
option = st.sidebar.selectbox(
    "Choose a view",
    [
        "Days vs Age",
        "Ratings Distribution",
        "Outlier Detection",
        "Ratings Histogram",
        "Release Date vs Rating",
        "Top 10 Movies by Rating",
        "Day of Week Distribution",
        "Ratings Statistics",
        "Most Active Users",
        "Genre Distribution",
        "Interactive Scatter Plot"
    ]
)

# Visualization functions
def plot_days_vs_age():
    st.subheader("Days vs Age")
    fig, ax = plt.subplots(figsize=(8,4))  # Smaller chart size
    sns.barplot(x="Days", y="age", data=df_final, ax=ax)
    ax.set_title("Days VS Age")
    ax.tick_params(axis='x', rotation=45)  # Rotate x labels
    st.pyplot(fig)

def plot_ratings_distribution():
    st.subheader("Ratings Distribution")
    fig, ax = plt.subplots(figsize=(8,4))  # Smaller chart size
    sns.boxplot(df_final["rating"], ax=ax)
    ax.set_ylabel("Ratings")
    ax.tick_params(axis='x', rotation=45)  # Rotate x labels
    st.pyplot(fig)

def plot_outlier_detection():
    st.subheader("Outlier Detection")
    df_final["rating"] = pd.to_numeric(df_final["rating"], errors='coerce')
    z_scores = stats.zscore(df_final["rating"].dropna())
    z_threshold = 3
    outlier_mask = (z_scores > z_threshold) | (z_scores < -z_threshold)
    outliers_df = df_final.loc[outlier_mask]

    st.write("Rows corresponding to outliers:")
    st.write(outliers_df)

    fig, ax = plt.subplots(figsize=(8,4))  # Smaller chart size
    sns.boxplot(x=df_final["rating"], showfliers=False, ax=ax)
    sns.stripplot(x=df_final["rating"][~outlier_mask], color="red", marker="o", size=8, label="Outliers", ax=ax)  
    ax.set_ylabel("Ratings")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)  # Rotate x labels
    st.pyplot(fig)

def plot_ratings_histogram():
    st.subheader("Ratings Histogram")
    fig, ax = plt.subplots(figsize=(8,4))  # Smaller chart size
    sns.histplot(data=df_final["rating"], bins=5, ax=ax)
    ax.set_ylabel("Ratings")
    ax.tick_params(axis='x', rotation=45)  # Rotate x labels
    st.pyplot(fig)

def plot_release_date_vs_rating():
    st.subheader("Release Date vs Rating")
    fig, ax = plt.subplots(figsize=(8,4))  # Smaller chart size
    sns.scatterplot(x="rating", y="release date", data=df_final, hue="rating", alpha=1, ax=ax)
    ax.tick_params(axis='x', rotation=45)  # Rotate x labels
    st.pyplot(fig)

def plot_top_10_movies_by_rating():
    st.subheader("Top 10 Movies by Rating")
    df_max_rating = df_final[df_final["rating"] == df_final["rating"].max()].sort_values(by="rating", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,6))  # Smaller chart size
    sns.barplot(x="movie title", y="rating", data=df_max_rating, ax=ax)
    ax.tick_params(axis='x', rotation=45)  # Rotate x labels
    st.pyplot(fig)

def plot_day_of_week_distribution():
    st.subheader("Day of the Week Distribution")
    fig, ax = plt.subplots(figsize=(8,4))  # Smaller chart size
    sns.histplot(data=df_final["Days"], bins=7, ax=ax)  # Changed bins to 7 for days of the week
    ax.tick_params(axis='x', rotation=45)  # Rotate x labels
    st.pyplot(fig)

def plot_ratings_statistics():
    st.subheader("Ratings Statistics")
    y = df_final["rating"].mean()
    x = df_final["rating"].median()
    l = df_final["rating"].std()
    st.write(f"The mean value is {y}")
    st.write(f"The median value is {x}")
    st.write(f"The standard deviation is {l}")

def plot_most_active_users():
    st.subheader("Most Active Users")
    x = df_final.groupby("occupation")["movie title"].count().sort_values(ascending=False).head(15).to_frame(name="Movie_count").reset_index()
    fig, ax = plt.subplots(figsize=(10,6))  # Smaller chart size
    sns.lineplot(x="occupation", y="Movie_count", data=x, ax=ax)
    ax.set_title("Most Active Users by Movie Rental")
    ax.tick_params(axis='x', rotation=45)  # Rotate x labels
    st.pyplot(fig)

def plot_genre_distribution():
    st.subheader("Genre Distribution")
    y = df_final.iloc[:, 11:24].sum().sort_values(ascending=False).reset_index(name="Count")
    fig, ax = plt.subplots(figsize=(10,6))  # Smaller chart size
    sns.barplot(x="index", y="Count", data=y, ax=ax)
    ax.set_xlabel("Movie Genre")
    ax.set_ylabel("Genre Count")
    ax.set_title("Genre by Movie Count")
    ax.tick_params(axis='x', rotation=45)  # Rotate x labels
    st.pyplot(fig)

def plot_interactive_scatter_plot():
    st.subheader("Interactive Scatter Plot")
    fig = px.scatter(df_final, x='occupation', y='rating', hover_data=['rating'])
    fig.update_layout(title='Interactive Scatter Plot', xaxis_title='Occupation', yaxis_title='Rating')
    st.plotly_chart(fig)

# Display the selected option
if option == "Days vs Age":
    plot_days_vs_age()
elif option == "Ratings Distribution":
    plot_ratings_distribution()
elif option == "Outlier Detection":
    plot_outlier_detection()
elif option == "Ratings Histogram":
    plot_ratings_histogram()
elif option == "Release Date vs Rating":
    plot_release_date_vs_rating()
elif option == "Top 10 Movies by Rating":
    plot_top_10_movies_by_rating()
elif option == "Day of the Week Distribution":
    plot_day_of_week_distribution()
elif option == "Ratings Statistics":
    plot_ratings_statistics()
elif option == "Most Active Users":
    plot_most_active_users()
elif option == "Genre Distribution":
    plot_genre_distribution()
elif option == "Interactive Scatter Plot":
    plot_interactive_scatter_plot()