import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from hybrid import hybrid_recommender


# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations based on User ID or Movie Name!")

# Load the necessary data
@st.cache_data
def load_data():
    # Load pre-trained models and data
    with open('./models/movies.pkl', 'rb') as f:
        movies_df = pickle.load(f)
    with open('./models/ratings.pkl', 'rb') as f:
        ratings_df = pickle.load(f)
    with open('./models/nmf_model.pkl', 'rb') as f:
        nmf_model = pickle.load(f)
    with open('./models/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('./models/user_item_matrix.pkl', 'rb') as f:
        user_item_matrix = pickle.load(f)
    
    return movies_df, ratings_df, nmf_model, tfidf_matrix, user_item_matrix

try:
    movies_df, ratings_df, nmf_model, tfidf_matrix, user_item_matrix = load_data()

    # Create tabs for different recommendation types
    tab1, tab2, tab3 = st.tabs(["User-based Recommendation", "Movie-based Recommendation","Hybrid Recommendation"])

    with tab1:
        st.header("User-based Recommendation")
        user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings_df['user_id'].max())
        
        if st.button("Get Recommendations for User"):
            # User-based recommendation using NMF
            if user_id in user_item_matrix.index:
                user_vector = user_item_matrix.loc[user_id].fillna(0).values.reshape(1, -1)
                W_user = nmf_model.transform(user_vector)
                pred_ratings = np.dot(W_user, nmf_model.components_).flatten()
                
                recommendations = pd.DataFrame({
                    'movie_id': user_item_matrix.columns,
                    'predicted_rating': pred_ratings
                })
                
                # Exclude movies already rated by the user
                rated_movies = user_item_matrix.loc[user_id].dropna().index
                recommendations = recommendations[~recommendations['movie_id'].isin(rated_movies)]
                
                # Get top recommendations
                top_recommendations = recommendations.merge(
                    movies_df[['movie_id', 'title']], on='movie_id'
                ).nlargest(5, 'predicted_rating')
                
                st.write("Recommended movies based on your ratings:")
                st.dataframe(top_recommendations[['title', 'predicted_rating']])
            else:
                st.warning("No ratings found for this user.")

    with tab2:
        st.header("Movie-based Recommendation")
        movie_list = movies_df['title'].tolist()
        selected_movie = st.selectbox("Select a Movie", movie_list)
        
        if st.button("Get Similar Movies"):
            # Content-based recommendation using TF-IDF
            input_movie = movies_df[movies_df['title'] == selected_movie]
            if not input_movie.empty:
                input_idx = input_movie.index[0]
                cosine_sim = cosine_similarity(
                    tfidf_matrix[input_idx:input_idx+1],
                    tfidf_matrix
                ).flatten()
                
                sim_scores = pd.DataFrame({
                    'movie_id': movies_df['movie_id'],
                    'title': movies_df['title'],
                    'similarity': cosine_sim
                })
                
                # Get top similar movies excluding the input movie
                recommendations = sim_scores[sim_scores['movie_id'] != input_movie.iloc[0]['movie_id']]
                recommendations = recommendations.nlargest(5, 'similarity')
                
                st.write("Similar movies you might like:")
                st.dataframe(recommendations[['title', 'similarity']])
    with tab3:
        st.header("Hybrid Recommendation")

        user_id_input = st.number_input(
            "Enter User ID ", 
            min_value=1, 
            max_value=ratings_df['user_id'].max(), 
            step=1, 
            value=1
        )

        movie_title_input = st.selectbox(
            "Select a Movie (optional)", 
            options=[""] + sorted(movies_df['title'].unique().tolist()),
            index=0
        )

        if st.button("Get Hybrid Recommendations"):
            hybrid_recs = hybrid_recommender(
                user_id=user_id_input if user_id_input else None,
                movie_title=movie_title_input if movie_title_input else None or "",  # Handle empty string
                ratings_df=ratings_df,
                user_item_matrix=user_item_matrix,
                nmf_model=nmf_model,
                movies_df=movies_df,
                tfidf_matrix=tfidf_matrix,
                alpha=0.5,
                top_n=5
            )

            if not hybrid_recs.empty:
                st.write("Top Hybrid Recommendations:")
                st.dataframe(hybrid_recs)
            else:
                st.warning("No recommendations could be generated.")


except FileNotFoundError:
    st.error("Please ensure that 'movies.csv' and 'ratings.csv' files are present in the directory.")

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")