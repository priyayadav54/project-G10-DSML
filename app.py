
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Assuming 'hybrid.py' contains the hybrid_recommender function
# Make sure hybrid.py is in the same directory or accessible in your PYTHONPATH
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

# Initialize session state for pagination if not already set
if 'feedback_page' not in st.session_state:
    st.session_state.feedback_page = 0
if 'unrated_movies_for_feedback' not in st.session_state:
    st.session_state.unrated_movies_for_feedback = pd.DataFrame()
if 'current_user_for_feedback' not in st.session_state:
    st.session_state.current_user_for_feedback = None

# Load the necessary data
@st.cache_data
def load_data():
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

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "User-based Recommendation",
        "Movie-based Recommendation",
        "Hybrid Recommendation",
        "⭐ Feedback & Smart Suggestions"
    ])

    # --- User-based ---
    with tab1:
        st.header("User-based Recommendation")
        user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings_df['user_id'].max())
        if st.button("Get Recommendations for User"):
            if user_id in user_item_matrix.index:
                user_vector = user_item_matrix.loc[user_id].fillna(0).values.reshape(1, -1)
                W_user = nmf_model.transform(user_vector)
                pred_ratings = np.dot(W_user, nmf_model.components_).flatten()
                recommendations = pd.DataFrame({
                    'movie_id': user_item_matrix.columns,
                    'predicted_rating': pred_ratings
                })
                rated_movies = user_item_matrix.loc[user_id].dropna().index
                recommendations = recommendations[~recommendations['movie_id'].isin(rated_movies)]
                top_recommendations = recommendations.merge(
                    movies_df[['movie_id', 'title']], on='movie_id'
                ).nlargest(5, 'predicted_rating')
                st.write("Recommended movies based on your ratings:")
                st.dataframe(top_recommendations[['title', 'predicted_rating']])
            else:
                st.warning("No ratings found for this user.")

    # --- Movie-based ---
    with tab2:
        st.header("Movie-based Recommendation")
        movie_list = movies_df['title'].tolist()
        selected_movie = st.selectbox("Select a Movie", movie_list)
        if st.button("Get Similar Movies"):
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
                recommendations = sim_scores[sim_scores['movie_id'] != input_movie.iloc[0]['movie_id']]
                recommendations = recommendations.nlargest(5, 'similarity')
                st.write("Similar movies you might like:")
                st.dataframe(recommendations[['title', 'similarity']])

    # --- Hybrid ---
    with tab3:
        st.header("Hybrid Recommendation")
        user_id_input = st.number_input(
            "Enter User ID",
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
                movie_title=movie_title_input if movie_title_input else None or "",
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

    # --- Feedback Tab ---
    with tab4:
        st.header("⭐ Rate Some Movies and Get Instant Suggestions!")
        user_id_feedback = st.number_input(
            "Your User ID for feedback",
            min_value=1,
            max_value=ratings_df['user_id'].max(),
            step=1,
            key="feedback_user_id_main"
        )

        # Reset pagination if user ID changes
        if st.session_state.current_user_for_feedback != user_id_feedback:
            st.session_state.feedback_page = 0
            st.session_state.unrated_movies_for_feedback = pd.DataFrame()
            st.session_state.current_user_for_feedback = user_id_feedback

        movies_per_page = 5

        st.markdown("### 🎥 Please rate these movies by clicking stars:")

        # Get unrated movies for the current user
        if st.session_state.unrated_movies_for_feedback.empty:
            rated_movies_for_user = ratings_df[ratings_df['user_id'] == user_id_feedback]['movie_id'].tolist()
            st.session_state.unrated_movies_for_feedback = movies_df[~movies_df['movie_id'].isin(rated_movies_for_user)].sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle once

        total_unrated_movies = len(st.session_state.unrated_movies_for_feedback)
        start_idx = st.session_state.feedback_page * movies_per_page
        end_idx = min(start_idx + movies_per_page, total_unrated_movies)

        movies_to_rate = st.session_state.unrated_movies_for_feedback.iloc[start_idx:end_idx]

        feedback_ratings = {}
        if not movies_to_rate.empty:
            for _, row in movies_to_rate.iterrows():
                rating = st.radio(
                    f"⭐ {row['title']}",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: "⭐" * x,
                    horizontal=True,
                    index=2,  # Initial rating is 3 (index 2 for 1,2,3,4,5)
                    key=f"rate_{user_id_feedback}_{row['movie_id']}_{st.session_state.feedback_page}"
                )
                feedback_ratings[row['movie_id']] = rating
        else:
            st.info("You've rated all available movies! Try a different user ID or reset the app.")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous 5 Movies", disabled=(st.session_state.feedback_page == 0)):
                st.session_state.feedback_page -= 1
                st.rerun()
        with col2:
            if st.button("Next 5 Movies", disabled=(end_idx >= total_unrated_movies)):
                st.session_state.feedback_page += 1
                st.rerun()

        if st.button("🎯 Submit All Ratings and Show Suggestions"):
            if feedback_ratings:
                new_feedback = pd.DataFrame([
                    {'user_id': user_id_feedback, 'movie_id': mid, 'rating': rating}
                    for mid, rating in feedback_ratings.items()
                ])
                try:
                    existing = pd.read_csv('./models/ratings.csv')
                    updated = pd.concat([existing, new_feedback], ignore_index=True)
                except FileNotFoundError:
                    updated = new_feedback
                updated.to_csv('./models/ratings.csv', index=False)
                st.success("✅ Your ratings have been saved!")

                # Invalidate the cached data so it reloads with new ratings
                load_data.clear()

                liked_ids = new_feedback[new_feedback['rating'] >= 4]['movie_id'].tolist()
                if liked_ids:
                    # Ensure movies_df is reloaded with potentially new data if needed,
                    # though for recommendations, the loaded data should be fine.
                    # This is more for the persistent storage of ratings.
                    current_movies_df, _, _, current_tfidf_matrix, _ = load_data()

                    liked_idx = current_movies_df[current_movies_df['movie_id'].isin(liked_ids)].index.tolist()
                    if liked_idx: # Check if liked_idx is not empty
                        cosine_scores = cosine_similarity(current_tfidf_matrix[liked_idx], current_tfidf_matrix).mean(axis=0)
                        similar_movies = pd.DataFrame({
                            'movie_id': current_movies_df['movie_id'],
                            'title': current_movies_df['title'],
                            'similarity': cosine_scores
                        })
                        similar_movies = similar_movies[~similar_movies['movie_id'].isin(liked_ids)]
                        top_similar = similar_movies.sort_values('similarity', ascending=False).head(5)

                        st.markdown("### 🎯 Based on your ratings, you may also like:")
                        emojis = ["🎬", "🍿", "🌟", "✨", "🔥"] # Unique emojis
                        for i, (_, row) in enumerate(top_similar.iterrows()):
                            st.write(f"{emojis[i % len(emojis)]} **{row['title']}** (Similarity: {row['similarity']:.2f})")
                        st.info("🍿 You can rerun to rate more and improve suggestions!")
                        # Advance to the next set of movies after submission
                        st.session_state.feedback_page += 1
                        st.session_state.unrated_movies_for_feedback = pd.DataFrame() # Clear to re-sample next set
                        st.rerun() # Rerun to show next set of movies and new suggestions
                    else:
                        st.warning("Could not find index for liked movies to generate recommendations.")
                else:
                    st.warning("⚠️ Please give at least one movie a 4⭐ or higher to get recommendations.")
            else:
                st.warning("Please rate at least one movie before submitting.")


except FileNotFoundError:
    st.error("❌ Please ensure all required model files are present in the './models' folder.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")