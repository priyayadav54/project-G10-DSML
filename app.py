import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Assuming 'hybrid.py' contains the hybrid_recommender function
# Make sure hybrid.py is in the same directory or accessible in your PYTHONPATH
from hybrid import hybrid_recommender

# --- Custom CSS for Theming ---
def apply_theme():
    # Load current theme from session state, default to 'dark'
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark' # Default theme

    if st.session_state.theme == 'dark':
        st.markdown(
            """
            <style>
            body {
                background-color: #0d1117; /* GitHub Dark Mode Background */
                color: #c9d1d9; /* Light text */
            }
            .stApp {
                background-color: #0d1117;
                color: #c9d1d9;
            }
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size: 1.1rem;
                font-weight: bold;
            }
            .stTabs [data-baseweb="tab-list"] button {
                border-radius: 5px 5px 0 0;
                padding: 10px 15px;
                margin-right: 5px;
                background-color: #161b22; /* Darker tab background */
                color: #8b949e; /* Tab text color */
            }
            .stTabs [data-baseweb="tab-list"] button:hover {
                background-color: #21262d; /* Hover darker */
                color: #c9d1d9;
            }
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                background-color: #0d1117; /* Active tab same as app background */
                color: #e6edf3; /* Brighter text for active tab */
                border-bottom: 2px solid #238636; /* Accent color for active tab */
            }
            .stTextInput>div>div>input {
                background-color: #161b22;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 5px;
            }
            .stNumberInput>div>div>input {
                background-color: #161b22;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 5px;
            }
            .stSelectbox>div>div {
                 background-color: #161b22;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 5px;
            }
            .stSelectbox>div>div>div[data-baseweb="select"]>div[aria-selected="true"] {
                 background-color: #21262d !important; /* Selected item background */
            }
             .stSelectbox>div>div>div[data-baseweb="select"] div[role="option"]:hover {
                background-color: #21262d !important; /* Option hover background */
            }
            .stRadio div[role="radiogroup"] {
                background-color: #161b22; /* Background for the radio group */
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #30363d;
            }
            .stRadio div[role="radio"] {
                color: #c9d1d9;
            }
            .stButton>button {
                background-color: #238636; /* GitHub green button */
                color: white;
                border-radius: 5px;
                border: none;
                padding: 10px 20px;
                font-weight: bold;
                transition: background-color 0.2s;
            }
            .stButton>button:hover {
                background-color: #2ea043;
            }
            .stDataFrame {
                color: #c9d1d9 !important;
            }
            table th {
                background-color: #21262d !important;
                color: #8b949e !important;
            }
            table td {
                background-color: #161b22 !important;
                color: #c9d1d9 !important;
            }
            /* Styling for markdown headers */
            h1, h2, h3, h4, h5, h6 {
                color: #e6edf3;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else: # Light theme
        st.markdown(
            """
            <style>
            body {
                background-color: #ffffff;
                color: #333333;
            }
            .stApp {
                background-color: #ffffff;
                color: #333333;
            }
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size: 1.1rem;
                font-weight: bold;
            }
            .stTabs [data-baseweb="tab-list"] button {
                border-radius: 5px 5px 0 0;
                padding: 10px 15px;
                margin-right: 5px;
                background-color: #f0f2f6;
                color: #555555;
            }
            .stTabs [data-baseweb="tab-list"] button:hover {
                background-color: #e0e2e6;
                color: #333333;
            }
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                background-color: #ffffff;
                color: #111111;
                border-bottom: 2px solid #007bff; /* Blue accent */
            }
            .stTextInput>div>div>input {
                background-color: #f8f9fa;
                color: #333333;
                border: 1px solid #ced4da;
                border-radius: 5px;
            }
            .stNumberInput>div>div>input {
                background-color: #f8f9fa;
                color: #333333;
                border: 1px solid #ced4da;
                border-radius: 5px;
            }
            .stSelectbox>div>div {
                 background-color: #f8f9fa;
                color: #333333;
                border: 1px solid #ced4da;
                border-radius: 5px;
            }
             .stSelectbox>div>div>div[data-baseweb="select"]>div[aria-selected="true"] {
                 background-color: #e9ecef !important; /* Selected item background */
            }
            .stSelectbox>div>div>div[data-baseweb="select"] div[role="option"]:hover {
                background-color: #e9ecef !important; /* Option hover background */
            }
            .stRadio div[role="radiogroup"] {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #ced4da;
            }
            .stRadio div[role="radio"] {
                color: #333333;
            }
            .stButton>button {
                background-color: #007bff; /* Bootstrap blue button */
                color: white;
                border-radius: 5px;
                border: none;
                padding: 10px 20px;
                font-weight: bold;
                transition: background-color 0.2s;
            }
            .stButton>button:hover {
                background-color: #0056b3;
            }
            .stDataFrame {
                color: #333333 !important;
            }
            table th {
                background-color: #e9ecef !important;
                color: #555555 !important;
            }
            table td {
                background-color: #f8f9fa !important;
                color: #333333 !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #222222;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# --- Apply the selected theme ---
apply_theme()


# Set page configuration (This can be done here, but custom CSS handles most of it)
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

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

# --- Theme Toggle Button ---
col_title, col_theme_toggle = st.columns([0.8, 0.2])

with col_title:
    st.title("🎬 Movie Recommendation System")
with col_theme_toggle:
    # Use a checkbox for theme toggle
    if st.session_state.theme == 'dark':
        if st.toggle("💡 Light Theme", value=False):
            st.session_state.theme = 'light'
            st.rerun()
    else:
        if st.toggle("💡 Light Theme", value=True): # Value=True if currently light, to show it's "on"
            st.session_state.theme = 'light' # Still light if already light and toggle clicked
        else: # Toggle turned off, switch to dark
            st.session_state.theme = 'dark'
            st.rerun()

st.write("Get personalized movie recommendations based on User ID or Movie Name!")


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
        user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings_df['user_id'].max(), key="user_id_ub")
        if st.button("Get Recommendations for User", key="get_recs_ub"):
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
                # Use a more attractive display for recommendations
                st.markdown("---")
                emojis = ["🎬", "🍿", "🌟", "✨", "🔥"]
                for i, (_, row) in enumerate(top_recommendations.iterrows()):
                    st.markdown(f"**{emojis[i % len(emojis)]} {row['title']}** (Predicted Rating: {row['predicted_rating']:.2f})")
                st.markdown("---")
            else:
                st.warning("No ratings found for this user.")

    # --- Movie-based ---
    with tab2:
        st.header("Movie-based Recommendation")
        movie_list = movies_df['title'].tolist()
        selected_movie = st.selectbox("Select a Movie", movie_list, key="selected_movie_mb")
        if st.button("Get Similar Movies", key="get_similar_mb"):
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
                # Use a more attractive display for recommendations
                st.markdown("---")
                emojis = ["🤩", "🚀", "💡", "💎", "💯"]
                for i, (_, row) in enumerate(recommendations.iterrows()):
                    st.markdown(f"**{emojis[i % len(emojis)]} {row['title']}** (Similarity: {row['similarity']:.2f})")
                st.markdown("---")


    # --- Hybrid ---
    with tab3:
        st.header("Hybrid Recommendation")
        user_id_input = st.number_input(
            "Enter User ID",
            min_value=1,
            max_value=ratings_df['user_id'].max(),
            step=1,
            value=1,
            key="user_id_hybrid"
        )
        movie_title_input = st.selectbox(
            "Select a Movie (optional)",
            options=[""] + sorted(movies_df['title'].unique().tolist()),
            index=0,
            key="movie_title_hybrid"
        )
        if st.button("Get Hybrid Recommendations", key="get_hybrid_recs"):
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
                # Use a more attractive display for recommendations
                st.markdown("---")
                emojis = ["🌟", "✨", "🔥", "💫", "👍"]
                for i, (_, row) in enumerate(hybrid_recs.iterrows()):
                    st.markdown(f"**{emojis[i % len(emojis)]} {row['title']}** (Score: {row['hybrid_score']:.2f})")
                st.markdown("---")
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
            # Sample all unrated movies once, then shuffle
            st.session_state.unrated_movies_for_feedback = movies_df[~movies_df['movie_id'].isin(rated_movies_for_user)].sample(frac=1, random_state=42).reset_index(drop=True)

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
            st.info("You've rated all available movies for this user ID! Try a different user ID or reset the app.")

        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.button("Previous 5 Movies", disabled=(st.session_state.feedback_page == 0), key="prev_movies_btn"):
                st.session_state.feedback_page -= 1
                st.rerun()
        with col_next:
            if st.button("Next 5 Movies", disabled=(end_idx >= total_unrated_movies), key="next_movies_btn"):
                st.session_state.feedback_page += 1
                st.rerun()

        if st.button("🎯 Submit All Ratings and Show Suggestions", key="submit_ratings_btn"):
            if feedback_ratings:
                new_feedback = pd.DataFrame([
                    {'user_id': user_id_feedback, 'movie_id': mid, 'rating': rating}
                    for mid, rating in feedback_ratings.items()
                ])
                try:
                    existing = pd.read_csv('./models/ratings.csv')
                    # Avoid adding duplicate ratings if user re-submits the same page without changing
                    new_feedback_filtered = new_feedback[~new_feedback.set_index(['user_id', 'movie_id']).index.isin(existing.set_index(['user_id', 'movie_id']).index)]
                    updated = pd.concat([existing, new_feedback_filtered], ignore_index=True)
                except FileNotFoundError:
                    updated = new_feedback
                updated.to_csv('./models/ratings.csv', index=False)
                st.success("✅ Your ratings have been saved!")

                # Invalidate the cached data so it reloads with new ratings for future recommendations
                load_data.clear()
                # Reload data to ensure recommendations are based on updated ratings_df
                _, ratings_df, _, tfidf_matrix, _ = load_data()


                liked_ids = new_feedback[new_feedback['rating'] >= 4]['movie_id'].tolist()
                if liked_ids:
                    # Ensure current_movies_df and current_tfidf_matrix are up-to-date
                    current_movies_df = movies_df # Use the global movies_df
                    current_tfidf_matrix = tfidf_matrix # Use the global tfidf_matrix

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
                        st.markdown("---")
                        emojis = ["🚀", "🔥", "🌟", "💡", "🎉"] # Different emojis for feedback suggestions
                        for i, (_, row) in enumerate(top_similar.iterrows()):
                            st.markdown(f"**{emojis[i % len(emojis)]} {row['title']}** (Similarity: {row['similarity']:.2f})")
                        st.markdown("---")
                        st.info("🍿 You can rate more movies and submit again to refine your suggestions!")
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
    st.warning("Make sure you have `movies.pkl`, `ratings.pkl`, `nmf_model.pkl`, `tfidf_matrix.pkl`, and `user_item_matrix.pkl` in the `./models` directory.")
    st.info("You might also need an initial `ratings.csv` if it's not generated by the app yet.")


