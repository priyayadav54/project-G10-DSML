import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
def recommend_nmf(user_id, user_item_matrix, nmf_model, movies_df, top_n=5):
    if user_id not in user_item_matrix.index:
        return f"User ID {user_id} not found."

    # Fill missing with 0s just like during training
    user_vector = user_item_matrix.loc[user_id].fillna(0).values.reshape(1, -1)

    # Project this user into latent space
    W_user = nmf_model.transform(user_vector)
    pred_ratings = np.dot(W_user, nmf_model.components_).flatten()

    # Create DataFrame of predictions
    recommendations = pd.DataFrame({
        'movie_id': user_item_matrix.columns,
        'nmf_score': pred_ratings
    })

    # Exclude movies already rated
    rated_movies = user_item_matrix.loc[user_id].dropna().index
    recommendations = recommendations[~recommendations['movie_id'].isin(rated_movies)]

    # Normalize scores
    recommendations['nmf_score'] = MinMaxScaler().fit_transform(recommendations[['nmf_score']])

    # Join with movie titles
    final_recommendations = recommendations.merge(
        movies_df[['movie_id', 'title']], on='movie_id'
    ).sort_values(by='nmf_score', ascending=False)

    return final_recommendations[['movie_id', 'title', 'nmf_score']].head(top_n)


def recommend_content_based(movie_title, movies_df, tfidf_matrix, top_n=5):
    
    # Find the input movie
    input_movie = movies_df[movies_df['title'].str.contains(movie_title, case=False)]
    if input_movie.empty:
        return pd.DataFrame(columns=['movie_id', 'title', 'similarity'])
    
    # Calculate similarities
    input_idx = input_movie.index[0]
    cosine_sim = cosine_similarity(
        tfidf_matrix[input_idx:input_idx+1], 
        tfidf_matrix
    ).flatten()
    
    # Get top-N most similar (excluding input movie)
    movies_df['similarity'] = cosine_sim
    recommendations = movies_df[movies_df['movie_id'] != input_movie.iloc[0]['movie_id']]
    
    return recommendations[['movie_id', 'title', 'similarity']].sort_values('similarity', ascending=False).head(top_n)


def hybrid_recommender(user_id=None,
                       movie_title=None,
                       ratings_df=None,
                       user_item_matrix=None,
                       nmf_model=None,
                       movies_df=None,
                       tfidf_matrix=None,
                       alpha=0.5,
                       top_n=5):
    cb_recs = pd.DataFrame()
    cf_recs = pd.DataFrame()

    # Content-Based Recommendations
    if movie_title:
        cb_recs = recommend_content_based(
            movie_title=movie_title,
            movies_df=movies_df,
            tfidf_matrix=tfidf_matrix,
            top_n=top_n
        )
        cb_recs = cb_recs.rename(columns={'similarity': 'cb_score'})

    # Collaborative Recommendations
    if user_id is not None:
        cf_recs = recommend_nmf(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            nmf_model=nmf_model,
            movies_df=movies_df,
            top_n=top_n * 2  # Use more for blending
        )
        cf_recs = cf_recs.rename(columns={'nmf_score': 'cf_score'})

    # Combine
    if not cb_recs.empty and not cf_recs.empty:
        combined = pd.merge(cb_recs, cf_recs, on=['movie_id', 'title'], how='outer').fillna(0)
        combined['score'] = alpha * combined['cf_score'] + (1 - alpha) * combined['cb_score']
        return combined.sort_values('score', ascending=False).head(top_n)[['movie_id', 'title', 'score']]

    elif not cb_recs.empty:
        cb_recs['score'] = cb_recs['cb_score']
        return cb_recs[['movie_id', 'title', 'score']]

    elif not cf_recs.empty:
        cf_recs['score'] = cf_recs['cf_score']
        return cf_recs[['movie_id', 'title', 'score']]

    else:
        return pd.DataFrame(columns=['movie_id', 'title', 'score'])
