import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Configuration ---
# Load secrets from Streamlit's secrets management
try:
    SPOTIFY_CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
    SPOTIFY_CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
    REDIRECT_URI = "https://vibelist-ai.streamlit.app/" # Or http://localhost:8501 for local
except (FileNotFoundError, KeyError) as e:
    st.error(f"Error loading secrets: {e}. Please check your .streamlit/secrets.toml file.")
    st.stop()

SCOPE = "playlist-modify-private playlist-modify-public"
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Caching and Loading ---
# Cache the loading of our model and data so it only runs once per session.
@st.cache_resource
def load_model_and_data():
    try:
        model = SentenceTransformer(MODEL_NAME)
        song_df = pd.read_pickle("song_data.pkl")
        faiss_index = faiss.read_index("song_index.faiss")
        return model, song_df, faiss_index
    except FileNotFoundError:
        st.error("Model/data files not found. Please run the `create_song_database.py` script first.")
        return None, None, None

model, song_df, faiss_index = load_model_and_data()

# --- Helper Functions ---

def get_spotify_auth_manager():
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI, scope=SCOPE, cache_path=".spotify_cache"
    )

# THIS IS OUR NEW CUSTOM MODEL FUNCTION
def recommend_songs_from_vibe(vibe, num_recommendations=15):
    """
    Uses the custom sentence transformer model and FAISS index to find songs.
    """
    if model is None or faiss_index is None:
        return pd.DataFrame() # Return empty if model isn't loaded

    # 1. Convert the user's vibe into a vector embedding
    vibe_embedding = model.encode([vibe])
    
    # 2. Search the FAISS index for the closest song vectors
    # D = distances, I = indices of the songs in the original dataframe
    D, I = faiss_index.search(vibe_embedding, num_recommendations)
    
    # 3. Get the details of the recommended songs from our dataframe
    recommended_songs_df = song_df.iloc[I[0]]
    
    return recommended_songs_df


def create_spotify_playlist(vibe, recommended_songs_df, sp_client):
    """Creates a new private playlist on Spotify and adds tracks to it."""
    track_uris = recommended_songs_df['id'].tolist()
    if not track_uris:
        st.warning("No tracks to add to the playlist.")
        return None
        
    try:
        user_id = sp_client.current_user()['id']
        playlist_name = f"VibeList AI (Custom): {vibe[:40]}"
        
        playlist = sp_client.user_playlist_create(
            user=user_id, name=playlist_name, public=False,
            description=f"A playlist by your custom model for the vibe: '{vibe}'."
        )
        
        chunk_size = 100
        for i in range(0, len(track_uris), chunk_size):
            chunk = track_uris[i:i+chunk_size]
            sp_client.playlist_add_items(playlist['id'], chunk)
            
        return playlist['external_urls']['spotify']
    except Exception as e:
        st.error(f"Failed to create Spotify playlist: {e}")
        return None

# --- Streamlit UI (Largely Unchanged) ---

st.set_page_config(page_title="VibeList AI (Custom)", page_icon="🎵")
st.title("🎵 VibeList AI (Custom Model)")
st.subheader("Your personal AI DJ, powered by your own recommendation model.")

if model is None:
    st.stop() # Stop execution if the data files aren't loaded

# --- Authentication Flow ---
auth_manager = get_spotify_auth_manager()
query_params = st.query_params
if "code" in query_params:
    try:
        token_info = auth_manager.get_access_token(query_params["code"])
        st.session_state['token_info'] = token_info
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Error getting access token: {e}")

# Main logic
if 'token_info' not in st.session_state:
    auth_url = auth_manager.get_authorize_url()
    st.info("To get started, connect your Spotify account.")
    st.link_button("Login with Spotify", auth_url)
else:
    token_info = st.session_state['token_info']
    if auth_manager.is_token_expired(token_info):
        try:
            st.session_state['token_info'] = auth_manager.refresh_access_token(token_info['refresh_token'])
            st.toast("Spotify token refreshed!")
        except Exception:
            st.error("Your session has expired. Please log in again.")
            del st.session_state['token_info']
            st.rerun()

    try:
        sp_client = spotipy.Spotify(auth=st.session_state['token_info']['access_token'])
        user_info = sp_client.current_user()
        st.success(f"Logged in as **{user_info['display_name']}**")
    except Exception as e:
        st.error(f"Error connecting to Spotify. Details: {e}")
        del st.session_state['token_info']
        st.rerun()

    # --- Main App Interface ---
    st.markdown("---")
    st.header("1. Describe your Vibe")
    vibe_input = st.text_area(
        "What's the mood?",
        placeholder="e.g., 'high energy happy dance music', 'sad and slow instrumental song'",
        height=100
    )

    if st.button("✨ Generate Playlist ✨", type="primary", use_container_width=True):
        if not vibe_input:
            st.warning("Please describe a vibe first!")
        else:
            with st.spinner("🧠 Consulting your custom AI model..."):
                recommended_songs = recommend_songs_from_vibe(vibe_input)

            if not recommended_songs.empty:
                st.info(f"Model recommended {len(recommended_songs)} songs.")
                
                with st.expander("See Recommended Songs"):
                    st.dataframe(recommended_songs[['name', 'artist']])
                    
                with st.spinner("🎶 Creating your new playlist on Spotify..."):
                    playlist_url = create_spotify_playlist(vibe_input, recommended_songs, sp_client)
                
                if playlist_url:
                    st.balloons()
                    st.header("🎉 Your playlist is ready!")
                    st.markdown(f"**[Click here to open it in Spotify]({playlist_url})**")
            else:
                st.error("The model couldn't find any songs for that vibe. Try being more descriptive about the sound (e.g., 'high energy', 'sad', 'danceable').")

    st.markdown("---")
    if st.button("Logout of Spotify"):
        del st.session_state['token_info']
        st.success("You have been logged out.")
        st.rerun()

