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

def recommend_songs_from_vibe(vibe, num_recommendations=15, exclude_track_ids=None):
    """
    Uses the custom sentence transformer model and FAISS index to find songs,
    excluding any IDs that are already in the playlist.
    """
    if model is None or faiss_index is None:
        return pd.DataFrame()

    if exclude_track_ids is None:
        exclude_track_ids = []

    # Fetch more candidates than needed to account for filtering
    num_candidates = num_recommendations + len(exclude_track_ids) + 30 
    
    vibe_embedding = model.encode([vibe])
    D, I = faiss_index.search(vibe_embedding, num_candidates)
    
    # Filter out already present songs
    candidate_songs_df = song_df.iloc[I[0]]
    recommended_songs_df = candidate_songs_df[~candidate_songs_df['id'].isin(exclude_track_ids)]
    
    return recommended_songs_df.head(num_recommendations)

def create_spotify_playlist(vibe, recommended_songs_df, sp_client):
    """Creates a new playlist and returns the full playlist object."""
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
        return playlist
    except Exception as e:
        st.error(f"Failed to create Spotify playlist: {e}")
        return None

def add_tracks_to_playlist(playlist_id, new_songs_df, sp_client):
    """Adds a new batch of tracks to an existing playlist."""
    track_uris = new_songs_df['id'].tolist()
    if not track_uris:
        st.warning("No new tracks found to add.")
        return False
    try:
        sp_client.playlist_add_items(playlist_id, track_uris)
        return True
    except Exception as e:
        st.error(f"Failed to add tracks to playlist: {e}")
        return False

def remove_tracks_from_playlist(playlist_id, track_ids_to_remove, sp_client):
    """Removes a list of tracks from a specified playlist."""
    if not track_ids_to_remove:
        return False
    try:
        tracks = [{"uri": f"spotify:track:{track_id}"} for track_id in track_ids_to_remove]
        sp_client.playlist_remove_specific_occurrences_of_items(playlist_id, tracks)
        return True
    except Exception as e:
        st.error(f"Failed to remove tracks: {e}")
        return False

def get_playlist_tracks(playlist_id, sp_client):
    """Fetches all tracks currently in a given playlist."""
    try:
        results = sp_client.playlist_items(playlist_id)
        tracks = []
        while results:
            for item in results.get('items', []):
                track = item.get('track')
                if track and track.get('id'):
                    tracks.append({
                        'id': track['id'],
                        'name': track['name'],
                        'artist': track['artists'][0]['name']
                    })
            # Check for pagination
            if results['next']:
                results = sp.next(results)
            else:
                results = None
        return pd.DataFrame(tracks)
    except Exception as e:
        st.error(f"Could not fetch playlist tracks: {e}")
        return pd.DataFrame()

# --- Streamlit UI ---

st.set_page_config(page_title="VibeList AI (Custom)", page_icon="🎵")
st.title("🎵 VibeList AI (Custom Model)")
st.subheader("Your personal AI DJ, powered by your own recommendation model.")

if model is None:
    st.stop()

# --- Authentication Flow ---
auth_manager = get_spotify_auth_manager()

if hasattr(st, 'query_params'):
    query_params = st.query_params
else:
    query_params = st.experimental_get_query_params()

if "code" in query_params:
    try:
        if isinstance(query_params, dict):
            code = query_params.get("code")
        else:
            code = query_params["code"][0]
        token_info = auth_manager.get_access_token(code)
        st.session_state['token_info'] = token_info
        if hasattr(st, 'query_params'):
            st.query_params.clear()
        if hasattr(st, 'rerun'):
            st.rerun()
        else:
            st.experimental_rerun()
    except Exception as e:
        st.error(f"Error getting access token: {e}")

# --- Main App Logic ---
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
            keys_to_clear = ['token_info', 'playlist_id', 'playlist_url', 'original_vibe']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            if hasattr(st, 'rerun'):
                st.rerun()
            else:
                st.experimental_rerun()

    try:
        sp_client = spotipy.Spotify(auth=st.session_state['token_info']['access_token'])
        user_info = sp_client.current_user()
        st.success(f"Logged in as **{user_info['display_name']}**")
    except Exception as e:
        st.error(f"Error connecting to Spotify. Details: {e}")
        del st.session_state['token_info']
        if hasattr(st, 'rerun'):
            st.rerun()
        else:
            st.experimental_rerun()

    st.markdown("---")

    # --- UI STATE: CHOOSE BETWEEN CREATE OR REFINE ---
    if 'playlist_id' not in st.session_state:
        # STATE 1: CREATE A NEW PLAYLIST
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
                    with st.spinner("🎶 Creating your new playlist on Spotify..."):
                        playlist_details = create_spotify_playlist(vibe_input, recommended_songs, sp_client)
                    if playlist_details:
                        st.session_state['playlist_id'] = playlist_details['id']
                        st.session_state['playlist_url'] = playlist_details['external_urls']['spotify']
                        st.session_state['original_vibe'] = vibe_input
                        if hasattr(st, 'rerun'):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                else:
                    st.error("The model couldn't find any songs for that vibe. Try being more descriptive.")
    else:
        # STATE 2: REFINE AN EXISTING PLAYLIST
        st.header("🎉 Your Playlist")
        st.markdown(f"**[Open on Spotify]({st.session_state['playlist_url']})**")

        # Display current tracks and add remove functionality
        st.subheader("Current Tracks")
        with st.spinner("Loading current playlist..."):
            current_tracks_df = get_playlist_tracks(st.session_state['playlist_id'], sp_client)

        if not current_tracks_df.empty:
            st.session_state['current_track_ids'] = current_tracks_df['id'].tolist()
            for index, row in current_tracks_df.iterrows():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{row['name']}** by {row['artist']}")
                with col2:
                    if st.button("Remove", key=f"remove_{row['id']}", use_container_width=True):
                        with st.spinner("Removing song..."):
                            success = remove_tracks_from_playlist(st.session_state['playlist_id'], [row['id']], sp_client)
                        if success:
                            st.toast(f"Removed '{row['name']}'!")
                            if hasattr(st, 'rerun'):
                                st.rerun()
                            else:
                                st.experimental_rerun()
                        else:
                            st.error("Failed to remove the song.")
        else:
            st.warning("Your playlist is currently empty.")
            st.session_state['current_track_ids'] = []

        st.markdown("---")
        st.header("2. Refine Your Playlist")
        st.info(f"Original vibe: **{st.session_state['original_vibe']}**")

        refinement_input = st.text_input(
            "What should we add?",
            placeholder="e.g., more rock music, something upbeat"
        )
        num_songs_to_add = st.slider("How many songs to add?", min_value=1, max_value=25, value=10)

        if st.button("➕ Add More Songs", use_container_width=True):
            if not refinement_input:
                st.warning("Please enter a refinement instruction.")
            else:
                new_vibe = f"{st.session_state['original_vibe']}, {refinement_input}"
                with st.spinner(f"Finding new songs for refined vibe..."):
                    new_songs = recommend_songs_from_vibe(
                        new_vibe,
                        num_recommendations=num_songs_to_add,
                        exclude_track_ids=st.session_state.get('current_track_ids', [])
                    )
                
                if not new_songs.empty:
                    with st.expander("Songs to be added", expanded=True):
                        display_df = new_songs[['name', 'artist_name']].copy()
                        display_df.rename(columns={'artist_name': 'Artist'}, inplace=True)
                        st.dataframe(display_df)
                    
                    with st.spinner("Adding new songs to your playlist..."):
                        success = add_tracks_to_playlist(st.session_state['playlist_id'], new_songs, sp_client)
                    
                    if success:
                        st.toast("✅ Songs added successfully!")
                        if hasattr(st, 'rerun'):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                else:
                    st.error("Couldn't find any new songs for that refinement.")
        
        if st.button("Start Over"):
            keys_to_clear = ['playlist_id', 'playlist_url', 'original_vibe', 'current_track_ids']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            if hasattr(st, 'rerun'):
                st.rerun()
            else:
                st.experimental_rerun()

    # --- LOGOUT BUTTON (always visible when logged in) ---
    st.markdown("---")
    if st.button("Logout of Spotify"):
        keys_to_clear = ['token_info', 'playlist_id', 'playlist_url', 'original_vibe', 'current_track_ids']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("You have been logged out.")
        if hasattr(st, 'rerun'):
            st.rerun()
        else:
            st.experimental_rerun()

