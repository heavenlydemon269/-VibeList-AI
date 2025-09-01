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
    if model is None or faiss_index is None:
        return pd.DataFrame()
    if exclude_track_ids is None:
        exclude_track_ids = []
    num_candidates = num_recommendations + len(exclude_track_ids) + 50
    vibe_embedding = model.encode([vibe])
    D, I = faiss_index.search(vibe_embedding, num_candidates)
    candidate_songs_df = song_df.iloc[I[0]]
    recommended_songs_df = candidate_songs_df[~candidate_songs_df['id'].isin(exclude_track_ids)]
    return recommended_songs_df.head(num_recommendations)

def create_spotify_playlist(vibe, recommended_songs_df, sp_client):
    track_uris = recommended_songs_df['id'].tolist()
    if not track_uris:
        st.warning("No tracks to add to the playlist.")
        return None
    try:
        user_id = sp_client.current_user()['id']
        playlist_name = f"VibeList AI: {vibe[:40]}"
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

# Initialize session state variables
if 'playlist_history' not in st.session_state:
    st.session_state.playlist_history = []

# --- Authentication Flow ---
auth_manager = get_spotify_auth_manager()

if hasattr(st, 'query_params'):
    query_params = st.query_params
else:
    query_params = st.experimental_get_query_params()

if "code" in query_params:
    try:
        if isinstance(query_params, dict): code = query_params.get("code")
        else: code = query_params["code"][0]
        token_info = auth_manager.get_access_token(code)
        st.session_state['token_info'] = token_info
        if hasattr(st, 'query_params'): st.query_params.clear()
        if hasattr(st, 'rerun'): st.rerun()
        else: st.experimental_rerun()
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
            keys_to_clear = ['token_info', 'current_playlist', 'playlist_history']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            if hasattr(st, 'rerun'): st.rerun()
            else: st.experimental_rerun()

    try:
        sp_client = spotipy.Spotify(auth=st.session_state['token_info']['access_token'])
        user_info = sp_client.current_user()
        st.success(f"Logged in as **{user_info['display_name']}**")
    except Exception as e:
        st.error(f"Error connecting to Spotify. Details: {e}")
        del st.session_state['token_info']
        if hasattr(st, 'rerun'): st.rerun()
        else: st.experimental_rerun()

    # --- Sidebar for Playlist Management ---
    st.sidebar.title("Your Playlists")
    if st.sidebar.button("➕ Create New Playlist", use_container_width=True):
        if 'current_playlist' in st.session_state:
            del st.session_state['current_playlist']
        if hasattr(st, 'rerun'): st.rerun()
        else: st.experimental_rerun()
    
    st.sidebar.markdown("---")

    if st.session_state.playlist_history:
        st.sidebar.subheader("Manage Existing")
        for p in reversed(st.session_state.playlist_history): # Show newest first
            if st.sidebar.button(p['name'], key=f"manage_{p['id']}", use_container_width=True):
                st.session_state['current_playlist'] = p
                if hasattr(st, 'rerun'): st.rerun()
                else: st.experimental_rerun()
    else:
        st.sidebar.info("You haven't created any playlists in this session yet.")

    st.markdown("---")

    # --- UI STATE: CHOOSE BETWEEN CREATE OR REFINE ---
    if 'current_playlist' not in st.session_state:
        # STATE 1: CREATE A NEW PLAYLIST
        st.header("1. Describe your Vibe")
        vibe_input = st.text_area(
            "What's the mood?",
            placeholder="e.g., 'high energy happy dance music', 'sad and slow instrumental song'",
            height=100)
        if st.button("✨ Generate Playlist ✨", type="primary", use_container_width=True):
            if vibe_input:
                with st.spinner("🧠 Consulting your custom AI model..."):
                    recommended_songs = recommend_songs_from_vibe(vibe_input)
                if not recommended_songs.empty:
                    with st.spinner("🎶 Creating your new playlist on Spotify..."):
                        playlist_details = create_spotify_playlist(vibe_input, recommended_songs, sp_client)
                    if playlist_details:
                        new_playlist = {
                            'id': playlist_details['id'],
                            'url': playlist_details['external_urls']['spotify'],
                            'vibe': vibe_input,
                            'name': playlist_details['name']
                        }
                        st.session_state.playlist_history.append(new_playlist)
                        st.session_state['current_playlist'] = new_playlist
                        if hasattr(st, 'rerun'): st.rerun()
                        else: st.experimental_rerun()
                else:
                    st.error("The model couldn't find any songs for that vibe.")
            else:
                st.warning("Please describe a vibe first!")
    else:
        # STATE 2: REFINE AN EXISTING PLAYLIST
        playlist = st.session_state['current_playlist']
        st.header(f"Editing: {playlist['name']}")
        st.markdown(f"**[Open on Spotify]({playlist['url']})**")

        st.subheader("Current Tracks")
        with st.spinner("Loading playlist..."):
            current_tracks_df = get_playlist_tracks(playlist['id'], sp_client)

        if not current_tracks_df.empty:
            st.session_state['current_track_ids'] = current_tracks_df['id'].tolist()
            for index, row in current_tracks_df.iterrows():
                col1, col2 = st.columns([4, 1])
                col1.write(f"**{row['name']}** by {row['artist']}")
                if col2.button("Remove", key=f"remove_{row['id']}", use_container_width=True):
                    with st.spinner(f"Removing '{row['name']}'..."):
                        if remove_tracks_from_playlist(playlist['id'], [row['id']], sp_client):
                            st.toast(f"Removed '{row['name']}'!")
                            if hasattr(st, 'rerun'): st.rerun()
                            else: st.experimental_rerun()
        else:
            st.warning("Your playlist is empty.")
            st.session_state['current_track_ids'] = []

        st.markdown("---")
        st.header("Refine Your Playlist")
        st.info(f"Original vibe: **{playlist['vibe']}**")
        refinement_input = st.text_input("What should we add?", placeholder="e.g., more rock music")
        num_songs_to_add = st.slider("How many songs to add?", 1, 25, 10)

        if st.button("➕ Add More Songs", use_container_width=True):
            if refinement_input:
                new_vibe = f"{playlist['vibe']}, {refinement_input}"
                with st.spinner("Finding new songs..."):
                    new_songs = recommend_songs_from_vibe(
                        new_vibe, num_songs_to_add,
                        st.session_state.get('current_track_ids', []))
                if not new_songs.empty:
                    with st.spinner("Adding songs to your playlist..."):
                        if add_tracks_to_playlist(playlist['id'], new_songs, sp_client):
                            st.toast("✅ Songs added successfully!")
                            if hasattr(st, 'rerun'): st.rerun()
                            else: st.experimental_rerun()
                else:
                    st.error("Couldn't find any new songs for that refinement.")
            else:
                st.warning("Please enter a refinement instruction.")

    # --- LOGOUT BUTTON ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout of Spotify"):
        keys_to_clear = ['token_info', 'current_playlist', 'playlist_history']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.success("You have been logged out.")
        if hasattr(st, 'rerun'): st.rerun()
        else: st.experimental_rerun()

