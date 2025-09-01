import streamlit as st
import google.generativeai as genai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os
import datetime

# --- New Imports for Database ---
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# --- Configuration ---
# Load secrets from Streamlit's secrets management
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SPOTIFY_CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
    SPOTIFY_CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]

    # This is the Redirect URI for your deployed app.
    # For local testing, change this to "http://localhost:8501"
    # AND update it in your Spotify Developer Dashboard.
    REDIRECT_URI = "https://vibelist-ai.streamlit.app/" # Or "http://localhost:8501"
except FileNotFoundError:
    st.error("Secrets file not found. Make sure you have a .streamlit/secrets.toml file.")
    st.stop()
except KeyError as e:
    st.error(f"Missing secret: {e}. Please check your .streamlit/secrets.toml file.")
    st.stop()

# Spotify authentication scope
SCOPE = "playlist-modify-private playlist-modify-public"

# Initialize Google AI client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Google AI client. Error: {e}")
    st.stop()

# --- Database Setup (Using SQLite) ---
Base = declarative_base()
# This creates a file named 'vibelist.db' in your project directory
engine = create_engine('sqlite:///vibelist.db?check_same_thread=False') 
Session = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    spotify_user_id = Column(String, unique=True, nullable=False)
    display_name = Column(String)
    # WARNING: In a production app, this MUST be encrypted for security.
    refresh_token = Column(String)
    playlists = relationship("Playlist", back_populates="creator")

class Playlist(Base):
    __tablename__ = 'playlists'
    id = Column(Integer, primary_key=True)
    playlist_name = Column(String, nullable=False)
    spotify_playlist_url = Column(String, nullable=False)
    vibe_description = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.now)
    user_id = Column(Integer, ForeignKey('users.id'))
    creator = relationship("User", back_populates="playlists")

# Create the tables in the database if they don't already exist
Base.metadata.create_all(engine)

# --- Helper Functions ---

def get_spotify_auth_manager():
    """Creates and returns a SpotifyOAuth object for handling authentication."""
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=".spotify_cache"
    )

def generate_song_list(vibe):
    """Uses Google's Gemini model to generate a list of songs."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""
    You are a world-class DJ. A user described a vibe: "{vibe}".
    Generate a list of 15 songs for this vibe.
    Your response MUST be a valid JSON object with a single key "songs", which is an array of objects.
    Each object must have two keys: "artist" and "track".
    Example: {{ "songs": [ {{ "artist": "Artist Name", "track": "Track Name" }} ] }}
    Only output the raw JSON.
    """
    try:
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        song_data = json.loads(response.text)
        return song_data.get("songs", [])
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Error communicating with AI. It might be helpful to try a different vibe. Details: {e}")
        st.error(f"AI Raw Response: {getattr(response, 'text', 'No response text available.')}")
        return None

def find_spotify_tracks(songs, sp_client):
    """Searches for tracks on Spotify and returns a list of their URIs."""
    track_uris = []
    not_found_tracks = []
    progress_bar = st.progress(0, text="Searching for tracks on Spotify...")
    for i, song in enumerate(songs):
        query = f"track:{song.get('track', '')} artist:{song.get('artist', '')}"
        try:
            results = sp_client.search(q=query, type='track', limit=1)
            tracks = results['tracks']['items']
            if tracks:
                track_uris.append(tracks[0]['uri'])
            else:
                not_found_tracks.append(f"{song.get('track', '')} by {song.get('artist', '')}")
        except Exception as e:
            st.warning(f"Could not search for '{song.get('track', '')}': {e}")
        progress_bar.progress((i + 1) / len(songs), text=f"Searching for: {song.get('track', '')}")
    progress_bar.empty()
    return track_uris, not_found_tracks

def create_spotify_playlist(vibe, track_uris, sp_client):
    """Creates a new private playlist on Spotify."""
    try:
        user_id = sp_client.current_user()['id']
        playlist_name = f"VibeList AI: {vibe[:50]}"
        playlist = sp_client.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=False,
            description=f"Generated by VibeList AI for the vibe: '{vibe}'."
        )
        sp_client.playlist_add_items(playlist['id'], track_uris)
        return playlist['external_urls']['spotify'], playlist_name
    except Exception as e:
        st.error(f"Failed to create Spotify playlist: {e}")
        return None, None

# --- Streamlit UI ---

st.set_page_config(page_title="VibeList AI", page_icon="🎵", layout="wide")
st.title("🎵 VibeList AI")
st.subheader("Your personal AI DJ for creating the perfect Spotify playlist from any vibe.")

# --- Authentication Flow ---
auth_manager = get_spotify_auth_manager()
db_session = Session()

# Check for auth code in URL
if "code" in st.query_params:
    try:
        code = st.query_params["code"]
        token_info = auth_manager.get_access_token(code, as_dict=True)
        st.session_state['token_info'] = token_info
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Error getting access token: {e}")

# Main logic
if 'token_info' not in st.session_state:
    auth_url = auth_manager.get_authorize_url()
    st.info("To get started, you need to connect your Spotify account.")
    st.link_button("Login with Spotify", auth_url, use_container_width=True)
else:
    token_info = st.session_state['token_info']

    # Refresh token if expired
    if auth_manager.is_token_expired(token_info):
        try:
            st.session_state['token_info'] = auth_manager.refresh_access_token(token_info['refresh_token'])
            token_info = st.session_state['token_info']
            st.toast("Spotify token refreshed!")
        except Exception as e:
            st.error("Your session has expired. Please log in again.")
            del st.session_state['token_info']
            st.rerun()

    # --- Authenticated User Logic (with Database) ---
    try:
        sp_client = spotipy.Spotify(auth=token_info['access_token'])
        spotify_user_info = sp_client.current_user()
        spotify_id = spotify_user_info['id']

        # Check if user exists in our database
        current_user = db_session.query(User).filter_by(spotify_user_id=spotify_id).first()

        if not current_user:
            new_user = User(
                spotify_user_id=spotify_id,
                display_name=spotify_user_info['display_name'],
                refresh_token=token_info['refresh_token']
            )
            db_session.add(new_user)
            db_session.commit()
            current_user = new_user
            st.toast(f"Welcome, {current_user.display_name}! Your account has been saved.")
        else:
            # Update refresh token in case it changed
            current_user.refresh_token = token_info['refresh_token']
            db_session.commit()
        
        # Store our internal DB user ID in the session state for later use
        st.session_state['db_user_id'] = current_user.id
        st.success(f"Logged in as **{current_user.display_name}**")

        # --- Main App Interface ---
        st.markdown("---")
        st.header("1. Describe your Vibe")
        vibe_input = st.text_area(
            "What's the mood? Be as descriptive as you like!",
            placeholder="e.g., 'Late night coding session in a rainy city', 'Driving on the Mumbai Sea Link at sunset'",
            height=100
        )

        if st.button("✨ Generate Playlist ✨", type="primary", use_container_width=True):
            if not vibe_input:
                st.warning("Please describe a vibe first!")
            else:
                with st.spinner("🎧 Asking the AI DJ for recommendations..."):
                    songs = generate_song_list(vibe_input)
                
                if songs:
                    st.info(f"AI recommended {len(songs)} songs. Finding them on Spotify...")
                    track_uris, not_found = find_spotify_tracks(songs, sp_client)

                    if not_found:
                        with st.expander(f"⚠️ Couldn't find {len(not_found)} tracks"):
                            for track in not_found:
                                st.write(f"- {track}")

                    if track_uris:
                        with st.spinner("🎶 Creating your new playlist on Spotify..."):
                            playlist_url, playlist_name = create_spotify_playlist(vibe_input, track_uris, sp_client)
                        
                        if playlist_url:
                            # Save to database
                            try:
                                new_playlist_db = Playlist(
                                    playlist_name=playlist_name,
                                    spotify_playlist_url=playlist_url,
                                    vibe_description=vibe_input,
                                    user_id=st.session_state['db_user_id']
                                )
                                db_session.add(new_playlist_db)
                                db_session.commit()
                                st.toast("Playlist saved to your history!")
                            except Exception as e:
                                st.warning(f"Could not save playlist to history: {e}")

                            st.balloons()
                            st.header("🎉 Your playlist is ready!")
                            st.markdown(f"**[Click here to open '{playlist_name}' in Spotify]({playlist_url})**")
                    else:
                        st.error("Could not find any of the recommended songs on Spotify. Try a different vibe!")

        # --- New Multi-User Features ---
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.header("📜 My Playlist History")
            my_playlists = db_session.query(Playlist).filter_by(user_id=st.session_state['db_user_id']).order_by(Playlist.created_at.desc()).all()
            if not my_playlists:
                st.info("You haven't created any playlists yet. Generate one above to see it here!")
            else:
                for p in my_playlists:
                    with st.container(border=True):
                        st.markdown(f"**[{p.playlist_name}]({p.spotify_playlist_url})**")
                        st.caption(f"Vibe: '{p.vibe_description}' | Created on: {p.created_at.strftime('%d-%b-%Y')}")
        
        with col2:
            st.header("🌍 Community Playlists")
            community_playlists = db_session.query(Playlist).order_by(Playlist.created_at.desc()).limit(10).all()
            if not community_playlists:
                st.info("No playlists have been created by the community yet.")
            else:
                for p in community_playlists:
                    with st.container(border=True):
                        st.markdown(f"**[{p.playlist_name}]({p.spotify_playlist_url})**")
                        st.caption(f"Vibe: '{p.vibe_description}' | By: {p.creator.display_name}")


    except Exception as e:
        st.error(f"An error occurred. It might be related to your Spotify connection. Details: {e}")
        # Clear potentially corrupt session state
        if 'token_info' in st.session_state:
            del st.session_state['token_info']
        if 'db_user_id' in st.session_state:
            del st.session_state['db_user_id']
        st.button("Log in again")


    st.markdown("---")
    if st.button("Logout of Spotify"):
        # Clear all session data
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("You have been logged out.")
        st.rerun()

db_session.close()
