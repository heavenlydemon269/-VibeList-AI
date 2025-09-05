import streamlit as st
import google.generativeai as genai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os

# --- Configuration ---
# Load secrets from Streamlit's secrets management
try:
    # Switched from OpenAI to Google Gemini
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SPOTIFY_CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
    SPOTIFY_CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
    
    # This is the Redirect URI for your deployed app. 
    # For local testing, you must change this to "http://localhost:8501" 
    # AND update it in your Spotify Developer Dashboard.
    REDIRECT_URI = "https://vibelist-ai.streamlit.app/"
except FileNotFoundError:
    st.error("Secrets file not found. Make sure you have a .streamlit/secrets.toml file with your API keys.")
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
    st.error(f"Failed to configure Google AI client. Please check your API key. Error: {e}")
    st.stop()

# --- Helper Functions ---

def get_spotify_auth_manager():
    """Creates and returns a SpotifyOAuth object for handling authentication."""
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=".spotify_cache" # Caches the token
    )

def generate_song_list(vibe):
    """
    Uses Google's Gemini model to generate a list of songs based on a vibe.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    prompt = f"""
    You are a world-class DJ and music curator, a master of crafting the perfect playlist for any mood or vibe.
    A user has described the following vibe: "{vibe}".

    Based on this vibe, generate a list of 15 songs that perfectly capture this feeling.
    Your response MUST be a valid JSON object. The object should have a single key "songs", 
    which is an array of objects. Each object in the array must have two keys: "artist" and "track".

    Example of the required JSON format:
    {{
      "songs": [
        {{ "artist": "A. R. Rahman", "track": "Chaiyya Chaiyya" }},
        {{ "artist": "Prateek Kuhad", "track": "cold/mess" }},
        {{ "artist": "Lana Del Rey", "track": "Summertime Sadness" }}
      ]
    }}

    Do not include any introductory text, explanations, or markdown formatting like ```json around the JSON object.
    Only output the raw JSON.
    """
    
    try:
        # Gemini has a specific generation config for forcing JSON output
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # The response text should be a clean JSON string
        song_data = json.loads(response.text)
        return song_data.get("songs", [])
    except json.JSONDecodeError as e:
        st.error(f"The AI returned an invalid format. It might be helpful to try a different vibe. Details: {e}")
        st.error(f"AI Raw Response: {response.text}") # Show what the AI sent back
        return None
    except Exception as e:
        st.error(f"An error occurred while communicating with the AI: {e}")
        return None


def find_spotify_tracks(songs, sp_client):
    """
    Searches for tracks on Spotify and returns a list of their URIs.
    """
    track_uris = []
    not_found_tracks = []
    
    progress_bar = st.progress(0, text="Searching for tracks on Spotify...")
    
    for i, song in enumerate(songs):
        artist = song.get("artist", "")
        track = song.get("track", "")
        query = f"track:{track} artist:{artist}"
        
        try:
            results = sp_client.search(q=query, type='track', limit=1)
            tracks = results['tracks']['items']
            if tracks:
                track_uris.append(tracks[0]['uri'])
            else:
                not_found_tracks.append(f"{track} by {artist}")
        except Exception as e:
            st.warning(f"Could not search for '{track} by {artist}': {e}")

        progress_bar.progress((i + 1) / len(songs), text=f"Searching for: {track} by {artist}")

    progress_bar.empty() # Clear the progress bar
    return track_uris, not_found_tracks


def create_spotify_playlist(vibe, track_uris, sp_client):
    """
    Creates a new private playlist on Spotify and adds tracks to it.
    """
    try:
        user_id = sp_client.current_user()['id']
        playlist_name = f"VibeList AI: {vibe[:50]}" # Truncate vibe for playlist name
        
        playlist = sp_client.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=False,
            description=f"A playlist generated by VibeList AI for the vibe: '{vibe}'."
        )
        
        # Spotify API can only add 100 tracks at a time, so we chunk it just in case
        chunk_size = 100
        for i in range(0, len(track_uris), chunk_size):
            chunk = track_uris[i:i+chunk_size]
            sp_client.playlist_add_items(playlist['id'], chunk)
            
        return playlist['external_urls']['spotify']
    except Exception as e:
        st.error(f"Failed to create Spotify playlist: {e}")
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="VibeList AI", page_icon="🎵")
st.title("🎵 VibeList AI")
st.subheader("Your personal AI DJ for creating the perfect Spotify playlist from any vibe.")

# --- Authentication Flow ---

auth_manager = get_spotify_auth_manager()

# Check if we are in the redirect phase
query_params = st.query_params
if "code" in query_params:
    try:
        code = query_params["code"]
        token_info = auth_manager.get_access_token(code)
        st.session_state['token_info'] = token_info
        # Clear query params to prevent re-running this block
        st.query_params.clear()
        st.rerun() # Rerun the script to move to the main app logic
    except Exception as e:
        st.error(f"Error getting access token: {e}")

# Main logic
if 'token_info' not in st.session_state:
    auth_url = auth_manager.get_authorize_url()
    st.info("To get started, you need to connect your Spotify account.")
    # Using st.link_button for a clean redirect
    st.link_button("Login with Spotify", auth_url)
else:
    # User is authenticated
    token_info = st.session_state['token_info']
    
    # Check if token is expired and refresh if needed
    if auth_manager.is_token_expired(token_info):
        try:
            st.session_state['token_info'] = auth_manager.refresh_access_token(token_info['refresh_token'])
            token_info = st.session_state['token_info']
            st.toast("Spotify token refreshed!")
        except Exception as e:
            st.error("Your session has expired. Please log in again.")
            del st.session_state['token_info']
            st.rerun()

    # Create the authenticated Spotipy client
    try:
        sp_client = spotipy.Spotify(auth=token_info['access_token'])
        user_info = sp_client.current_user()
        st.success(f"Logged in as **{user_info['display_name']}**")
    except Exception as e:
        st.error(f"Error connecting to Spotify. Please try logging in again. Details: {e}")
        del st.session_state['token_info']
        st.rerun()


    # --- Main App Interface ---
    st.markdown("---")
    st.header("1. Describe your Vibe")
    vibe_input = st.text_area(
        "What's the mood? Be as descriptive as you like!",
        placeholder="e.g., 'Late night coding session in a rainy city', 'Driving on the Mumbai Sea Link at sunset', 'Sunday morning coffee and a good book'",
        height=100
    )

    if st.button("✨ Generate Playlist ✨", type="primary", use_container_width=True):
        if not vibe_input:
            st.warning("Please describe a vibe first!")
        else:
            with st.spinner("🎧 Asking the AI DJ for recommendations... (This might take a moment)"):
                songs = generate_song_list(vibe_input)

            if songs:
                st.info(f"AI recommended {len(songs)} songs. Now finding them on Spotify...")
                
                track_uris, not_found = find_spotify_tracks(songs, sp_client)
                
                if not_found:
                    with st.expander(f"⚠️ Couldn't find {len(not_found)} tracks on Spotify"):
                        for track in not_found:
                            st.write(f"- {track}")

                if track_uris:
                    with st.spinner("🎶 Creating your new playlist on Spotify..."):
                        playlist_url = create_spotify_playlist(vibe_input, track_uris, sp_client)
                    
                    if playlist_url:
                        st.balloons()
                        st.header("🎉 Your playlist is ready!")
                        st.markdown(f"**[Click here to open your new playlist in Spotify]({playlist_url})**", unsafe_allow_html=True)
                else:
                    st.error("Could not find any of the recommended songs on Spotify. Try a different vibe!")

    st.markdown("---")
    if st.button("Logout of Spotify"):
        del st.session_state['token_info']
        st.success("You have been logged out.")
        st.rerun()

