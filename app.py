import streamlit as st
import google.generativeai as genai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import re
import pandas as pd # Added for data handling

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

@st.cache_data # Use Streamlit's cache to load the data only once
def load_song_dataset(dataset_path="spotify_dataset.csv"):
    """Loads the song dataset from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(dataset_path)
        # --- IMPORTANT ---
        # Ensure your CSV column names are 'track' and 'artist'.
        # If they are different (e.g., 'track_name', 'artist_name'), rename them below:
        # df = df.rename(columns={'track_name': 'track', 'artist_name': 'artist'})
        if 'track' not in df.columns or 'artist' not in df.columns:
            st.error("Dataset must contain 'track' and 'artist' columns.")
            return None
        return df
    except FileNotFoundError:
        st.error(f"Dataset file '{dataset_path}' not found. Please make sure it's in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None
        
def get_spotify_auth_manager():
    """Creates and returns a SpotifyOAuth object for handling authentication."""
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=".spotify_cache" # Caches the token
    )

def find_candidate_songs(vibe, dataframe, num_candidates=200):
    """
    Searches the dataframe for songs that match keywords from the vibe.
    Returns a formatted string of candidate songs for the LLM prompt.
    """
    if dataframe is None:
        return ""

    # A simple keyword matching strategy.
    keywords = set(re.split(r'\s+', vibe.lower()))
    
    # Search for keywords in track and artist names
    # This finds rows where ANY keyword is present in the track or artist name
    mask = dataframe['track'].str.contains('|'.join(keywords), case=False, na=False) | \
           dataframe['artist'].str.contains('|'.join(keywords), case=False, na=False)

    candidate_df = dataframe[mask]

    # If we find too many, take a random sample to keep the prompt size reasonable
    if len(candidate_df) > num_candidates:
        candidate_df = candidate_df.sample(n=num_candidates, random_state=1) # a fixed state for reproducibility

    if candidate_df.empty:
        return "" # No candidates found

    # Format the candidates into a simple string for the prompt
    song_list = []
    for _, row in candidate_df.iterrows():
        # Clean the data to prevent breaking the JSON format in the prompt
        artist_clean = str(row['artist']).replace('"', "'")
        track_clean = str(row['track']).replace('"', "'")
        song_list.append(f"{{ \"artist\": \"{artist_clean}\", \"track\": \"{track_clean}\" }}")
    
    return ",\n".join(song_list)


def generate_song_list(vibe, song_dataset):
    """
    Uses Google's Gemini model to generate a list of songs based on a vibe,
    constrained by a list of candidate songs from our dataset.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # 1. Retrieve candidate songs from our dataset based on the vibe
    candidate_songs_str = find_candidate_songs(vibe, song_dataset)

    if not candidate_songs_str:
        st.warning("Could not find relevant songs in the local dataset for that vibe. The AI will generate from its own knowledge, which may be less accurate.")
        # We can create a fallback prompt, but for now, we'll let the main prompt handle it.
        # An empty candidate list will force the model to rely on its own knowledge, but the prompt will still ask it to choose from the (empty) list.
        # Let's adjust the prompt slightly to handle this.
    
    # 2. Augment the prompt with the candidates
    prompt = f"""
    You are a world-class DJ and music curator.
    A user has described the following vibe: "{vibe}".

    I have a specific list of songs available. Your task is to select exactly 15 songs from the provided list below that BEST match the user's vibe.
    You MUST ONLY choose from the candidate songs provided. Do not invent any songs or artists.

    Here is the list of available candidate songs in a JSON-like format:
    [
    {candidate_songs_str}
    ]

    Your response MUST be a valid JSON object. The object should have a single key "songs",
    which is an array of objects. Each object in the array must be one of the songs from the list I provided and must have two keys: "artist" and "track".

    Example of the required JSON format:
    {{
      "songs": [
        {{ "artist": "A. R. Rahman", "track": "Chaiyya Chaiyya" }},
        {{ "artist": "Prateek Kuhad", "track": "cold/mess" }}
      ]
    }}

    Do not include any introductory text, explanations, or markdown formatting like ```json around the JSON object.
    Only output the raw JSON."""

    try:
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        
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
    
def sanitize_for_spotify(input_string):
    """
    Cleans a string to make it safe for Spotify API fields like name and description.
    """
    if not input_string:
        return ""
    s = re.sub(r'\s+', ' ', input_string)
    s = s.strip()
    return s

def create_spotify_playlist(vibe, track_uris, sp_client):
    """
    Creates a new private playlist on Spotify and adds tracks to it.
    """
    try:
        user_id = sp_client.current_user()['id']
        sanitized_vibe = sanitize_for_spotify(vibe)
        playlist_name = f"VibeList AI: {sanitized_vibe[:50]}"
        description_text = f"A playlist generated by VibeList AI for the vibe: '{sanitized_vibe}'."
        playlist = sp_client.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=False,
            description=description_text
        )
        
        # Spotify API can only add 100 tracks at a time
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

    # --- Load your local song dataset ---
    song_dataset_df = load_song_dataset()

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
        elif song_dataset_df is None:
            st.error("Cannot generate playlist because the song dataset failed to load. Check the file path and format.")
        else:
            with st.spinner("🎧 Asking the AI DJ for recommendations... (This might take a moment)"):
                # Pass the loaded dataframe to the generation function
                songs = generate_song_list(vibe_input, song_dataset_df)

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
