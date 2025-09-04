import streamlit as st
import google.generativeai as genai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import re

# --- Configuration ---
# Load secrets from Streamlit's secrets management
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SPOTIFY_CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
    SPOTIFY_CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
    
    # This is the Redirect URI for your deployed app. 
    # For local testing, change this to "http://localhost:8501" 
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

def get_next_song_suggestion(vibe, found_songs, rejected_songs):
    """Asks the LLM for a single new song suggestion based on the current context."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    found_songs_str = "\n".join([f"- {s['track']} by {s['artist']}" for s in found_songs])
    rejected_songs_str = ", ".join(rejected_songs)
    prompt = f"""
    You are a music curator building a 15-song playlist for the vibe: "{vibe}".
    
    Here are the songs you have successfully found so far:
    {found_songs_str if found_songs else "None yet."}

    Here are the songs you suggested that could NOT be found on Spotify. Do NOT suggest these again:
    {rejected_songs_str if rejected_songs else "None."}

    Your task is to suggest the NEXT SINGLE SONG to add to the playlist.
    
    Your response MUST be a valid JSON object with two keys: "artist" and "track".
    Example: {{ "artist": "Arijit Singh", "track": "Tera Yaar Hoon Main" }}
    
    Only output the raw JSON.
    """
    try:
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        return json.loads(response.text)
    except Exception as e:
        st.warning(f"AI failed to suggest a song: {e}")
        return None

def get_clarification_from_llm(original_suggestion, spotify_options):
    """Asks the LLM to choose from a list of possible matches from Spotify."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    options_str = "\n".join([f"- {opt['track']} by {opt['artist']}" for opt in spotify_options])
    
    prompt = f"""
    You are a helpful music assistant. I was trying to find your suggestion "{original_suggestion['track']} by {original_suggestion['artist']}" on Spotify, but couldn't find an exact match.
    
    However, Spotify provided the following similar tracks:
    {options_str}

    Is one of these the correct song you intended to suggest? 
    
    - If YES, respond with the corrected JSON for that song (e.g., {{ "artist": "Correct Artist", "track": "Correct Track" }}).
    - If NO, respond with JSON: {{ "artist": "None", "track": "None" }}.
    
    Only output the raw JSON.
    """
    try:
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        clarified_choice = json.loads(response.text)
        if clarified_choice.get("track") == "None":
            return None
        return clarified_choice
    except Exception as e:
        st.warning(f"AI failed to clarify: {e}")
        return None

def verify_song_on_spotify(suggestion, sp_client):
    """
    Verifies a song suggestion on Spotify with a 3-tier logic:
    1. Exact search.
    2. Broad search for clarification.
    3. Hard fail.
    Returns a dictionary with status and data.
    """
    track_name = suggestion['track']
    artist_name = suggestion['artist']
    
    # 1. Exact Search
    query_exact = f"track:\"{track_name}\" artist:\"{artist_name}\""
    results_exact = sp_client.search(q=query_exact, type='track', limit=1)
    if results_exact['tracks']['items']:
        return {'status': 'FOUND', 'data': results_exact['tracks']['items'][0]}

    # 2. Broaden Search if Exact Fails
    query_broad = f"track:\"{track_name}\""
    results_broad = sp_client.search(q=query_broad, type='track', limit=5)
    if results_broad['tracks']['items']:
        options = [
            {'track': item['name'], 'artist': item['artists'][0]['name']} 
            for item in results_broad['tracks']['items'] if item and item['artists']
        ]
        if options:
            return {'status': 'NEEDS_CLARIFICATION', 'data': options}
        
    # 3. Hard Fail
    return {'status': 'NOT_FOUND', 'data': None}


def generate_verified_song_list(vibe, sp_client, num_songs=15):
    """
    Runs the collaborative loop with an advanced clarification step. This is the master function.
    """
    verified_tracks = []
    rejected_suggestions = []
    max_attempts = num_songs * 4 # Safety break to prevent infinite loops

    progress_bar = st.progress(0, text=f"Starting playlist generation... (0/{num_songs})")

    for attempt in range(max_attempts):
        if len(verified_tracks) >= num_songs:
            break

        progress_text = f"Found {len(verified_tracks)}/{num_songs} songs. Asking AI for the next idea..."
        progress_bar.progress(len(verified_tracks) / num_songs, text=progress_text)
        
        found_songs_info = [{'artist': t['artist'], 'track': t['track']} for t in verified_tracks]
        suggestion = get_next_song_suggestion(vibe, found_songs_info, rejected_suggestions)

        if not suggestion or not suggestion.get('track') or not suggestion.get('artist'):
            continue

        suggestion_str = f"{suggestion['track']} by {suggestion['artist']}"
        if suggestion_str in rejected_suggestions:
            continue
            
        progress_bar.progress(len(verified_tracks) / num_songs, text=f"Found {len(verified_tracks)}/{num_songs}. Verifying: '{suggestion_str}'...")

        try:
            verification = verify_song_on_spotify(suggestion, sp_client)

            if verification['status'] == 'FOUND':
                track_data = verification['data']
                st.toast(f"✅ Found: {track_data['name']} by {track_data['artists'][0]['name']}")
                verified_tracks.append({
                    'uri': track_data['uri'],
                    'track': track_data['name'],
                    'artist': track_data['artists'][0]['name']
                })

            elif verification['status'] == 'NEEDS_CLARIFICATION':
                st.toast(f"🤔 '{suggestion_str}' wasn't an exact match. Asking AI to clarify...")
                clarified_suggestion = get_clarification_from_llm(suggestion, verification['data'])
                
                if clarified_suggestion:
                    # Final check on the clarified choice
                    final_verification = verify_song_on_spotify(clarified_suggestion, sp_client)
                    if final_verification['status'] == 'FOUND':
                        track_data = final_verification['data']
                        st.toast(f"✅ Corrected & Found: {track_data['name']}")
                        verified_tracks.append({
                            'uri': track_data['uri'],
                            'track': track_data['name'],
                            'artist': track_data['artists'][0]['name']
                        })
                    else: # Clarified choice also failed
                        rejected_suggestions.append(suggestion_str)
                else: # LLM said none were correct
                    st.toast(f"❌ AI confirmed no good match for '{suggestion_str}'.")
                    rejected_suggestions.append(suggestion_str)

            elif verification['status'] == 'NOT_FOUND':
                st.toast(f"❌ '{suggestion_str}' not found. Asking AI for another.")
                rejected_suggestions.append(suggestion_str)
        
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            rejected_suggestions.append(suggestion_str)


    progress_bar.empty()
    if len(verified_tracks) < num_songs:
        st.warning(f"Could only find {len(verified_tracks)} songs after {max_attempts} attempts. The playlist will be created with these songs.")

    return [track['uri'] for track in verified_tracks]

def sanitize_for_spotify(input_string):
    """Cleans a string to make it safe for Spotify API fields like name and description."""
    if not input_string:
        return ""
    s = re.sub(r'\s+', ' ', input_string)
    s = s.strip()
    return s

def create_spotify_playlist(vibe, track_uris, sp_client):
    """Creates a new private playlist on Spotify and adds tracks to it."""
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

query_params = st.query_params
if "code" in query_params:
    try:
        code = query_params["code"]
        token_info = auth_manager.get_access_token(code)
        st.session_state['token_info'] = token_info
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Error getting access token: {e}")

# Main logic
if 'token_info' not in st.session_state:
    auth_url = auth_manager.get_authorize_url()
    st.info("To get started, you need to connect your Spotify account.")
    st.link_button("Login with Spotify", auth_url)
else:
    token_info = st.session_state['token_info']
    
    if auth_manager.is_token_expired(token_info):
        try:
            st.session_state['token_info'] = auth_manager.refresh_access_token(token_info['refresh_token'])
            token_info = st.session_state['token_info']
            st.toast("Spotify token refreshed!")
        except Exception as e:
            st.error("Your session has expired. Please log in again.")
            del st.session_state['token_info']
            st.rerun()

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
            # This single function call now handles the entire intelligent loop.
            # It will return a list of URIs that are 100% verified to exist on Spotify.
            verified_uris = generate_verified_song_list(vibe_input, sp_client)

            if verified_uris:
                with st.spinner("🎶 Creating your new playlist on Spotify..."):
                    playlist_url = create_spotify_playlist(vibe_input, verified_uris, sp_client)
                
                if playlist_url:
                    st.balloons()
                    st.header("🎉 Your playlist is ready!")
                    st.markdown(f"**[Click here to open your new playlist in Spotify]({playlist_url})**", unsafe_allow_html=True)
            else:
                st.error("Could not find enough songs for that vibe after multiple attempts. Please try being more specific or using a different vibe!")

    st.markdown("---")
    if st.button("Logout of Spotify"):
        del st.session_state['token_info']
        st.success("You have been logged out.")
        st.rerun()
