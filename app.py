import streamlit as st
import google.generativeai as genai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os

# --- App Configuration ---
st.set_page_config(
    page_title="Vibelist: AI Playlist Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #282828;
        color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #444444;
    }
    .stButton > button {
        background-color: #1DB954;
        color: #FFFFFF;
        border-radius: 50px;
        padding: 10px 24px;
        border: none;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #1ED760;
    }
    .stSpinner > div {
        border-top-color: #1DB954;
    }
    .stAlert {
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #1DB954;
    }
</style>
""", unsafe_allow_html=True)


# --- Function to call Gemini API ---
def generate_song_ideas(theme, track_count, api_key):
    """Generates song ideas using the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

        prompt = f"""
        You are an expert music curator named Vibemaster. Your task is to generate a list of songs based on a user's theme or vibe.

        The user's theme is: "{theme}"

        Generate a list of exactly {track_count} songs that fit this theme. Also, generate a creative and fitting title for this playlist.

        You MUST format your response as a valid JSON object. The object should have two keys:
        1. "playlist_title": A string containing the creative playlist title.
        2. "playlist_tracks": An array of objects. Each object in the array must have two keys: "song_title" and "artist".

        Do not include any text, explanations, or markdown formatting before or after the JSON object.
        Example response:
        {{
          "playlist_title": "Neon City Nights",
          "playlist_tracks": [
            {{
              "song_title": "Blinding Lights",
              "artist": "The Weeknd"
            }},
            {{
              "song_title": "Nightcall",
              "artist": "Kavinsky"
            }}
          ]
        }}
        """

        response = model.generate_content(prompt)
        # Clean up potential markdown formatting from the response
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)

    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        st.error(f"Raw AI Response was: {response.text if 'response' in locals() else 'No response'}")
        return None


# --- Function to Create Spotify Playlist ---
def create_spotify_playlist(username, client_id, client_secret, redirect_uri, playlist_title, track_list):
    """Creates a Spotify playlist and adds tracks."""
    try:
        scope = "playlist-modify-public"
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
            username=username,
            cache_path=f".cache-{username}"
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)

        # Create the playlist
        user_id = sp.current_user()['id']
        playlist = sp.user_playlist_create(user_id, playlist_title, public=True)
        playlist_id = playlist['id']
        playlist_url = playlist['external_urls']['spotify']

        # Search for tracks and add them
        track_uris = []
        not_found = []
        for track in track_list:
            query = f"track:{track['song_title']} artist:{track['artist']}"
            result = sp.search(q=query, type='track', limit=1)
            if result['tracks']['items']:
                track_uris.append(result['tracks']['items'][0]['uri'])
            else:
                not_found.append(f"{track['song_title']} by {track['artist']}")

        # Add tracks to the playlist in batches of 100
        if track_uris:
            for i in range(0, len(track_uris), 100):
                batch = track_uris[i:i+100]
                sp.playlist_add_items(playlist_id, batch)

        return playlist_url, not_found

    except Exception as e:
        st.error(f"An error occurred with Spotify: {e}")
        st.warning("Please double-check your Spotify credentials and ensure the Redirect URI is set correctly in your Spotify Developer App settings.")
        return None, []


# --- Streamlit UI ---

st.title("🎵 Vibelist")
st.header("Your AI-Powered Playlist Generator")
st.write("Describe a mood, a memory, or a vibe, and let AI curate the perfect playlist for you on Spotify.")

# --- Setup Columns ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Step 1: Describe Your Vibe")
    theme_prompt = st.text_area(
        "Enter a theme for your playlist:",
        placeholder="e.g., A melancholic train ride on a rainy autumn day, 90s hip-hop for a summer BBQ, coding late into the night...",
        height=150
    )
    track_count = st.slider("How many songs?", min_value=10, max_value=100, value=20)

    st.subheader("Step 2: Connect to Your Accounts")
    # Use st.secrets for the Google API key if deployed, otherwise allow manual input
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        st.info("Google AI API Key loaded from secrets.", icon="✅")
    except (FileNotFoundError, KeyError):
        google_api_key = st.text_input("Enter your Google AI API Key:", type="password")


    spotify_client_id = st.secrets["SPOTIFY_CLIENT_ID"]
    spotify_client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
    spotify_username = st.text_input("Enter your Spotify Username (or User ID):")
    spotify_redirect_uri = st.text_input(
        "Enter your Spotify Redirect URI:",
        value="https://vibelist-ai.streamlit.app/",
        help="This must exactly match the URI you set in your Spotify Developer Dashboard."
    )

    st.subheader("Step 3: Generate!")
    if st.button("Create My Playlist"):
        if not all([theme_prompt, track_count, google_api_key, spotify_client_id, spotify_client_secret, spotify_username, spotify_redirect_uri]):
            st.warning("Please fill in all the fields before generating.")
        else:
            with st.spinner("Brainstorming song ideas with the AI..."):
                song_data = generate_song_ideas(theme_prompt, track_count, google_api_key)

            if song_data and song_data.get("playlist_tracks"):
                playlist_title = song_data.get("playlist_title", f"Vibelist: {theme_prompt[:30]}")
                st.success(f"AI generated a playlist idea: **{playlist_title}**")
                
                with st.spinner("Connecting to Spotify and creating your playlist... This may require you to log in on a new browser tab."):
                    playlist_url, not_found_tracks = create_spotify_playlist(
                        username=spotify_username,
                        client_id=spotify_client_id,
                        client_secret=spotify_client_secret,
                        redirect_uri=spotify_redirect_uri,
                        playlist_title=playlist_title,
                        track_list=song_data["playlist_tracks"]
                    )

                if playlist_url:
                    st.balloons()
                    st.success("Your playlist has been created!")
                    st.markdown(f"### [Click here to listen on Spotify]({playlist_url})", unsafe_allow_html=True)

                    if not_found_tracks:
                        st.warning("Could not find the following tracks on Spotify:")
                        for track in not_found_tracks:
                            st.write(f"- {track}")
            else:
                st.error("Could not generate song ideas. Please try a different prompt.")

with col2:
    with st.expander("How to Get Your Spotify Credentials", expanded=True):
        st.markdown("""
        1.  **Go to the Spotify Developer Dashboard:**
            [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
        2.  **Log in** with your Spotify account.
        3.  Click **'Create App'**. Give it a name (e.g., "Vibelist App") and a description.
        4.  Once created, you'll see your **Client ID** and you can click **'Show client secret'** to get the secret key.
        5.  Click **'Edit Settings'**. In the 'Redirect URIs' section, add the URI you see in the input box to the left. For local testing, this is usually `http://localhost:8501`.
        6.  Click **'Save'** at the bottom.
        7.  Your **Spotify Username** can be found in your Spotify account settings.
        """)
    with st.expander("How to Get Your Google AI API Key"):
        st.markdown("""
        1. Go to [Google AI Studio](https://aistudio.google.com/).
        2. Log in with your Google account.
        3. Click on the **'Get API key'** option.
        4. **'Create API key in new project'**.
        5. Copy the generated key and paste it into the input box.
        """)
