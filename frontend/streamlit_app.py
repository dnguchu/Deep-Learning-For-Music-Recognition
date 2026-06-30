import os

import requests
import streamlit as st


st.set_page_config(page_title="Music Recognition", page_icon="🎵", layout="wide")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1f2937 100%);
        color: #e5e7eb;
    }
    .hero {
        padding: 2rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
    }
    .hero h1 {
        margin-bottom: 0.25rem;
        font-size: 3rem;
    }
    .hero p {
        margin-top: 0;
        color: #cbd5e1;
        font-size: 1.05rem;
    }
    .result-card {
        padding: 1rem 1.25rem;
        border-radius: 18px;
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>Music Recognition</h1>
        <p>Upload a short audio clip and let the backend compare its embedding against the cached reference songs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Backend")
    st.caption("Update this if your FastAPI server runs elsewhere.")
    api_url = st.text_input("API base URL", value=API_BASE_URL)
    top_k = st.slider("Top matches", min_value=1, max_value=10, value=5)
    st.link_button("API health", f"{api_url.rstrip('/')}/health")


uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav", "m4a", "ogg"])


col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Request")
    st.write("The app sends your file to the backend for embedding extraction and song matching.")
    if uploaded_file is not None:
        st.audio(uploaded_file)

with col2:
    st.subheader("Result")
    if st.button("Recognize song", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("Upload an audio file first.")
        else:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
                response = requests.post(
                    f"{api_url.rstrip('/')}/recognize",
                    params={"top_k": top_k},
                    files=files,
                    timeout=120,
                )
                response.raise_for_status()
                payload = response.json()

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3 style="margin-top:0;">Best Match</h3>
                        <p><strong>{payload.get('matched_song') or 'No match returned'}</strong></p>
                        <p>Distance: {payload.get('matched_distance')}</p>
                        <p>Embedding shape: {payload.get('embedding_shape')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.subheader("Top candidates")
                st.dataframe(payload.get("candidates", []), use_container_width=True, hide_index=True)
            except requests.RequestException as exc:
                st.error(f"Backend request failed: {exc}")
