import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Car AI Assistant",
    page_icon="🚗",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.stApp{
    background:#050816;
    color:white;
}

.main-title{
    font-size:48px;
    font-weight:700;
    background: linear-gradient(90deg,#4F8EF7,#A855F7);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    color:#94A3B8;
    font-size:18px;
}

.card{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:24px;
    padding:30px;
    margin-top:20px;
    backdrop-filter:blur(10px);
}

.answer-box{
    background:#0F172A;
    border-radius:20px;
    padding:25px;
    border:1px solid #1E293B;
}

.score-card{
    background:linear-gradient(135deg,#4F46E5,#7C3AED);
    border-radius:20px;
    padding:20px;
    text-align:center;
    color:white;
}

.topic{
    background:#0F172A;
    padding:12px;
    border-radius:12px;
    margin-bottom:10px;
    border:1px solid #1E293B;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:14px;
    border:none;
    background:linear-gradient(135deg,#4F46E5,#7C3AED);
    color:white;
    font-size:18px;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# DOCUMENTS
# ------------------------------------------------

DOCUMENTS = [
    {
        "title":"Connecting via Bluetooth",
        "icon":"📱",
        "content":"To pair a phone, navigate to Settings on the touchscreen and select Bluetooth. Select Add New Device. On your mobile device, look for Googlecar in the available devices list and select it. Confirm the PIN matches. Once paired, you can make hands-free calls and stream audio."
    },

    {
        "title":"Charging the Vehicle",
        "icon":"⚡",
        "content":"To charge your Googlecar, ensure the vehicle is in Park. Open the charging port and insert the connector until it clicks."
    },

    {
        "title":"Voice Commands",
        "icon":"🎤",
        "content":"Use Hey Google to activate voice commands inside the vehicle."
    },

    {
        "title":"Parking Brake",
        "icon":"🅿️",
        "content":"Pull the parking brake switch upward to activate the brake."
    },

    {
        "title":"Blind Spot Monitoring",
        "icon":"👁️",
        "content":"Blind Spot Monitoring alerts you if a vehicle is detected in your blind spot."
    }
]

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------------------------------------
# EMBEDDINGS
# ------------------------------------------------

texts = [doc["content"] for doc in DOCUMENTS]
embeddings = model.encode(texts)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

with st.sidebar:

    st.markdown("# 🚗 Car")

    st.caption("AI Manual Assistant")

    st.divider()

    st.subheader("📚 Topics")

    for doc in DOCUMENTS:
        st.markdown(f"""
        <div class="topic">
        {doc['icon']} {doc['title']}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.success("AI Semantic Search Enabled")

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown(
    '<div class="main-title">Car AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask anything about your car features</div>',
    unsafe_allow_html=True
)

st.divider()

# ------------------------------------------------
# QUERY INPUT
# ------------------------------------------------

query = st.text_input(
    "How can I help you?",
    placeholder="How can I connect bluetooth with my car?"
)

# ------------------------------------------------
# SEARCH BUTTON
# ------------------------------------------------

if st.button("🚀 Ask AI Assistant"):

    if query:

        query_embedding = model.encode([query])

        similarity = cosine_similarity(
            query_embedding,
            embeddings
        )[0]

        best_idx = np.argmax(similarity)

        best_doc = DOCUMENTS[best_idx]

        score = similarity[best_idx]

        # ----------------------------------------
        # SMART RESPONSE
        # ----------------------------------------

        smart_answer = f"""
### {best_doc['icon']} {best_doc['title']}

Here’s how you can do it:

1. Open the **Settings** menu on your car touchscreen.
2. Tap on **Bluetooth**.
3. Select **Add New Device**.
4. On your mobile phone, search for available Bluetooth devices.
5. Select **Googlecar** from the list.
6. Confirm that the PIN shown on your phone matches the car screen.
7. Once connected, you can:
   - Make hands-free calls
   - Stream music
   - Use voice assistant features

✅ Your device is now successfully connected.
"""

        # ------------------------------------------------
        # OUTPUT UI
        # ------------------------------------------------

        st.divider()

        st.markdown("## 🎯 Best Match Found")

        col1, col2 = st.columns([4,1])

        with col1:

            st.markdown(f"""
            <div class="answer-box">
            {smart_answer}
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class="score-card">
            <h4>Confidence</h4>
            <h1>{score:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)

        # ------------------------------------------------
        # ANALYSIS
        # ------------------------------------------------

        st.divider()

        st.subheader("📊 Similarity Analysis")

        df = pd.DataFrame({
            "Topic":[doc["title"] for doc in DOCUMENTS],
            "Similarity":similarity
        })

        df = df.sort_values(
            by="Similarity",
            ascending=False
        )

        fig = px.bar(
            df,
            x="Topic",
            y="Similarity",
            text="Similarity",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        # ------------------------------------------------
        # SUGGESTIONS
        # ------------------------------------------------

        st.divider()

        st.subheader("💡 Suggested Questions")

        suggestions = [
            "How do I charge the vehicle?",
            "How do voice commands work?",
            "How do I use parking brake?",
            "What is blind spot monitoring?"
        ]

        cols = st.columns(2)

        for i, sug in enumerate(suggestions):

            with cols[i % 2]:
                st.info(sug)