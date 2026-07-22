"""
ui.py — Streamlit UI for the Enterprise RAG System.

Features:
  - Supabase Login / Signup
  - PDF Document Upload
  - RAG Chat Interface
  - Execution Trace Panel (cache hits, security, CRAG, Self-RAG)
"""

import streamlit as st
import requests
import json
import time

# ── Configuration ────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .trace-box {
        background: rgba(30, 30, 60, 0.8);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .cache-hit {
        color: #48bb78;
        font-weight: bold;
    }
    .cache-miss {
        color: #f56565;
    }
    .blocked {
        color: #f56565;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "paused_thread_id" not in st.session_state:
    st.session_state.paused_thread_id = None
if "paused_query" not in st.session_state:
    st.session_state.paused_query = None


def get_auth_headers():
    """Return Authorization headers if logged in."""
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


# =====================================================================
# Sidebar: Authentication
# =====================================================================

with st.sidebar:
    st.markdown("### 🔐 Authentication")

    if st.session_state.access_token:
        st.success(f"Logged in as: **{st.session_state.user_email}**")
        if st.button("Logout", use_container_width=True):
            st.session_state.access_token = None
            st.session_state.user_email = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        auth_tab = st.radio("Action", ["Login", "Sign Up"], horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if auth_tab == "Login":
            if st.button("Login", use_container_width=True):
                if email and password:
                    try:
                        resp = requests.post(
                            f"{API_BASE}/login",
                            json={"email": email, "password": password},
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.access_token = data["access_token"]
                            st.session_state.user_email = email
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error(f"Login failed: {resp.json().get('detail', 'Unknown error')}")
                    except requests.ConnectionError:
                        st.error("Cannot connect to API. Is the server running?")
                else:
                    st.warning("Please enter email and password.")

        else:  # Sign Up
            if st.button("Sign Up", use_container_width=True):
                if email and password:
                    try:
                        resp = requests.post(
                            f"{API_BASE}/signup",
                            json={"email": email, "password": password},
                        )
                        if resp.status_code == 200:
                            st.success("Sign up successful! Please check your email and then login.")
                        else:
                            st.error(f"Sign up failed: {resp.json().get('detail', 'Unknown error')}")
                    except requests.ConnectionError:
                        st.error("Cannot connect to API. Is the server running?")
                else:
                    st.warning("Please enter email and password.")

    st.markdown("---")

    # ── Sidebar: Document Upload ─────────────────────────────────────
    st.markdown("### 📄 Upload Documents")
    if st.session_state.access_token:
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            help="Upload a PDF to index for RAG queries.",
        )
        if uploaded_file and st.button("📤 Ingest Document", use_container_width=True):
            with st.spinner("Parsing, chunking, and embedding..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/documents",
                        headers=get_auth_headers(),
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(result["message"])
                    elif resp.status_code == 429:
                        st.error("⏱️ Rate limit exceeded. Please wait and try again.")
                    else:
                        try:
                            error_detail = resp.json().get("detail", "Unknown error")
                        except Exception:
                            error_detail = resp.text or "Unknown error"
                        st.error(f"Ingestion failed: {error_detail}")
                except requests.ConnectionError:
                    st.error("Cannot connect to API.")
    else:
        st.info("Login to upload documents.")

    st.markdown("---")

    # ── Sidebar: System Status & Document Management ─────────────────
    st.markdown("### 📊 Document Management")
    if st.session_state.access_token:
        try:
            resp = requests.get(
                f"{API_BASE}/documents/count",
                headers=get_auth_headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                st.metric("Total Chunks Indexed", data.get("documents_indexed", 0))
            else:
                st.metric("Total Chunks Indexed", 0)

            # Fetch list of user's uploaded documents
            doc_list_resp = requests.get(
                f"{API_BASE}/documents/list",
                headers=get_auth_headers(),
            )
            if doc_list_resp.status_code == 200:
                docs = doc_list_resp.json().get("documents", [])
                if docs:
                    with st.expander(f"📁 Your Documents ({len(docs)})", expanded=True):
                        for doc_info in docs:
                            fname = doc_info["filename"]
                            c_count = doc_info["chunk_count"]
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{fname}**\n<small>{c_count} chunks</small>", unsafe_allow_html=True)
                            with col2:
                                if st.button("🗑️", key=f"del_{fname}", help=f"Delete '{fname}' from Qdrant"):
                                    del_resp = requests.delete(
                                        f"{API_BASE}/documents/{fname}",
                                        headers=get_auth_headers(),
                                    )
                                    if del_resp.status_code == 200:
                                        st.success(f"Deleted {fname}!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete.")
                else:
                    st.caption("No uploaded documents yet.")

            st.markdown("---")
            if st.button("🧹 Clear Answer Cache", use_container_width=True, help="Purge all your cached answers from Upstash Redis"):
                cache_del_resp = requests.delete(
                    f"{API_BASE}/cache",
                    headers=get_auth_headers(),
                )
                if cache_del_resp.status_code == 200:
                    st.success(cache_del_resp.json().get("message", "Cache cleared!"))
                else:
                    st.error("Failed to clear cache.")
        except requests.ConnectionError:
            st.warning("API not reachable.")
    else:
        st.metric("Total Chunks Indexed", 0)
        st.caption("Login to view your indexed documents.")


# =====================================================================
# Main Content: RAG Chat Interface
# =====================================================================

st.markdown('<p class="main-header">🔍 Enterprise RAG System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Secure, cached, and grounded AI answers from your documents</p>',
    unsafe_allow_html=True,
)

if not st.session_state.access_token:
    st.info("👈 Please login using the sidebar to start querying.")
    st.stop()

# Display chat history
for msg in st.session_state.chat_history:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and "trace" in msg:
            with st.expander("🔎 Execution Trace"):
                trace = msg["trace"]

                # Cache status
                cache_status = "✅ CACHE HIT" if trace.get("cache_hit") else "❌ CACHE MISS"
                cache_class = "cache-hit" if trace.get("cache_hit") else "cache-miss"
                st.markdown(f'<span class="{cache_class}">{cache_status}</span>', unsafe_allow_html=True)

                # Sources
                if trace.get("sources"):
                    st.markdown("**📚 Sources:**")
                    for src in trace["sources"]:
                        st.markdown(
                            f"- `{src.get('filename', '?')}` page {src.get('page_number', '?')} "
                            f"(relevance: {src.get('relevance_score', 0):.3f})"
                        )

                # Security details
                if trace.get("security_details"):
                    st.markdown("**🛡️ Security Pipeline:**")
                    st.json(trace["security_details"])

# If there is a paused thread waiting for HITL approval, show the prompt buttons
if st.session_state.paused_thread_id:
    st.warning("⚠️ **Web Search Required**: No relevant context was found in your uploaded documents. Do you want to run a web search?")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👍 Yes, search the web", use_container_width=True):
            with st.spinner("Resuming with web search..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/query/resume",
                        headers={
                            **get_auth_headers(),
                            "Content-Type": "application/json",
                        },
                        json={
                            "thread_id": st.session_state.paused_thread_id,
                            "approve": True,
                            "query": st.session_state.paused_query
                        }
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": data["answer"],
                            "trace": {
                                "cache_hit": data.get("cache_hit", False),
                                "sources": data.get("sources", []),
                                "security_details": data.get("security_details", {}),
                                "response_time": "Resumed"
                            }
                        })
                        # Clear pause state
                        st.session_state.paused_thread_id = None
                        st.session_state.paused_query = None
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to resume query: {e}")
                    
    with col2:
        if st.button("❌ No, cancel", use_container_width=True):
            with st.spinner("Resuming with fallback..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/query/resume",
                        headers={
                            **get_auth_headers(),
                            "Content-Type": "application/json",
                        },
                        json={
                            "thread_id": st.session_state.paused_thread_id,
                            "approve": False,
                            "query": st.session_state.paused_query
                        }
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": data["answer"],
                        })
                        # Clear pause state
                        st.session_state.paused_thread_id = None
                        st.session_state.paused_query = None
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to resume query: {e}")

# Chat input (hidden if waiting for HITL approval)
if not st.session_state.paused_thread_id:
    if prompt := st.chat_input("Ask a question about your uploaded documents..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call RAG API
        with st.chat_message("assistant"):
            with st.spinner("Processing through RAG pipeline..."):
                start_time = time.time()
                try:
                    resp = requests.post(
                        f"{API_BASE}/query",
                        headers={
                            **get_auth_headers(),
                            "Content-Type": "application/json",
                        },
                        json={"query": prompt},
                        timeout=600,
                    )
                    elapsed = time.time() - start_time

                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        st.markdown(answer)

                        # Build trace
                        trace = {
                            "cache_hit": data.get("cache_hit", False),
                            "sources": data.get("sources", []),
                            "security_details": data.get("security_details", {}),
                            "response_time": f"{elapsed:.2f}s",
                        }

                        with st.expander("🔎 Execution Trace"):
                            cache_status = "✅ CACHE HIT" if trace["cache_hit"] else "❌ CACHE MISS"
                            st.markdown(f"**Cache:** {cache_status}")
                            st.markdown(f"**Response Time:** {trace['response_time']}")

                            if trace["sources"]:
                                st.markdown("**📚 Sources:**")
                                for src in trace["sources"]:
                                    st.markdown(
                                        f"- `{src.get('filename', '?')}` page {src.get('page_number', '?')} "
                                        f"(relevance: {src.get('relevance_score', 0):.3f})"
                                    )

                            if trace["security_details"]:
                                st.markdown("**🛡️ Security:**")
                                st.json(trace["security_details"])

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "trace": trace,
                        })

                    elif resp.status_code == 202:
                        # Paused on HITL interrupt
                        data = resp.json()
                        st.session_state.paused_thread_id = data["thread_id"]
                        st.session_state.paused_query = prompt
                        st.rerun()

                    elif resp.status_code == 400:
                        detail = resp.json().get("detail", {})
                        error_msg = f"🚫 **Query blocked:** {detail.get('blocked_by', 'Unknown reason')}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

                    elif resp.status_code == 429:
                        st.error("⏱️ Rate limit exceeded. Please wait before sending more queries.")

                    elif resp.status_code == 401:
                        st.error("🔐 Session expired. Please login again.")
                        st.session_state.access_token = None
                        st.rerun()

                    else:
                        st.error(f"Error: {resp.status_code} — {resp.text}")

                except requests.ConnectionError:
                    st.error("❌ Cannot connect to the API server. Is it running?")
                except requests.Timeout:
                    st.error("⏱️ Request timed out. The query may be too complex.")
