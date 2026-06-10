import os
import sqlite3
import requests
import streamlit as st
from datetime import datetime

API_URL = os.getenv("API_URL", "http://api-service-1:8000")
DB_PATH = os.getenv("DB_PATH", "/data/history.db")

st.set_page_config(page_title="Hate Speech Detector", layout="wide")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            text      TEXT    NOT NULL,
            label     TEXT    NOT NULL,
            prediction INTEGER NOT NULL,
            timestamp TEXT    NOT NULL,
            model_version TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_results(results, model_version):
    conn = sqlite3.connect(DB_PATH)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in results:
        conn.execute(
            "INSERT INTO history (text, label, prediction, timestamp, model_version) VALUES (?,?,?,?,?)",
            (r["text"], r["label"], r["prediction"], ts, model_version)
        )
    conn.commit()
    conn.close()

def load_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, text, label, timestamp, model_version FROM history ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows

def delete_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()

init_db()

st.markdown("""
<style>
.blurred { filter: blur(5px); transition: filter 0.2s; cursor: pointer; }
.blurred:hover { filter: blur(0px); }
.tag-hate {
    background: #FCEBEB; color: #A32D2D;
    padding: 2px 10px; border-radius: 6px;
    font-size: 13px; font-weight: 500;
}
.tag-neutral {
    background: #EAF3DE; color: #3B6D11;
    padding: 2px 10px; border-radius: 6px;
    font-size: 13px; font-weight: 500;
}
.warning-box {
    background: #FAEEDA; color: #633806;
    border-left: 3px solid #EF9F27;
    padding: 8px 12px; border-radius: 0 6px 6px 0;
    font-size: 13px; margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

st.title("Hate Speech Detector")

tab_detect, tab_history = st.tabs(["Deteksi", "Riwayat"])

with tab_detect:
    if "comments" not in st.session_state:
        st.session_state.comments = [""]
    if "results" not in st.session_state:
        st.session_state.results = []
    if "show_states" not in st.session_state:
        st.session_state.show_states = {}

    col_add, col_clear = st.columns([1, 5])
    with col_add:
        if st.button("+ Tambah komentar"):
            st.session_state.comments.append("")
            st.rerun()

    for i, comment in enumerate(st.session_state.comments):
        cols = st.columns([10, 1])
        with cols[0]:
            st.session_state.comments[i] = st.text_area(
                f"Komentar {i+1}",
                value=comment,
                key=f"comment_{i}",
                height=80,
                label_visibility="collapsed",
                placeholder=f"Komentar {i+1}..."
            )
        with cols[1]:
            if len(st.session_state.comments) > 1:
                if st.button("✕", key=f"del_{i}"):
                    st.session_state.comments.pop(i)
                    st.session_state.results = []
                    st.rerun()

    st.divider()

    if st.button("Detect", type="primary", use_container_width=False):
        texts = [t.strip() for t in st.session_state.comments if t.strip()]
        if not texts:
            st.warning("Masukkan minimal satu komentar.")
        else:
            try:
                with st.spinner("Memproses..."):
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={"texts": texts},
                        timeout=15
                    )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.results = data["predictions"]
                st.session_state.model_version = data.get("model_version", "-")
                st.session_state.show_states = {
                    i: True for i, r in enumerate(data["predictions"]) if r["label"] == "harmful"
                }
                save_results(data["predictions"], data.get("model_version", "-"))
            except requests.exceptions.ConnectionError:
                st.error(f"Tidak dapat terhubung ke API ({API_URL}). Pastikan layanan berjalan.")
            except requests.exceptions.HTTPError as e:
                st.error(f"Error dari API: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

    if st.session_state.results:
        st.markdown(f"<small style='color:gray'>Model v{st.session_state.get('model_version', '-')}</small>", unsafe_allow_html=True)
        st.markdown("### Hasil")
        for i, r in enumerate(st.session_state.results):
            is_harmful = r["label"] == "harmful"
            is_shown = st.session_state.show_states.get(i, not is_harmful)

            with st.container():
                col_text, col_badge, col_toggle = st.columns([6, 2, 2])

                with col_text:
                    if is_harmful and not is_shown:
                        st.markdown(
                            f'<span class="blurred" title="Hover untuk lihat">{r["text"]}</span>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.write(r["text"])

                with col_badge:
                    if is_harmful:
                        st.markdown('<span class="tag-hate">⚠ Hate speech</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="tag-neutral">✓ Aman</span>', unsafe_allow_html=True)

                with col_toggle:
                    if is_harmful:
                        label = "Tampilkan" if not is_shown else "Sembunyikan"
                        if st.button(label, key=f"toggle_{i}"):
                            st.session_state.show_states[i] = not is_shown
                            st.rerun()

                if is_harmful:
                    st.markdown(
                        '<div class="warning-box">Konten ini berpotensi mengandung ujaran kebencian.</div>',
                        unsafe_allow_html=True
                    )

                st.divider()

with tab_history:
    col_h, col_del = st.columns([6, 2])
    with col_h:
        st.markdown("### Riwayat deteksi")
    with col_del:
        if st.button("Hapus semua riwayat", type="secondary"):
            delete_history()
            st.success("Riwayat dihapus.")
            st.rerun()

    rows = load_history()
    if not rows:
        st.info("Belum ada riwayat.")
    else:
        for row in rows:
            rid, text, label, ts, mv = row
            is_harmful = label == "harmful"
            badge = '<span class="tag-hate">Hate speech</span>' if is_harmful else '<span class="tag-neutral">Aman</span>'
            with st.expander(f"{ts}  —  {text[:60]}{'…' if len(text) > 60 else ''}"):
                st.write(text)
                st.markdown(badge, unsafe_allow_html=True)
                st.caption(f"Model v{mv}")