
import io
import re
from collections import Counter

import pandas as pd
import pdfplumber
import plotly.express as px
import streamlit as st


# =========================
# BRAND SETTINGS
# =========================
BRAND_NAME = "ARTHA AI"
TAGLINE = "Annual Report Analyzer • Offline • Demo-friendly"
LOGO_PATH = "logo.png"          # optional: keep a logo.png next to app.py
FONT_NAME = "Open Sans"         # YOU ASKED: Open Sans
ACCENT = "#7C5CFF"              # violet accent


# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title=BRAND_NAME, layout="wide")


# =========================
# GLOBAL STYLES (Open Sans applies to EVERYTHING)
# =========================
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
.stApp {
  background: radial-gradient(circle at 20% 10%, #1b2a4a 0%, #0b0f19 45%, #070a12 100%);
}
.stApp, .stApp * {
  font-family: 'Open Sans', sans-serif !important;
}

h1,h2,h3,p,span,label {
  color: #E9EEFC !important;
}
.small-muted {
  color: rgba(233,238,252,0.70) !important;
}

.badge {
  display:inline-block;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(124,92,255,0.16);
  border: 1px solid rgba(124,92,255,0.35);
  color: #E9EEFC;
  font-size: 12px;
}

.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 22px;
  padding: 18px 20px;
  box-shadow: 0 16px 40px rgba(0,0,0,0.28);
  margin-bottom: 26px;
}

hr {
  border: 0;
  height: 1px;
  background: rgba(255,255,255,0.10);
}

input, textarea {
  border-radius: 14px !important;
}
</style>
""",
    unsafe_allow_html=True
)

# =========================
# PDF HELPERS
# =========================
@st.cache_data(show_spinner=False)
def extract_pages_from_pdf(file_bytes: bytes) -> list[dict]:
    """Return list of {"page": int, "text": str}."""
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = page.extract_text() or ""
            pages.append({"page": i, "text": t})
    return pages


def join_pages(pages: list[dict]) -> str:
    return "\n\n".join(p["text"] for p in pages if p["text"])


# =========================
# KEYWORD EXPLORER (Finder-style counting + dropdown snippets)
# =========================
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def build_pattern(query: str, mode: str):
    """
    mode:
      - "substring" => exact findings (water matches deepwater)
      - "whole_word" => strict (water != deepwater)
    """
    q = query.strip()
    if not q:
        return None

    if mode == "whole_word" and re.fullmatch(r"[A-Za-z]+", q):
        return re.compile(rf"\b{re.escape(q)}\b", re.IGNORECASE)

    # default substring / phrase
    return re.compile(re.escape(q), re.IGNORECASE)


def count_occurrences_and_snippets(
    pages: list[dict],
    query: str,
    mode: str = "substring",
    window: int = 140,
    max_snippets: int = 200,
    max_snippets_per_page: int = 8,
):
    """
    Returns:
      total_occurrences (Finder-style)
      snippets_for_dropdown (list of {page, snippet})
    """
    pattern = build_pattern(query, mode)
    if pattern is None:
        return 0, []

    total = 0
    snippets = []

    for p in pages:
        t = normalize_text(p["text"])
        if not t:
            continue

        matches = list(pattern.finditer(t))
        total += len(matches)

        # make dropdown usable: keep a few snippets per page
        taken = 0
        for m in matches:
            start = max(0, m.start() - window)
            end = min(len(t), m.end() + window)
            snippet = t[start:end].strip()
            snippets.append({"page": p["page"], "snippet": snippet})
            taken += 1
            if taken >= max_snippets_per_page:
                break
            if len(snippets) >= max_snippets:
                return total, snippets

    return total, snippets


def top_word_frequencies(text: str, min_len: int = 4, top_n: int = 25) -> pd.DataFrame:
    words = re.findall(r"[A-Za-z']+", (text or "").lower())
    words = [w for w in words if len(w) >= min_len]
    c = Counter(words)
    return pd.DataFrame(c.most_common(top_n), columns=["word", "count"])


# =========================
# FINANCIAL EXTRACTION (Best-effort ₹ Cr)
# =========================
def detect_unit_hint(lines: list[str]) -> str:
    sample = "\n".join(lines[:2500]).lower()
    if "₹ in crore" in sample or "rs. in crore" in sample or "in crore" in sample:
        return "crore"
    if "₹ in lakh" in sample or "rs. in lakh" in sample or "in lakh" in sample or "lakhs" in sample:
        return "lakh"
    return "unknown"


def parse_amount_to_crore(raw: str, global_unit_hint: str) -> float | None:
    s = raw.strip().lower()
    s = s.replace("₹", "").replace("rs.", "").replace("rs", "").replace("inr", "").strip()
    s = s.replace("crores", "crore").replace("cr.", "cr")

    m = re.search(r"(\d[\d,]*\.?\d*)", s)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))

    if "crore" in s or re.search(r"\bcr\b", s):
        return num
    if "lakh" in s or "lakhs" in s:
        return num / 100.0

    if global_unit_hint == "crore":
        return num

    if num > 1e7:  # rupees -> crore guess
        return num / 1e7

    return num


def parse_year_tokens(line: str) -> list[int]:
    years = set()
    # 2024-25 => treat as 2025
    for m in re.findall(r"(20\d{2})\s*[-–]\s*(\d{2})", line):
        start = int(m[0])
        end2 = int(m[1])
        end = (start // 100) * 100 + end2
        years.add(end)
    for y in re.findall(r"\b(20\d{2})\b", line):
        years.add(int(y))
    return sorted(years)


def extract_metric_candidates(text: str, keywords: list[str], min_value_crore: float = 500.0) -> tuple[list[dict], str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    unit_hint = detect_unit_hint(lines)
    key_pat = re.compile("|".join([re.escape(k) for k in keywords]), re.IGNORECASE)

    candidates = []
    for ln in lines:
        if not key_pat.search(ln):
            continue

        years = parse_year_tokens(ln)
        if not years:
            continue

        amounts = re.findall(r"(₹?\s?\d[\d,]*\.?\d*\s?(?:cr|crore|lakh|lakhs)?)", ln, flags=re.IGNORECASE)
        if not amounts:
            continue

        for i, y in enumerate(years[: len(amounts)]):
            val = parse_amount_to_crore(amounts[i], unit_hint)
            if val is None:
                continue
            if val < min_value_crore:
                continue
            candidates.append({"Year": y, "ValueCr": float(val), "SourceLine": ln})

    return candidates, unit_hint


def build_year_series(candidates: list[dict]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    df = pd.DataFrame(candidates)
    out = df.groupby("Year", as_index=False)["ValueCr"].median().sort_values("Year")
    out["Growth_%"] = out["ValueCr"].pct_change() * 100
    return out


# =========================
# OFFLINE ASK SEARCH (paragraph-level)
# =========================
def smart_search(text: str, query: str, top_k: int = 4):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    terms = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", query or "") if len(w) > 2]
    scored = []
    for p in paras:
        pl = p.lower()
        score = sum(pl.count(t) for t in terms)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown(f"### {BRAND_NAME}")
    st.caption("Upload a PDF and explore it.")
    show_audit = st.toggle("Show extraction audit trail", value=False)
    uploaded_files = st.file_uploader("Upload Annual Reports (PDFs)", type=["pdf"], accept_multiple_files=True)
    st.markdown("---")
    match_mode = st.radio(
        "Keyword counting mode",
        ["Finder-like (substring)", "Strict (whole word)"],
        index=0,
    )
    st.caption("Tip: Finder-like counts `water` inside `deepwater` too.")

    mode_value = "substring" if match_mode.startswith("Finder-like") else "whole_word"


# =========================
# HEADER
# =========================
h1, h2 = st.columns([1, 6])
with h1:
    try:
        st.image(LOGO_PATH, width=70)
    except:
        pass

with h2:
    st.markdown(f"<h1 style='margin-bottom:0'>{BRAND_NAME}</h1>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>{TAGLINE}</span>", unsafe_allow_html=True)
    st.markdown("<p class='small-muted'>Upload → Extract → Visualize → Keyword Explorer → Ask</p>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)


# =========================
# LANDING
# =========================
uploaded_file = st.file_uploader(
    "Upload Annual Report (PDF)",
    type=["pdf"]
)
if not uploaded_file:
    st.markdown(
        """
        <div class='card'>
          <h2 style="margin:0;">📄 Upload a PDF to begin</h2>
          <p class='small-muted' style="margin-top:8px;">
            Intrepreting the annual report by the words used.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='card'>
          <h2 style="margin:0;">🔎 Keyword Explorer</h2>
          <p class='small-muted' style="margin-top:8px;">
            Frequency that matches Finder-style counting + a dropdown to jump to context snippets (with page).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# =========================
# READ PDF
# =========================
pdf_bytes = uploaded_file.getvalue()
with st.spinner("Reading PDF text…"):
    pages = extract_pages_from_pdf(pdf_bytes)
    text = join_pages(pages)

st.success("PDF loaded ✅")


# =========================
# MAIN LAYOUT
# =========================
left, right = st.columns([1.35, 1])


# =========================
# LEFT: FINANCIAL OVERVIEW
# =========================
with left:
    st.markdown(
        """
        <div class='card'>
          <h2 style="margin:0;">📈 Financial Overview</h2>
          <p class='small-muted' style="margin-top:8px;">
            Best-effort extraction from PDF text. Use audit trail to verify the lines used.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    revenue_keywords = ["revenue", "total income", "total revenue", "income from operations", "turnover"]
    rev_candidates, unit_hint = extract_metric_candidates(text, revenue_keywords, min_value_crore=500.0)
    rev_series = build_year_series(rev_candidates)

    if rev_series.empty:
        st.warning(
            "Revenue could not be reliably extracted from the PDF text layer (tables can be messy). "
            "Turn on audit trail to see what lines were captured."
        )
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Latest Revenue (₹ Cr)", f"{rev_series['ValueCr'].iloc[-1]:,.2f}")
        yoy = rev_series["Growth_%"].iloc[-1]
        k2.metric("YoY Growth", "—" if pd.isna(yoy) else f"{yoy:.2f}%")
        k3.metric("Unit Hint", unit_hint)

        fig = px.line(
            rev_series,
            x="Year",
            y="ValueCr",
            markers=True,
            labels={"ValueCr": "Revenue (₹ Cr)"},
        )
        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            dragmode="pan",
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_traces(marker=dict(size=9))
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

        fig2 = px.bar(
            rev_series,
            x="Year",
            y="Growth_%",
            labels={"Growth_%": "YoY Growth (%)"},
        )
        fig2.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"scrollZoom": True})

    if show_audit:
        st.markdown(
            """
            <div class='card'>
              <h3 style="margin:0;">🧾 Extraction Audit Trail</h3>
              <p class='small-muted' style="margin-top:8px;">
                Raw captured lines used in revenue extraction.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        audit_df = pd.DataFrame(rev_candidates).head(60)
        if audit_df.empty:
            st.info("No audit lines captured.")
        else:
            st.dataframe(audit_df, use_container_width=True)


# =========================
# RIGHT: KEYWORD EXPLORER + ASK
# =========================
with right:

    # --- Keyword Explorer
    st.markdown(
        """
        <div class='card'>
          <h2 style="margin:0;">🔎 Keyword Explorer</h2>
          <p class='small-muted' style="margin-top:8px;">
            Frequency (Finder-style) + dropdown to jump to context snippets (with page number).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Multiple keyword input
    kw_input = st.text_input("Enter keywords separated by commas")
    keywords = [k.strip() for k in kw_input.split(",") if k.strip()]

    # Single keyword search
    kw = st.text_input("Keyword / phrase (e.g., cash flow, risk, debt)", key="kw_query")

    if kw:

        total_occ, snippets = count_occurrences_and_snippets(
            pages,
            kw,
            mode=mode_value,
            window=140,
            max_snippets=200,
            max_snippets_per_page=8,
        )

        st.write(f"**Occurrences (like Finder):** {total_occ}")
        st.write(f"**Dropdown items:** {len(snippets)}")

        if snippets:

            options = [f"Page {h['page']} — {h['snippet'][:120]}..." for h in snippets]

            picked = st.selectbox(
                "Dropdown: choose a match",
                options,
                key="kw_dropdown"
            )

            chosen = snippets[options.index(picked)]

            st.markdown("**Context snippet:**")
            st.write(chosen["snippet"])
            st.caption(f"Page: {chosen['page']}")

        else:
            st.info("No matches found. Try a broader word or switch to Finder-like mode.")

        # Word frequency chart
        with st.expander("Top words (frequency)"):

            freq_df = top_word_frequencies(text, min_len=4, top_n=25)

            st.dataframe(freq_df, use_container_width=True)

            figw = px.bar(freq_df, x="word", y="count")

            figw.update_layout(
                template="plotly_dark",
                height=280,
                margin=dict(l=10, r=10, t=10, b=10)
            )

            st.plotly_chart(figw, use_container_width=True, config={"scrollZoom": True})    # --- Ask (Offline search)
    st.markdown(
        """
        <div class='card'>
          <h2 style="margin:0;">💬 Ask ARTHA AI (Offline Search)</h2>
          <p class='small-muted' style="margin-top:8px;">
            Finds relevant paragraphs quickly. (Shows “Match • score …”)
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q = st.text_input("Try: risk factors, borrowings, capex, cash flow, litigation", key="ask_query")

    if q:
        hits = smart_search(text, q, top_k=4)
        if not hits:
            st.info("No strong matches found. Try different keywords.")
        else:
            for i, (score, para) in enumerate(hits, 1):
                st.markdown(f"**Match {i} • score {score}**")
                st.write(para)
                st.markdown("<hr/>", unsafe_allow_html=True)
