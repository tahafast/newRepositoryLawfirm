# app/modules/lawfirmchatbot/services/docgen/manager.py

import re, json, html
from typing import List, Dict, Optional, Tuple
from app.modules.lawfirmchatbot.services.retrieval.vector_search import search_similar_documents
from app.modules.lawfirmchatbot.services.llm import chat_completion

# -------------------- Intents & detection (typo tolerant) --------------------
DOCGEN_VERBS = r"(generate|draft|make|prepare|compose|create)"

# typo-/variant-tolerant tokens (synopsis/affidavit common typos)
TOKEN = lambda *xs: [x.lower() for x in xs]
DOC_TOKEN_MAP = {
    "synopsis": TOKEN("synopsis","synop","synopsys","synopisis","sypnosis","synopses"),
    "affidavit_generic": TOKEN("affidavit","affidavit generic","sworn statement","affidavit of"),
    "counter_affidavit": TOKEN("counter affidavit","counter-affidavit"),
    "affidavit_support": TOKEN("affidavit in support","supporting affidavit"),
    "affidavit_rejoinder": TOKEN("affidavit in rejoinder","rejoinder affidavit"),
    "stay_application": TOKEN("stay application","interim application","order xxxix","o xxxix","o. xxxix"),
    "written_statement": TOKEN("written statement","ws"),
    "plaint": TOKEN("plaint","civil plaint","suit plaint"),
    "writ_petition": TOKEN("writ petition","constitutional petition","w.p.","wp."),
    "rejoinder": TOKEN("rejoinder"),
    "legal_notice": TOKEN("legal notice","notice to"),
    "application_misc": TOKEN("application","misc application","miscellaneous application"),
}

def _contains_any(q: str, words: List[str]) -> bool:
    ql = q.lower()
    return any(w in ql for w in words)

def detect_doc_type(user_query: str) -> Optional[str]:
    """Enter DocGen only when a verb+doc token appears (typo tolerant)."""
    q = (user_query or "").lower().strip()
    if not re.search(DOCGEN_VERBS, q):
        return None
    for dtype, tokens in DOC_TOKEN_MAP.items():
        if _contains_any(q, tokens):
            return dtype
    # common fallback: user typed "draft affidavit"
    if "affidavit" in q:
        return "affidavit_generic"
    return None

# -------------------- Intake Schemas --------------------

BASE_COMMON = [
    {"key":"court","q":"Court & Bench (e.g., Lahore High Court, Principal Seat, Lahore)", "required":True},
    {"key":"case_type_no","q":"Case type & number (e.g., W.P. No. ___ of ____ / Suit No. ___ of ____)", "required":True},
    {"key":"title","q":"Case title (e.g., A v. B & another)", "required":True},
    {"key":"party_role","q":"You represent (e.g., Petitioner / Respondent No.__ / Defendant No.__)", "required":True},
    {"key":"deponent","q":"Deponent/Signatory full name", "required":True},
    {"key":"address","q":"Address / city", "required":True},
    {"key":"authorization","q":"Capacity/authority (e.g., authorized representative/petitioner)", "required":True},
    {"key":"facts","q":"Brief facts/stance (3–6 lines)", "required":True},
    {"key":"place_date","q":"Place & date for verification/signing", "required":True},
    # optional
    {"key":"cma_no","q":"C.M.A / Petition/Application No. (optional)", "required":False},
    {"key":"father","q":"Father's name (optional)", "required":False},
    {"key":"cnic","q":"CNIC (optional)", "required":False},
]

INTAKE_BY_TYPE: Dict[str, List[Dict]] = {
    "affidavit_generic": BASE_COMMON + [
        {"key":"para_wise","q":"Include numbered paragraphs? (yes/no)", "required":True},
        {"key":"exhibits","q":"List exhibits/annexures (comma-separated; optional)", "required":False},
        {"key":"prayer","q":"Any specific prayer/relief (1–2 lines; optional)", "required":False},
    ],
    "counter_affidavit": BASE_COMMON + [
        {"key":"prelim","q":"Preliminary objections (comma-separated: locus standi, laches, concealment, non-maintainability, alternate remedy)", "required":False},
        {"key":"para_wise","q":"Include para-wise reply? (yes/no)", "required":True},
    ],
    "affidavit_support": BASE_COMMON + [
        {"key":"grounds","q":"Grounds (3–6 bullets, comma-separated)", "required":True},
        {"key":"prayer","q":"Prayer sought (1–3 lines)", "required":True},
    ],
    "affidavit_rejoinder": BASE_COMMON + [
        {"key":"counter_points","q":"Points to rebut (comma-separated)", "required":True},
    ],
    "stay_application": BASE_COMMON + [
        {"key":"grounds","q":"Grounds for stay (prima facie/balance of convenience/irreparable loss + specifics, comma-separated)", "required":True},
        {"key":"prayer","q":"Prayer (stay to continue / restrain / suspend action, 1–2 lines)", "required":True},
    ],
    "written_statement": BASE_COMMON + [
        {"key":"prelim","q":"Preliminary objections (comma-separated)", "required":False},
        {"key":"issues","q":"If framed, list issues (comma-separated; optional)", "required":False},
    ],
    "plaint": BASE_COMMON + [
        {"key":"cause_of_action","q":"Cause of action (2–4 lines)", "required":True},
        {"key":"valuation","q":"Valuation for court fee/jurisdiction (optional)", "required":False},
        {"key":"reliefs","q":"Reliefs sought (comma-separated)", "required":True},
    ],
    "writ_petition": BASE_COMMON + [
        {"key":"grounds","q":"Constitutional grounds (e.g., Art. 4/9/10A, arbitrariness, lack of jurisdiction; comma-separated)", "required":True},
        {"key":"prayer","q":"Prayer (writs/orders sought)", "required":True},
    ],
    "rejoinder": BASE_COMMON + [
        {"key":"counter_points","q":"Respondent's points you rebut (comma-separated)", "required":True},
    ],
    "synopsis": BASE_COMMON + [
        {"key":"issues","q":"Key issues (comma-separated)", "required":True},
        {"key":"reliefs","q":"Reliefs sought (comma-separated)", "required":True},
    ],
    "legal_notice": BASE_COMMON + [
        {"key":"recipient","q":"Notice to (name/designation/address)", "required":True},
        {"key":"demands","q":"Specific demands (comma-separated)", "required":True},
        {"key":"deadline","q":"Compliance deadline (e.g., 7 days)", "required":True},
    ],
    "application_misc": BASE_COMMON + [
        {"key":"under_law","q":"Application under (Order/Rule/Section)", "required":True},
        {"key":"grounds","q":"Grounds (comma-separated)", "required":True},
        {"key":"prayer","q":"Prayer (1–2 lines)", "required":True},
    ],
}

def get_intake_schema(doc_type: str) -> List[Dict]:
    return INTAKE_BY_TYPE.get(doc_type, BASE_COMMON)

# -------------------- Cleaners to prevent grey/inline code highlighting -----
def _sanitize_html(s: str) -> str:
    if not s: return ""
    # strip backticks and <code> wrappers the model might emit
    s = s.replace("```", "").replace("`", "")
    s = re.sub(r"</?code>", "", s, flags=re.I)
    return s

def _ensure_json_or_retry(messages, keys: List[str], max_tokens=1900, temperature=0.35) -> Dict[str,str]:
    """Ask once; if not valid JSON or contains tutorial patterns, retry with stricter instruction."""
    def _ask(extra_sys=""):
        resp = chat_completion(
            messages=[{"role":"system","content":messages[0]["content"] + extra_sys},
                      {"role":"user","content":messages[1]["content"]}],
            is_legal_query=True, max_tokens=max_tokens, temperature=temperature
        )
        return resp

    raw = _ask()
    def _looks_tutorial(txt: str) -> bool:
        t = (txt or "").lower()
        bad = ["how to","key components","drafting an affidavit","structure of the",
               "practical notes","example affidavit structure","prompt engineering"]
        return any(b in t for b in bad)
    try:
        data = json.loads(raw)
        if any(k not in data for k in keys) or _looks_tutorial(raw):
            raise ValueError("not-json-or-tutorial")
        return {k: _sanitize_html(data.get(k,"")) for k in keys}
    except Exception:
        stricter = (
            "\nIMPORTANT: Output STRICT JSON ONLY with the specified keys. "
            "Do not include headings like 'Key Components', 'How to', 'Structure', or any tutorial text. "
            "Return document body only."
        )
        raw2 = _ask(stricter)
        try:
            data2 = json.loads(raw2)
            return {k: _sanitize_html(data2.get(k,"")) for k in keys}
        except Exception:
            # last-resort empty shape
            return {k: "" for k in keys}

def next_questions(doc_type: str, answers: Dict[str, str]) -> List[str]:
    schema = get_intake_schema(doc_type)
    ask = [f["q"] for f in schema if f["required"] and not answers.get(f["key"])]
    return ask[:6]  # ask up to 6 per turn

def merge_kv_answers(ans: Dict[str,str], text: str, doc_type: str):
    """Merge 'key: value' lines; if single unlabeled line, fill next required."""
    if not text: return
    # parse key: value pairs
    for line in text.split("\n"):
        if ":" in line:
            k,v = line.split(":",1)
            k = k.strip().lower().replace(" ", "_")
            ans[k] = v.strip()
    # single-line fallback
    if len(text.split("\n"))==1 and ":" not in text:
        for f in get_intake_schema(doc_type):
            if f["required"] and not ans.get(f["key"]):
                ans[f["key"]] = text.strip()
                break

# -------------------- Retrieve exemplar cues (headings language only) --------------------

async def retrieve_template_cues(doc_type: str, query: str) -> Tuple[str, List[str], List[int]]:
    docs, sources, pages = await search_similar_documents(f"{doc_type} {query}".strip(), k=6)
    cue = "\n\n".join([d.page_content.strip() for d in docs[:4] if d.page_content])[:3500]
    return cue, (sources or []), (pages or [])

# -------------------- Section synthesis (general) --------------------

def _nz(v: Optional[str]) -> str: 
    return (v or "").strip()

async def synthesize_sections(doc_type: str, a: Dict[str,str], cue_text: str) -> Dict[str,str]:
    """
    Produce long-form sections for the requested doc type.
    STRICT RULE: Output JSON ONLY; no how-to text; no meta headings.
    """
    # Determine blueprint parameters
    bp = {
        "affidavit_generic": dict(style="affidavit",min_words=900,max_words=1600, min_paras=10),
        "counter_affidavit": dict(style="affidavit",min_words=1000,max_words=1800, min_paras=10),
        "affidavit_support": dict(style="affidavit",min_words=900,max_words=1500),
        "synopsis": dict(style="synopsis",min_words=600,max_words=1000),
        "affidavit_rejoinder": dict(style="affidavit",min_words=900,max_words=1500),
        "stay_application": dict(style="pleading",min_words=900,max_words=1600),
        "written_statement": dict(style="pleading",min_words=1000,max_words=1800),
        "plaint": dict(style="pleading",min_words=1000,max_words=1800),
        "writ_petition": dict(style="pleading",min_words=1000,max_words=1800),
        "rejoinder": dict(style="pleading",min_words=900,max_words=1500),
        "legal_notice": dict(style="pleading",min_words=600,max_words=1200),
        "application_misc": dict(style="pleading",min_words=900,max_words=1500),
    }.get(doc_type, dict(style="pleading", min_words=900,max_words=1500))
    
    target_words = max(bp["min_words"], min(bp["max_words"], 300 + len(cue_text)//2))
    style_line = {
        "affidavit":"Formal affidavit style; no meta/tutorial headings.",
        "pleading":"Formal pleading style with numbered paras where natural; no meta/tutorial headings.",
        "synopsis":"Concise court synopsis style; crisp headings only as per sections; no tutorials."
    }[bp["style"]]

    json_keys = {
        "affidavit_generic": ["deponent_intro","facts_html","numbered_html","prayer_html","verification_html","exhibits_html"],
        "counter_affidavit": ["deponent_intro","prelim_html","facts_html","parawise_html","prayer_html","verification_html"],
        "affidavit_support": ["deponent_intro","grounds_html","facts_html","prayer_html","verification_html"],
        "affidavit_rejoinder": ["deponent_intro","facts_html","rejoinder_points_html","prayer_html","verification_html"],
        "stay_application": ["intro_html","grounds_html","facts_html","prayer_html","verification_html"],
        "written_statement": ["prelim_html","facts_html","parawise_html","issues_html","prayer_html","verification_html"],
        "plaint": ["parties_html","jurisdiction_html","cause_html","facts_html","reliefs_html","verification_html"],
        "writ_petition": ["intro_html","grounds_html","facts_html","prayer_html","verification_html"],
        "rejoinder": ["intro_html","rejoinder_points_html","facts_html","prayer_html","verification_html"],
        "synopsis": ["issues_html","facts_html","reliefs_html"],
        "legal_notice": ["notice_open_html","facts_html","demands_html","deadline_html","notice_close_html"],
        "application_misc": ["intro_html","grounds_html","facts_html","prayer_html","verification_html"],
    }.get(doc_type, ["deponent_intro","facts_html","verification_html"])

    system = (
        "You draft Pakistan court documents. When asked to generate, NEVER explain how to draft. "
        f"Use {style_line} Output STRICT JSON only with the requested keys. "
        "Avoid words/sections like 'Key Components', 'How to', 'Drafting', 'Structure of the'. "
        "No backticks or <code> tags."
    )
    
    user = f"""
Exemplar cues (phrasing/heading style only; do not copy names/facts):
---
{cue_text[:3000]}
---

Doc type: {doc_type}
Target length: ~{target_words} words; minimum numbered paras: {bp.get('min_paras', 0)}
Fields:
Court: {a.get('court')} | Case/No: {a.get('case_type_no')} | Title: {a.get('title')} | Party: {a.get('party_role')}
Deponent: {a.get('deponent')} s/o {a.get('father','')} | CNIC: {a.get('cnic','')} | Address: {a.get('address')}
Authority: {a.get('authorization')} | Place/Date: {a.get('place_date')}
Facts: {a.get('facts','')}
Prelim: {a.get('prelim','')} | Para-wise: {a.get('para_wise','')} | Grounds: {a.get('grounds','')}
Issues: {a.get('issues','')} | Reliefs: {a.get('reliefs','')} | Prayer: {a.get('prayer','')} | Exhibits: {a.get('exhibits','')}

JSON keys to return: {", ".join(json_keys)}
Semantics:
- If a '*_html' key exists, produce valid HTML (<p>, <ol>, <ul>) with no inline styles.
- For numbered/para-wise keys, produce at least {bp.get('min_paras', 8)} concise numbered paras (<p>Para 1. ...</p>).
"""
    msgs = [{"role":"system","content":system},{"role":"user","content":user}]
    data = _ensure_json_or_retry(msgs, json_keys)
    return data

# -------------------- HTML rendering (caption + dynamic body) --------------------

def _caption(a: Dict[str,str]) -> Dict[str,str]:
    title = _nz(a.get("title")) or "[TITLE]"
    left = title.split(" v. ")[0] if " v. " in title else title
    right = title.split(" v. ")[1] if " v. " in title else "[Respondents/Defendants]"
    return {
        "court": _nz(a.get("court")) or "[COURT & BENCH]",
        "case_no": _nz(a.get("case_type_no")) or "[CASE TYPE & NO.]",
        "cma_no": _nz(a.get("cma_no","")),
        "left": left, "right": right
    }

def render_html(doc_type: str, a: Dict[str,str], s: Dict[str,str], references: Optional[str]) -> str:
    # Sanitize all section content before rendering
    for k in list(s.keys()):
        s[k] = _sanitize_html(s.get(k,""))
    
    cap = _caption(a)
    
    heading_map = {
        "affidavit_generic": "AFFIDAVIT",
        "counter_affidavit": "COUNTER-AFFIDAVIT",
        "affidavit_support": "AFFIDAVIT IN SUPPORT",
        "affidavit_rejoinder": "AFFIDAVIT IN REJOINDER",
        "stay_application": "APPLICATION FOR STAY",
        "written_statement": "WRITTEN STATEMENT",
        "plaint": "PLAINT",
        "writ_petition": "CONSTITUTIONAL PETITION",
        "rejoinder": "REJOINDER",
        "synopsis": "SYNOPSIS",
        "legal_notice": "LEGAL NOTICE",
        "application_misc": "APPLICATION",
    }
    heading = heading_map.get(doc_type, "DOCUMENT")
    
    deponent = _nz(a.get("deponent")) or "[Deponent]"
    father   = _nz(a.get("father",""))
    cnic     = _nz(a.get("cnic",""))
    addr     = _nz(a.get("address")) or "[Address]"
    auth     = _nz(a.get("authorization")) or "[Authorized capacity]"
    place_date = _nz(a.get("place_date")) or "[City, Date]"
    ref_html = f"<p class='refs'><strong>References used for format:</strong> {references}</p>" if references else ""

    body_parts = []
    if s.get("deponent_intro"):
        body_parts.append(f"<p>{s['deponent_intro']}</p>")
    if s.get("intro_html"):
        body_parts.append(f"{s['intro_html']}")
    if s.get("prelim_html"):
        body_parts.append(f"<h3>Preliminary Legal Objections</h3>{s['prelim_html']}")
    if s.get("facts_html"):
        body_parts.append(f"<h3>Factual Averments</h3>{s['facts_html']}")
    if s.get("parawise_html"):
        body_parts.append(f"<h3>Para-wise Reply</h3>{s['parawise_html']}")
    if s.get("numbered_html"):
        body_parts.append(f"{s['numbered_html']}")
    if s.get("grounds_html"):
        body_parts.append(f"<h3>Grounds</h3>{s['grounds_html']}")
    if s.get("issues_html"):
        body_parts.append(f"<h3>Issues</h3>{s['issues_html']}")
    if s.get("parties_html"):
        body_parts.append(f"<h3>Parties</h3>{s['parties_html']}")
    if s.get("jurisdiction_html"):
        body_parts.append(f"<h3>Jurisdiction</h3>{s['jurisdiction_html']}")
    if s.get("cause_html"):
        body_parts.append(f"<h3>Cause of Action</h3>{s['cause_html']}")
    if s.get("reliefs_html"):
        body_parts.append(f"<h3>Reliefs</h3>{s['reliefs_html']}")
    if s.get("rejoinder_points_html"):
        body_parts.append(f"<h3>Response in Rejoinder</h3>{s['rejoinder_points_html']}")
    if s.get("prayer_html"):
        body_parts.append(f"<h3>Prayer</h3>{s['prayer_html']}")
    if s.get("exhibits_html"):
        body_parts.append(f"<h3>Exhibits</h3>{s['exhibits_html']}")
    if s.get("notice_open_html"):
        body_parts.append(f"{s['notice_open_html']}")
    if s.get("demands_html"):
        body_parts.append(f"<h3>Demands</h3>{s['demands_html']}")
    if s.get("deadline_html"):
        body_parts.append(f"{s['deadline_html']}")
    if s.get("notice_close_html"):
        body_parts.append(f"{s['notice_close_html']}")
    if s.get("verification_html"):
        body_parts.append(f"<h3>Verification</h3>{s['verification_html']}")
    else:
        body_parts.append(f"<h3>Verification</h3><p>Verified at {place_date} that the contents hereof are true and correct to the best of my knowledge and belief.</p>")

    # Identity block for affidavits/pleadings
    identity_block = ""
    if doc_type not in ["synopsis", "legal_notice"]:
        identity_block = f"<p>I, <strong>{deponent}</strong>{f', s/o {father}' if father else ''}{f', CNIC {cnic}' if cnic else ''}, resident of {addr}, being the {auth}, do hereby state as under:</p>"

    html = f"""
<!doctype html><html><head><meta charset="utf-8"/><title>{heading}</title>
<style>
body{{font-family:'Times New Roman',serif;color:#111}}
.page{{width:8.27in;margin:0.6in auto}}
.center{{text-align:center}}
h1,h2,h3{{margin:0.22in 0 0.12in}}
h1{{font-size:20pt;text-transform:uppercase;letter-spacing:.5px}}
h2{{font-size:16pt}} h3{{font-size:13pt}}
p{{margin:0 0 .14in;line-height:1.35;text-align:justify}}
.caption p{{text-align:center}} .dots{{letter-spacing:2px}}
.small{{font-size:10pt}} .hr{{border-top:1px solid #444;margin:.2in 0}}
.refs{{margin-top:.3in;font-size:10pt}} .sig{{margin-top:.5in}}
</style></head><body><div class="page">

<div class="caption center">
  <h1>IN THE {cap['court']}</h1>
  <p class="small">{cap['case_no']}</p>
  {f"<p class='small'>{cap['cma_no']}</p>" if cap['cma_no'] else ""}
  <p class="dots">{cap['left']} .............................................. Plaintiff/Petitioner</p>
  <p>Versus</p>
  <p class="dots">{cap['right']} ............................................. Respondent(s)/Defendant(s)</p>
</div>

<h2 class="center">{heading}</h2>
{identity_block}

{''.join(body_parts)}

<div class="hr"></div>
<div class="sig"><p class="center"><strong>______________________________</strong><br/>DEPONENT: {deponent}</p></div>
{ref_html}
</div></body></html>
""".strip()
    return html
