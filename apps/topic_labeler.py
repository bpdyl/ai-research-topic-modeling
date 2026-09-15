"""Run: python -m streamlit run apps/topic_labeler.py --server.address 127.0.0.1"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from nrtm.labeling.workflow import load_evidence, new_run, generate_pending, record_review, save_run, export_labels

st.set_page_config(page_title="Nepal AI | Topic labelling", page_icon="🔎", layout="wide")
st.title("From topic words to research themes")
st.caption("Nepal-affiliated AI research · Frozen 2015–2025 corpus · Task 1 labelling workspace")

try:
    entries = load_evidence(ROOT)
except (ValueError, OSError, KeyError) as exc:
    st.error(f"Could not load the frozen topic evidence: {exc}")
    st.stop()

left, middle, right = st.columns(3)
left.metric("Discovered topics", len(entries))
middle.metric("Model configurations", len({e["model_name"] for e in entries}))
right.metric("Existing audited labels", sum(bool(e["archived_audited_label"]) for e in entries))
st.info("All 39 topics already have archived LLM-assisted labels. This app generates new API-backed drafts and records your review separately.")

with st.sidebar:
    st.header("Generation settings")
    provider = st.selectbox("Provider", ["openai", "anthropic", "gemini"], format_func=lambda p: {"openai": "OpenAI", "anthropic": "Claude (Anthropic)", "gemini": "Gemini (Google)"}[p])
    env_prefix = {"openai": "OPENAI", "anthropic": "ANTHROPIC", "gemini": "GEMINI"}[provider]
    model = st.text_input("Model ID", value=os.getenv(env_prefix + "_MODEL", ""), key="model_" + provider,
                          help="Enter the exact API model identifier available to your account.")
    supplied_key = st.text_input("API key", type="password", key="key_" + provider,
                                 help=f"Or set {env_prefix}_API_KEY before starting the app. Keys are not saved in run files.")
    key = supplied_key or os.getenv(env_prefix + "_API_KEY", "")
    if provider == "gemini":
        key = key or os.getenv("GOOGLE_API_KEY", "")
    st.caption("Credential available" if key else "Enter a key to enable generation. Browsing needs no key.")
    max_tokens = st.number_input("Maximum output tokens per topic", min_value=512, max_value=16384, value=2048, step=512)
    st.caption("One request per topic. No automatic retries. Calls use your provider account and may incur charges.")

browse, generate, review = st.tabs(["Explore evidence", "Generate labels", "Review & export"])

with browse:
    model_name = st.selectbox("Topic model", list(dict.fromkeys(e["model_name"] for e in entries)))
    subset = [e for e in entries if e["model_name"] == model_name]
    topic = st.selectbox("Topic", subset, format_func=lambda e: f"T{e['topic_id']} · {e['archived_audited_label']}")
    st.subheader(topic["archived_audited_label"])
    st.write(" · ".join(topic["top_words"]))
    c1, c2 = st.columns(2)
    c1.caption("Original assistant-session draft")
    c1.write(topic["archived_draft"])
    c2.caption("Archived automated audit label")
    c2.write(topic["archived_audited_label"])
    st.markdown("**Representative papers**")
    if not topic["representative_documents"]:
        st.warning("This archived random-search run contains top words but no document/topic matrix. Its labels use words only; representative papers are unavailable.")
    for doc in topic["representative_documents"]:
        st.write(f"{doc['year']} · {doc['title']}")
        st.caption(f"{doc['doc_id']} · topic weight {doc['weight']:.4f}")
    with st.expander("Exact generation prompt and source provenance"):
        st.code(topic["prompt"], language=None)
        st.json({k: topic[k] for k in ["source_run", "source_hashes", "document_alignment", "prompt_sha256"]})

with generate:
    selected_models = st.multiselect("Models to label", list(dict.fromkeys(e["model_name"] for e in entries)), default=list(dict.fromkeys(e["model_name"] for e in entries)))
    eligible = [e for e in entries if e["model_name"] in selected_models]
    selected_keys = st.multiselect("Topics to label", [e["key"] for e in eligible], default=[e["key"] for e in eligible],
                                  format_func=lambda k: next(f"{e['model_name']} · T{e['topic_id']}" for e in entries if e["key"] == k))
    chosen = [e for e in eligible if e["key"] in selected_keys]
    st.write(f"{len(chosen)} requests · top words and up to four public paper titles per topic")
    st.caption("New results are saved in results/labeling/. The manuscript's archived labels and ratings remain tied to their original experiment.")
    if st.button("Generate selected labels", type="primary", disabled=not (key and model.strip() and chosen)):
        path, run = new_run(ROOT, chosen, provider, model.strip(), int(max_tokens))
        bar = st.progress(0.0)
        with st.spinner("Generating and checkpointing topic labels…"):
            generate_pending(path, run, key, progress=lambda n, total: bar.progress(n / total))
        st.session_state["active_label_run"] = str(path)
        success = sum(i["status"] == "generated" for i in run["items"])
        st.success(f"Saved {success}/{len(run['items'])} labels in run {run['run_id']}.")
        for item in run["items"]:
            if item["status"] == "failed":
                st.error(item["error"])
        st.caption("Open Review & export to inspect this run. Failed or pending topics can be resumed there.")

with review:
    paths = sorted((ROOT / "results/labeling").glob("*/labels.json"), reverse=True)
    if not paths:
        st.write("No API generation runs yet. Explore the archived labels or generate a new draft set.")
    else:
        active = st.session_state.get("active_label_run")
        options = [str(p) for p in paths]
        selected_path = st.selectbox("Saved run", options, index=options.index(active) if active in options else 0, format_func=lambda p: Path(p).parent.name)
        path = Path(selected_path)
        run = json.loads(path.read_text(encoding="utf-8"))
        st.caption(f"{run['provider']} · {run['requested_model']} · {run['created_at']}")
        pending = [i for i in run["items"] if i["status"] != "generated"]
        st.write(f"{len(run['items']) - len(pending)}/{len(run['items'])} generated · "
                 f"{sum(bool(i.get('review')) for i in run['items'])} reviewed by a person")
        if pending:
            if st.button("Resume unfinished topics", disabled=not key or provider != run["provider"]):
                generate_pending(path, run, key)
                st.rerun()
            for item in pending:
                if item.get("error"):
                    st.warning(item["error"])
        generated = [i for i in run["items"] if i["status"] == "generated"]
        if generated:
            index = st.selectbox("Label to review", list(range(len(generated))),
                                 format_func=lambda j: f"{generated[j]['evidence']['model_name']} · T{generated[j]['evidence']['topic_id']}")
            item = generated[index]
            st.subheader(item["draft"]["label"])
            st.write(item["draft"]["rationale"])
            st.caption("Model assessment: " + item["draft"]["coherence"])
            st.write(" · ".join(item["evidence"]["top_words"]))
            for d in item["evidence"]["representative_documents"]:
                st.write(f"{d['year']} · {d['title']}")
            previous = item.get("review") or {}
            with st.form("review_" + run["run_id"] + "_" + str(index)):
                decision = st.selectbox("Decision", ["accept", "revise", "reject"])
                final = st.text_input("Final label (used when revising)", value=previous.get("final_label") or item["draft"]["label"])
                reviewer_name = st.text_input("Your name or initials", value=previous.get("reviewer", ""))
                note = st.text_area("Evidence and review note", value=previous.get("note", ""))
                if st.form_submit_button("Save my review"):
                    try:
                        record_review(item, decision, final, reviewer_name, note)
                        save_run(path, run)
                        st.success("Your review was saved, with the original draft and review history retained.")
                    except ValueError as exc:
                        st.error(str(exc))
            with st.expander("Request, response and review history"):
                st.json(item)
        st.download_button("Download complete audit trail", json.dumps(run, indent=2, ensure_ascii=False), file_name=run["run_id"] + "_audit.json", mime="application/json")
        st.download_button("Download human-reviewed labels", json.dumps(export_labels(run), indent=2, ensure_ascii=False), file_name=run["run_id"] + "_reviewed.json", mime="application/json")
        st.download_button("Download API drafts (unreviewed)", json.dumps(export_labels(run, False), indent=2, ensure_ascii=False), file_name=run["run_id"] + "_drafts.json", mime="application/json")
        st.caption("Exports state their coverage. Rejected or unreviewed labels are excluded from the reviewed export; a partial export is not a complete labelled topic set.")
