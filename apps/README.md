# Topic labelling app

The proposal's automated labelling step already has saved outputs for all 39
topics. `data/processed/llm_labels_draft.json` records that those drafts were
produced in an assistant session, without API calls. `final_topic_labels.json`
contains the subsequent automated audit labels. Neither is a human validation
study. The older script `08_label_topics.py` assembles and merges these records;
it does not invoke a provider.

This Streamlit app adds a runnable, provider-independent generation and review
workflow. It uses each provider's native HTTP API, through one `complete()`
interface. No provider SDK, model retraining or notebook execution is required.

## Start

From the repository root:

```powershell
python -m pip install -r apps/requirements.txt
python -m streamlit run apps/topic_labeler.py --global.developmentMode false --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open the local URL printed by Streamlit. Browse evidence without a key. To
generate, select OpenAI, Claude or Gemini, enter the exact model identifier
available to your account, and enter an API key in the password field. You can
instead supply `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` in the
server environment (Google's `GOOGLE_API_KEY` is also accepted). Optional model
defaults are `OPENAI_MODEL`, `ANTHROPIC_MODEL` and `GEMINI_MODEL`.

Use a local environment or the app field for credentials; do not place them in
the repository. The app binds to localhost and does not write keys into its
audit files. A ChatGPT/Claude/Gemini consumer subscription is not itself a key.

## Workflow

1. Explore a model and topic. Compare its historical draft and audited label,
   top words and highest-weight paper titles. Expand the exact prompt and hashes.
2. Select any or all topics and generate. Each topic makes one request; charges
   depend on your provider account. The app does not automatically retry requests.
3. Inspect the saved run in Review & export. Failed/pending topics can resume
   with the original run's provider, model and token limit; successes are skipped.
4. Accept, revise or reject a draft with your name and an evidence note. This
   explicitly records a person reviewing a label, separately from generation.
5. Download the audit JSON, raw draft mapping or reviewed mapping. Exports state
   expected/exported counts. Rejecting a label leaves it out of the reviewed
   mapping; an incomplete mapping is not a completed topic set.

Runs are checkpointed atomically in `results/labeling/<run-id>/labels.json`.
Each contains the evidence, source run/hash, exact prompt, prompt version,
requested/resolved provider model, completion text, response ID, usage and
timestamp. Review history preserves previous decisions. API responses are
validated for required fields, allowed coherence values and supplied document
IDs. This verifies format and evidence references, not semantic correctness.

Only top words and up to four public paper titles/IDs/years/topic weights are
sent. Full abstracts, author details and corpus files are not sent. The eight
random-search topics have words only: the archived driver did not save its
document-topic matrix or model. This is disclosed rather than reconstructing a
different model and treating it as the archived one.

The manuscript continues to use its original label/audit experiment. Generating
a new run does not overwrite `final_topic_labels.json`, rating records or figures.
To adopt new labels in a future report, first review the full set, select the
run explicitly and update figures and provenance together. New labels alone do
not constitute new interpretability ratings or a cross-provider comparison.

## Verification and API references

Adapters, error/partial-response handling, resume/review/export behavior and the
Streamlit browsing/credential gate were tested with simulated provider responses.
No live provider completion was run during implementation because no API key was
configured. Actual access and model compatibility must be checked with a small
run using your account. Sampling uses provider defaults (temperature omitted);
identical prompts do not guarantee identical future output.

- [OpenAI Responses API quickstart](https://developers.openai.com/api/docs/quickstart)
- [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create)
- [Gemini generateContent API](https://ai.google.dev/api/generate-content)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)

```powershell
$env:PYTHONPATH = 'src'
python -m pytest tests/test_labeling_workflow.py
```

The existing environment with the optional `labeling` dependency also runs the
app. The small `apps/requirements.txt` avoids installing the training stack for
users who only need the app. Tested app version: Streamlit 1.63.0, Python 3.12.
