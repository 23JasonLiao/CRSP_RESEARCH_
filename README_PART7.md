# Part7 Evidence-Grounded LLM Critic

Part7 does not replace or recompute Part6 predictions. It retrieves saved Part1–Part6 visual/model evidence, retrieves local point-in-time documents, and asks an OpenAI Responses API model to act as a critic that seeks both supporting and contradicting evidence.

Part7 accepts up to eight Part6 events in one critic request. The frontend defaults to the events selected in the Part6 SHAP comparison, while still allowing an independent multi-selection.

## Run without an API key

Start the existing FastAPI app and press **Run Evidence Critic**. The endpoint returns `status: preview`, including:

- selected Part6 event and manager;
- stable evidence IDs (`E001`, `E002`, ...);
- local RAG results;
- the complete instructions and input prompt that would be sent later.

No model request is made in preview mode.

## Add research documents

Put `.txt`, `.md`, `.json`, or `.csv` documents in `data/part7_knowledge/`. Include source, publication date, evidence type, URL, and text. See `data/part7_knowledge/README.md` for the schema.

## Enable the API later

Install dependencies and set secrets in the backend process, never in `static/app.js` or browser storage:

```powershell
pip install -r requirements.txt
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-5.6"
python main.py
```

The model name is configurable. `gpt-5-o` is not an API model ID; use an ID listed in the current OpenAI model catalog.

With **同期 web search** enabled, the Responses API is allowed to find macro/industry news, FOMC and rate texts, fund reports, and manager commentary near the selected report date. The prompt explicitly requires counterevidence, point-in-time discipline, structural-break checks, data-limit warnings, and SHAP non-causality warnings.

Outputs are saved to `outputs/part7/part7_critic_latest.json` plus a timestamped audit copy.
