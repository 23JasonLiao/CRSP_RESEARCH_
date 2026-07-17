# Part7 local RAG knowledge base

Place point-in-time documents here. Part7 reads `.txt`, `.md`, `.json`, and `.csv` files recursively; it never sends the API key to the browser.

Recommended evidence types:

- `macro_news`
- `industry_news`
- `fomc_or_rates`
- `fund_report`
- `manager_commentary`
- `methodology`

For JSON, use one object, an array, or `{ "documents": [...] }`. For CSV, use the same columns:

```json
{
  "title": "Document title",
  "date": "2023-12-13",
  "source": "Federal Reserve",
  "evidence_type": "fomc_or_rates",
  "url": "https://source.example/document",
  "text": "Full text or a legally stored excerpt."
}
```

Dates and source links matter: the critic is instructed to distinguish evidence available at the report date from hindsight. Do not place secrets or API keys in this directory.

