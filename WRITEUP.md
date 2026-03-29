# PRISM — Pipeline for Ranked Intelligent Search & Matching
### ML Engineering Challenge — Submission by David

---

Hi, I'm David. This is my solution for the ML Engineering internship challenge.

I'll be upfront: when I first read the problem, my instinct was to just send every
company to an LLM and ask "does this match?". That works. But it's expensive, slow,
and doesn't scale — and honestly, it doesn't show much engineering thinking either.

So I spent time designing something smarter. A system I'd actually be proud to put
in production. I called it **PRISM** — Pipeline for Ranked Intelligent Search & Matching.

The core insight is simple: **not every company deserves the same amount of compute.**
A company that's obviously wrong should be eliminated for free. A company that's
obviously right should be qualified without an expensive LLM call. Only the
genuinely ambiguous cases should get the full treatment.

Here's what I built.

---

## 3.1 System Architecture

### The Pipeline at a Glance
```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 · Query Intelligence Engine                │
│  1 LLM call · structured execution plan             │
└──────────────────────┬──────────────────────────────┘
                       │ filters + expanded terms + signals
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 · Hard Filter Gate                         │
│  Pure Python · 0 API calls · eliminates ~60%        │
└──────────────────────┬──────────────────────────────┘
                       │ ~200 companies remain
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3 · Dual Semantic Ranker                     │
│  BM25 (35%) + Embedding cosine (65%)                │
│  Enriched query with expanded terms                 │
└──────────────────────┬──────────────────────────────┘
                       │ all ranked, top 30 go forward
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4 · Cross-Encoder Reranker                   │
│  Processes (query, company) pairs together          │
│  Much more precise than embeddings alone            │
└──────────────────────┬──────────────────────────────┘
                       │ final_score = 0.4 × semantic + 0.6 × cross_encoder
                       ▼
┌──────────────┬───────┴────────┬───────────────────┐
│ Auto-Qualify │   Gray Zone    │   Auto-Reject      │
│ score≥θ_high │ θ_low to θ_high│   score<θ_low      │
│  no LLM      │  → LLM batch  │   no LLM           │
└──────────────┴───────┬────────┴───────────────────┘
                       │ 15 companies per LLM call
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 6 · Final Merger & Explainer                 │
│  Combines all sources · ranked list + explanations  │
└─────────────────────────────────────────────────────┘
```

### Stage 1 — Query Intelligence Engine

**File:** `pipeline/query_analyst.py`

Before touching a single company record, I make exactly one LLM call to
analyze the user's query. The model produces a structured JSON execution plan
that every downstream stage will use.

Here's a real example. Input query:

> *"Fast-growing fintech companies competing with traditional banks in Europe"*

Output from Stage 1:
```json
{
  "query_type": "hybrid",
  "difficulty": "hard",
  "structured_filters": {
    "continent": "Europe",
    "founded_after": 2010
  },
  "target_industries": ["Financial Services", "Banking", "Fintech"],
  "expanded_terms": [
    "digital banking platforms", "neobank", "challenger bank",
    "mobile payment solutions", "alternative lending"
  ],
  "negative_terms": [
    "traditional bank", "incumbent financial institution",
    "legacy banking"
  ],
  "implicit_signals": [
    "rapid growth", "innovation", "digital-first", "disruptive model"
  ],
  "reasoning": "Query requires identifying fintech disruptors with
                growth indicators and European presence"
}
```

Why this matters: the `expanded_terms` field is gold. A fintech startup might
call itself a "neobank" or "challenger bank" without ever using the word "fintech".
Without expansion, the semantic ranker would miss them. With expansion, they rank
near the top.

The `negative_terms` are equally important. They get passed to the LLM qualifier
later with explicit instructions: *"these types of companies are NOT what we want"*.

I also implemented a **disk cache** for query analyses — once a query is analyzed,
the result is saved to `results/query_analysis_cache.json`. Re-running the pipeline
doesn't waste API calls re-analyzing the same queries.

**Cost: exactly 1 LLM call per query, regardless of dataset size.**

---

### Stage 2 — Hard Filter Gate

**File:** `pipeline/hard_filter.py`

Pure Python. Zero API calls. Runs in milliseconds. Typically eliminates 50-70%
of the dataset before any ML is involved.

The filter checks the following fields using the structured output from Stage 1:

| Filter | Field in data | Example |
|---|---|---|
| Country | `address.country_code` | "ro" for Romania |
| Continent | `address.country_code` (mapped) | Europe = 20+ country codes |
| Min employees | `employee_count` | >= 1000 |
| Max employees | `employee_count` | <= 200 |
| Min revenue | `revenue` | >= 50,000,000 |
| Is public | `is_public` | True |
| Founded after | `year_founded` | >= 2018 |

One important detail: the `address` field in the dataset is not a plain string —
it's a nested dictionary with `country_code`, `town`, `region_name`, and coordinates.
My filter correctly extracts `country_code` and maps it to the country or continent
from the query.

**Important design decision: missing data ≠ rejection.**

Real company datasets are always incomplete. If a company doesn't have a
`revenue` field, I don't eliminate it — I mark it as uncertain and apply a
15% score penalty later. A hard rejection on missing data would silently
throw away valid companies.

**Three-state logic per filter:**
```
Field exists AND matches condition   →  PASS
Field exists AND fails condition     →  REJECT
Field is missing / null              →  UNCERTAIN (continues with penalty flag)
```

---

### Stage 3 — Dual Semantic Ranker

**File:** `pipeline/semantic_ranker.py`

Two complementary techniques run in parallel on every company that survived
the hard filter.

**BM25 (keyword matching)**

I serialize each company into a single text document:
```
"Meridian Logistics GmbH freight forwarding customs brokerage
 warehousing B2B Service Provider automotive manufacturing
 Freight Transportation Arrangement"
```

BM25 then scores this document against the enriched query. It's great at
exact and near-exact term matches, handles term frequency naturally, and
runs in milliseconds on 500 companies.

**Embedding similarity**

I use `all-MiniLM-L6-v2` — a lightweight but capable sentence transformer
that produces 384-dimensional embeddings. Cosine similarity between the
query embedding and each company embedding captures semantic meaning that
BM25 misses.

**The key insight: query enrichment**

I don't embed the raw query. I embed an enriched version:
```
"Fast-growing fintech companies competing with traditional banks in Europe
 digital banking platforms neobank challenger bank mobile payment solutions
 Financial Services Banking Fintech
 rapid growth innovation digital-first disruptive model"
```

This dramatically improves recall. A company describing itself as a
"challenger bank" with "innovative payment solutions" will now score
highly even if it never uses the word "fintech".

**Combined score:**
```
semantic_score = 0.35 × BM25_normalized + 0.65 × embedding_cosine
```

I weight embeddings higher because they handle semantic meaning better
for the complex, interpretation-heavy queries in this challenge.

Companies with missing fields from Stage 2 get a 15% penalty applied here:
```python
score *= 0.85  # if has_missing_fields == True
```

---

### Stage 4 — Cross-Encoder Reranker

**File:** `pipeline/cross_encoder_reranker.py`

This is the most technically interesting part of the system. The top 30
companies from Stage 3 go through a cross-encoder model:
`cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Why cross-encoders are better than bi-encoders (embeddings):**

With embeddings, the query and each company are encoded *separately*, then
compared via cosine similarity. The model never sees them together, so it
can't model their interaction.

A cross-encoder takes the query and the company description *concatenated*
as a single input and produces a relevance score directly. It sees both
at once, which means it can answer questions like:

- "Does this company *supply* packaging, or does it *use* packaging?"
- "Is this company a fintech, or does it just *sell software to* fintechs?"
- "Is this pharmaceutical company *in* Switzerland, or just *operating* there?"

These distinctions are exactly the ones that trip up embedding-based systems.

**Concrete example from my results:**

For query *"Public software companies with more than 1,000 employees"*:

| Company | Semantic Score | Cross-Encoder Score | Final Score | Movement |
|---|---|---|---|---|
| Capgemini | 0.871 (rank 1) | 0.593 | 0.705 | ↓ rank 5 |
| Tata Consultancy Services | 0.662 (rank 6) | 1.000 | 0.865 | ↑ rank 1 |
| ASGN | 0.784 (rank 3) | 0.849 | 0.823 | ↑ rank 2 |
| Globant | 0.809 (rank 2) | 0.748 | 0.772 | ↓ rank 4 |

The cross-encoder correctly identified TCS as the better match — it's a
massive public IT company with hundreds of thousands of employees, which
is exactly what the query asks for. Capgemini dropped because the cross-
encoder noticed it's primarily a consulting company that happens to have
software practices, and the candidate it was evaluating was actually the
Canadian subsidiary with a smaller footprint.

**I only apply the cross-encoder to the top 30 candidates.** Applying it
to all 500 would be too slow. Companies outside the top 30 after semantic
ranking almost certainly aren't good matches anyway.

**Final score formula:**
```
final_score = 0.4 × semantic_score + 0.6 × cross_encoder_score
```

---

### Stage 5 — Confidence Splitter

**File:** `pipeline/confidence_splitter.py`

After reranking, I split companies into three groups. The thresholds are
dynamic based on the query difficulty from Stage 1:

| Difficulty | Auto-qualify (θ_high) | Auto-reject (θ_low) |
|---|---|---|
| easy | ≥ 0.75 | < 0.30 |
| medium | ≥ 0.70 | < 0.25 |
| hard | ≥ 0.60 | < 0.20 |

For hard queries I lower the auto-qualify bar because the semantic and
cross-encoder scores are less reliable — more companies deserve a second
look from the LLM.

Typical distribution on a 477-company dataset:
```
Auto-qualify  ██████░░░░░░░░░░░░░░  ~25% (no LLM needed)
Gray zone     ████████████░░░░░░░░  ~35% (goes to LLM)
Auto-reject   ████████████████░░░░  ~40% (no LLM needed)
```

Only the gray zone reaches Stage 6.

---

### Stage 6 — Batch LLM Qualifier

**File:** `pipeline/batch_qualifier.py`

The gray zone companies go to the LLM — but **15 at a time**, not one by one.

Here's why batching matters:

| Approach | API calls for 60 gray zone companies | Consistency |
|---|---|---|
| One company per call | 60 calls | Lower — each call is independent |
| 15 companies per call | 4 calls | Higher — model sees all in same context |

When the model sees 15 companies simultaneously, it can compare them against
each other, not just against the query in isolation. This produces much more
consistent scores.

My prompt includes:
- The original query
- Target industries
- Implicit signals (what to look for)
- Negative terms (what NOT to look for — this is crucial)

The model returns a score 0-10 and a one-sentence reason for each company.

**Final score for LLM-evaluated companies:**
```
combined = 0.5 × pipeline_score + 0.5 × llm_score_normalized

if combined >= 0.55 → QUALIFIED
else               → REJECTED
```

---

## 3.2 Tradeoffs

### What I optimized for

**Accuracy over raw speed** — the cross-encoder adds 2-3 seconds per query
but meaningfully improves ranking quality. For a qualification system, getting
the right answer matters more than getting a fast wrong answer.

**Cost reduction over maximum accuracy** — the confidence splitter deliberately
avoids sending every company to the LLM. This reduces API usage by ~15x at the
cost of some accuracy on borderline cases. At production scale, this is essential.

**Robustness over simplicity** — the three-state filter logic and the
missing-data handling add complexity. But real datasets are messy and incomplete,
and a simpler system would silently fail on edge cases.

### Intentional tradeoffs

| Decision | Gained | Sacrificed |
|---|---|---|
| Cross-encoder on top 30 only | Speed | Accuracy for ranks 31+ |
| Dynamic difficulty thresholds | Adaptability | Simplicity |
| Missing fields → penalty not rejection | Recall | Some precision |
| BM25 + embeddings combined | Coverage | Simplicity |
| 15 companies per LLM call | Cost, consistency | Granular control |
| Local LLM (Gemma 3 4B via Ollama) | No rate limits, free, offline | Weaker complex reasoning |
| Disk cache for query analysis | No repeated API calls | Slightly stale if query intent changes |

**The local LLM tradeoff deserves explanation.** I started with the Google
Gemini API, but the free tier limit of 20 requests/day was exhausted almost
immediately across 12 queries. Rather than paying for API access, I switched
to running Gemma 3 4B locally via Ollama — zero cost, zero rate limits, runs
completely offline. The downside is that a 4B parameter local model is less
capable than a large cloud model on complex reasoning tasks, which occasionally
causes Stage 1 to generate imperfect structured filters for the hardest queries.

---

## 3.3 Error Analysis

### Query-by-query results

| Query | Companies Qualified | Notes |
|---|---|---|
| Logistic companies in Romania | 1 | Dataset has very few Romanian companies |
| Public software companies >1,000 emp. | 10 | Good — TCS correctly at rank 1 |
| Food & beverage manufacturers in France | 5 | Accurate — all French F&B |
| Packaging suppliers for cosmetics brands | 22 | Excellent — actual suppliers, not brands |
| Construction companies in US >$50M revenue | 10 | Solid, revenue filter works well |
| Pharmaceutical companies in Switzerland | 20 | Very good — all Swiss pharma |
| B2B SaaS HR solutions in Europe | 19 | Good — European HR software companies |
| Clean energy startups post-2018 <200 emp. | 13 | Good, mostly Scandinavian startups |
| Fast-growing fintech competing with banks | 8 | Inconsistent — complex query |
| E-commerce companies using Shopify | 4 | Limited — few Shopify mentions in data |
| Renewable energy equipment in Scandinavia | FAILED | Consistent failure — see below |
| EV battery component manufacturers | 27 | Excellent — very precise results |

### Where the system genuinely struggles

**1. Thin dataset coverage**

*"Logistic companies in Romania"* — the dataset contains very few Romanian
companies overall, and none that are logistics-focused. The system correctly
returns near-zero results. This is a data gap, not a system failure, but it's
worth flagging as something to monitor in production.

**2. Complex temporal and growth signals**

*"Fast-growing fintech companies"* — growth rate is not a field in the company
data. The system approximates it through implicit signals (recent founding,
startup language in descriptions) but can't actually verify growth. This is a
fundamental limitation of working with static company profiles.

**3. Technology-specific queries**

*"E-commerce companies using Shopify"* — very few company descriptions mention
Shopify by name. The system returns e-commerce adjacent companies (payment
processors, online storefronts) but can't reliably identify actual Shopify users
without technology stack data.

**4. Local LLM filter errors on hard queries**

For the most complex queries, Gemma 3 4B sometimes generates structured filters
that don't quite match the intent. For example, it might add employee count
filters for queries that don't mention company size, which can silently eliminate
valid companies before they reach the semantic stages.

**Concrete misclassification example:**

In query 9 (fast-growing fintech in Europe), "Magnora" — a Norwegian renewable
energy company — appeared as the top result in one run. The local LLM generated
filters that allowed energy companies through, and the semantic ranker found
enough financial vocabulary overlap ("capital", "investment", "revenue growth").
The cross-encoder should have caught this but the company description mentions
project financing extensively, which confused it.

---

## 3.4 Scaling

### Current: 477 companies, 12 queries

Works well. Full pipeline runs in 5-10 minutes end-to-end on a laptop.

### At 100,000 companies per query

| Component | Change needed |
|---|---|
| Hard Filter | None — Python filtering on 100K rows is still milliseconds |
| BM25 | Replace `rank-bm25` with Elasticsearch or OpenSearch |
| Embeddings | Pre-compute and store in vector DB (Pinecone, pgvector, Weaviate) with ANN search |
| Cross-Encoder | None — still only sees top 30 |
| Batch LLM | None — gray zone stays bounded even at 100K |

The critical addition is a **vector database with ANN indexing**. Instead of
computing cosine similarity against all 100K embeddings at query time (O(n)),
we retrieve top-500 candidates via approximate nearest neighbor search (O(log n))
using an HNSW or IVF index. Then the full pipeline runs on those 500.

### At 1,000,000 companies

Add a two-stage retrieval before everything else:
```
1M companies
    ↓ BM25 / Elasticsearch → top 10,000
    ↓ Embedding ANN search → top 500
    ↓ [existing pipeline: hard filter → semantic → cross-encoder → LLM]
    ↓ Final results
```

This is the standard "retrieve-then-rerank" pattern used in production search
systems. It keeps each stage processing manageable volumes regardless of total
dataset size.

---

## 3.5 Failure Modes

### When the system produces confidently wrong results

**1. Vocabulary overlap without role match**

A software company that builds logistics management tools will use a lot of
logistics vocabulary in its description. The system might rank it highly for
"logistics companies" queries. The cross-encoder partially mitigates this —
it can detect "provides software for logistics" vs "is a logistics company" —
but it's not perfect.

**2. Local LLM filter hallucinations**

The biggest risk in the current setup. When Gemma generates incorrect structured
filters in Stage 1, those errors cascade through the entire pipeline. A wrong
country code or a spurious employee count filter can silently eliminate the best
matches before they're even ranked.

**3. Description quality variance**

Some companies in the dataset have rich, detailed descriptions. Others have one
sentence. The system heavily relies on description quality for semantic ranking,
so sparse descriptions lead to unreliable scores.

**4. Cross-encoder score saturation**

When multiple companies score near 1.0 with the cross-encoder, normalization
compresses real differences. Two companies at 0.92 and 0.89 become nearly
indistinguishable after normalization, even if one is clearly more relevant.

### What I would monitor in production

| Signal | What it detects | Alert threshold |
|---|---|---|
| Gray zone size per query | Thresholds need tuning | > 50% of candidates |
| LLM qualification rate | LLM not adding value | < 10% change from pre-LLM ranks |
| Cross-encoder vs semantic correlation | Reranker effectiveness | Pearson r > 0.95 |
| Queries returning 0 results | Dataset coverage gaps | Any sustained pattern |
| Hard filter pass rate | LLM generating bad filters | < 5% of dataset passing |
| Stage 1 filter accuracy | Spot check | Monthly manual review |

---

## Reflection

### What works well

The cascade architecture is genuinely effective at concentrating compute where
it matters. For clearly structured queries (country + size + public status), the
hard filter resolves 80% of the work for free. The cross-encoder meaningfully
improves ranking quality on ambiguous cases — the TCS vs Capgemini example above
is a good demonstration of this.

Query expansion via Stage 1 is the single highest-impact feature for recall.
Companies don't always use the vocabulary of the query, and enriching the
embedding with expanded terms catches a lot of matches that would otherwise
be missed.

### What I'd prioritize next

**1. Negative term filtering at semantic level** — currently negative terms
only affect the LLM prompt. I'd add explicit embedding-based penalties for
companies that match negative terms, earlier in the pipeline.

**2. A proper evaluation framework** — right now I'm judging results by eye.
Even 50-100 hand-labeled (query, company, relevant?) triples would let me
compute Precision@K and NDCG@K and actually measure whether changes improve
things.

**3. Stronger local model** — Gemma 3 4B is the weak link in Stage 1 for
complex queries. A 7B or 13B model via Ollama would improve filter quality
significantly without adding API costs.

**4. Deduplication** — a few queries return the same company twice (different
records for the same entity). Simple name similarity dedup would fix this.

**5. Confidence calibration** — the score thresholds in Stage 5 are currently
tuned by intuition. With labeled data, I'd calibrate them using precision-recall
curves per query type.

### Key assumptions

- Company descriptions accurately reflect actual business activities
- English is the dominant language for descriptions in the dataset
- Query intent can be reliably extracted from query text alone
- Static profiles contain enough signal to judge supply chain roles

All four are violated to some degree in practice. The most interesting one is
assumption 4 — determining whether a company is a *supplier* vs a *customer*
vs a *competitor* often requires relationship data that isn't in a static
company profile. This is a genuinely hard problem and I'd love to understand
how it gets approached at scale in production systems.

---

*Stack: Python · sentence-transformers · rank-bm25 · Ollama (Gemma 3 4B) ·
scikit-learn · pandas*

*Models used: all-MiniLM-L6-v2 (~90MB) · cross-encoder/ms-marco-MiniLM-L-6-v2
(~85MB) · Gemma 3 4B (~3.3GB local via Ollama)*