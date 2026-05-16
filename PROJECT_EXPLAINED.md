# PG Schema Discovery — Full Technical Explanation

This document explains the entire backend pipeline: what it does, why it exists, how every script and module works at both a conceptual and code level.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Big Picture: Three Stages](#2-the-big-picture-three-stages)
3. [Core Concepts You Need to Know](#3-core-concepts-you-need-to-know)
4. [Stage 1 — Ground Truth Extraction](#4-stage-1--ground-truth-extraction)
5. [Stage 2 — Schema Inference (Neo4j + LLM)](#5-stage-2--schema-inference-neo4j--llm)
6. [Stage 3 — Evaluation / Comparison](#6-stage-3--evaluation--comparison)
7. [Entry-Point Scripts](#7-entry-point-scripts)
8. [Data Flow Summary](#8-data-flow-summary)
9. [Results and What They Mean](#9-results-and-what-they-mean)
10. [Why Results Differ Across Datasets](#10-why-results-differ-across-datasets)

---

## 1. What Is This Project?

A **Property Graph (PG)** is a database model made of nodes and edges, where both can have typed properties. Neo4j is the most common PG database. A **schema** for such a graph declares: what node types exist, what edge types exist, which nodes an edge can connect, and what properties each type must or may carry.

The project answers the question: **can an LLM automatically discover the schema of a property graph it has never seen before, by looking at the data?**

The workflow is:
1. You have a graph loaded into Neo4j.
2. The system mines statistical patterns from Neo4j using Cypher queries.
3. Those patterns are formatted into a compact text profile and sent to **Gemini 2.5 Flash**.
4. Gemini returns a structured JSON schema.
5. The inferred schema is compared against a human-written ground-truth schema using precision/recall/F1.

---

## 2. The Big Picture: Three Stages

```
┌──────────────────────────────┐
│  01_gts/gt_data_<dataset>/   │  ← .pgs files (human-authored schema specs)
└──────────────────────────────┘
              │
              ▼
      [Stage 1: extract_gt.py]
              │
              ▼
┌────────────────────────────────────────────────────┐
│  03_outputs/schemas/ground_truth/<ds>/gt_<ds>.json  │
└────────────────────────────────────────────────────┘
                                                       ▲
                                                       │ [Stage 3: compare.py]
┌────────────────────────────────────────────────────┐ │
│  03_outputs/schemas/inferred/<ds>/inf_<ds>.json     │─┘
└────────────────────────────────────────────────────┘
              ▲
              │
      [Stage 2: infer.py]
              │
       ┌──────┴──────┐
       │             │
   [Neo4j]       [Gemini API]
```

---

## 3. Core Concepts You Need to Know

### Node Pattern (Definition 3.5 from PG-HIVE, EDBT 2026)
A **node pattern** is a pair `(L, K)` where:
- `L` = the exact set of labels on a node (e.g. `{Person, Employee}`)
- `K` = the exact set of property keys present on that node

Two nodes with the same label set but different properties are **distinct patterns** within the same **type**.

### Edge Pattern (Definition 3.6)
An **edge pattern** is a triple `(L, K, R)` where:
- `L` = the relationship type (e.g. `KNOWS`)
- `K` = property keys on the edge
- `R = (Ls, Lt)` = the label sets of the source and target nodes

### Cardinality
Expressed as `"a..b : c..d"` where each side is `0..1`, `0..N`, `1..1`, or `1..N`.
- Left side = outgoing (source perspective): how many edges of this type does each source node send?
- Right side = incoming (target perspective): how many edges of this type does each target receive?
- Uses OPTIONAL MATCH so nodes with zero edges count as zero, giving a true minimum.

### MANDATORY vs OPTIONAL
A property is **MANDATORY** if its `fill_ratio = 1.0` — it appears on every single instance of that node/edge type. If even one instance lacks it, it becomes OPTIONAL.

---

## 4. Stage 1 — Ground Truth Extraction

### High-Level Purpose
PG-Schema files (`.pgs`) are a formal language for declaring graph schemas — similar to SQL's `CREATE TABLE` but for graphs. The ground truth extractor **parses these files and converts them to the same JSON format** that the inference stage produces, so the two can be compared directly.

### Files Involved
| File | Role |
|---|---|
| `scripts/extract_gt.py` | Entry point, sets I/O paths, calls the pipeline |
| `src/pg_schema_llm/pipeline/extract_gt.py` | All the actual parsing logic |

---

### `scripts/extract_gt.py` (entry point)

```
python scripts/extract_gt.py fib25
```

This script does only three things:
1. Parses the dataset name from the command line.
2. Constructs paths: input = `01_gts/gt_data_fib25/`, output = `03_outputs/schemas/ground_truth/fib25/gt_fib25.json`.
3. Calls `run_extract_gt(input_dir, output_file)` from the pipeline module.

It adds `src/` to `sys.path` so Python can find the `pg_schema_llm` package without installing it.

---

### `src/pg_schema_llm/pipeline/extract_gt.py` (the real parser)

#### What a `.pgs` file looks like

```sql
CREATE NODE TYPE (Person : Person {
    id Long,
    name String,
    OPTIONAL birthDate String
});

CREATE EDGE TYPE (: Person)-[:KNOWS { weight Double }]->(: Person);

CREATE GRAPH TYPE MyGraph LOOSE { ... };
```

#### Step 1 — Strip comments
`_strip_comments(content)` removes `--`, `//`, and `/* */` style comments using regex before any parsing happens. This prevents comment text from being confused with schema keywords.

#### Step 2 — Split on semicolons
The file is split on `;` to get individual statements. Each statement is classified as a node type, edge type, or graph type declaration (graph type is ignored — it's just a container declaration).

#### Step 3 — Parse node types: `_parse_node_stmt(stmt)`

Parses statements of the form `CREATE NODE TYPE (TypeName : Label1 & Label2 { properties })`.

- Finds the outermost `( )` and splits it into the definition part and property block.
- The definition part `TypeName : Label1 & Label2` is split on `:` to get the type name and labels. Labels separated by `&` are all collected into a list.
- The property block `{ prop1 Long, OPTIONAL prop2 String }` is passed to `_parse_props()`.
- The `name` field of the output is set to the shortest label (since the `.pgs` type name may differ from any single label).

#### Step 4 — Parse properties: `_parse_props(block)`

Goes line by line through a property block. For each line:
- Checks if the line starts with `OPTIONAL` (removes the keyword, marks `constraint = "OPTIONAL"`).
- Otherwise marks `constraint = "MANDATORY"`.
- Splits on whitespace to get the property name (first token) and type (last token).
- Maps the raw PGS type to a canonical generic type via `_TYPE_MAP`:
  - `Long` → `INTEGER`, `Double` → `DOUBLE`, `String` → `STRING`, `Boolean` → `BOOLEAN`, `Date`/`DateTime` → `DATE`, `StringArray`/`Array` → `LIST`, `Point` → `STRING` (spatial is treated as string for comparison).

#### Step 5 — Parse edge types: `_parse_edge_stmt(stmt)`

Parses statements of the form `(: SrcLabel1 | : SrcLabel2)-[:REL_TYPE { props }]->(: TgtLabel)`.

- Finds the arrow pattern `-[...]->` using regex to locate the edge definition.
- The bracket contents give the relationship name (after `:`) and optional properties.
- Everything before the arrow = source block; everything after = target block.
- Source/target blocks may contain `|` alternatives (different node types that can be on either end).
- `_parse_label_alternatives()` splits on `|` and handles `&` conjunctions.
- **Cross-product expansion**: if there are 2 source alternatives and 3 target alternatives, 6 separate edge entries are produced. This is why edge counts in GT can be large.

#### Output JSON format

```json
{
  "dataset_name": "gt_data_fib25",
  "node_types": [
    {
      "name": "Neuron",
      "labels": ["Neuron"],
      "properties": [
        {"name": "bodyId", "type": "INTEGER", "constraint": "MANDATORY"},
        {"name": "type", "type": "STRING", "constraint": "OPTIONAL"}
      ]
    }
  ],
  "edge_types": [
    {
      "name": "ConnectsTo",
      "source_labels": ["Neuron"],
      "target_labels": ["Neuron"],
      "is_canonical": true,
      "cardinality": null,
      "properties": [...]
    }
  ]
}
```

Note that GT has `cardinality: null` because `.pgs` files do not declare cardinality — only the mined inferred schema has it.

---

## 5. Stage 2 — Schema Inference (Neo4j + LLM)

This is the core of the project. It runs entirely against a live Neo4j database and an LLM API.

### Files Involved
| File | Role |
|---|---|
| `scripts/infer.py` | Entry point |
| `src/pg_schema_llm/pipeline/infer_schema.py` | Orchestration: connects to Neo4j, calls miner, calls LLM, saves output |
| `src/pg_schema_llm/io/neo4j_io.py` | All Cypher queries, pattern mining, type inference, cardinality |
| `src/pg_schema_llm/llm/prompts.py` | Constructs the prompt sent to Gemini |

---

### `scripts/infer.py` (entry point)

```
python scripts/infer.py fib25
```

Same structure as the GT entry point: parses dataset name, constructs paths, calls `run_infer_schema(data_dir, out_file)`. The `data_dir` argument is accepted but **ignored** — it was used in an older CSV-based version. Data now comes exclusively from Neo4j.

---

### `src/pg_schema_llm/pipeline/infer_schema.py` (orchestrator)

#### `InferConfig` dataclass

A configuration container holding:
- Neo4j connection strings (fall back to `.env` if not set)
- `expand_edge_subsets: bool = True` — whether to generate subset permutations
- `gemini_model: str = "gemini-2.5-flash"` — which Gemini model to use
- `verbose: bool = True` — whether to print progress

#### `_make_driver(cfg)` — Neo4j connection

Loads `.env` via `python-dotenv`, reads `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and creates a `neo4j.GraphDatabase.driver`. Defaults to `bolt://localhost:7687` / `neo4j` if variables are missing.

#### `infer_schema_from_folder(data_dir, config)` — main flow

```python
driver = _make_driver(cfg)
patterns = mine_patterns(driver, expand_edge_subsets=True, database=None)
profile_text = build_profile_text_from_patterns(patterns)
prompt = build_inference_prompt(profile_text)
raw_res = call_gemini_api(prompt, ...)
schema = extract_json(raw_res)
schema["_mined_patterns"] = patterns   # attached for debugging
```

The `_mined_patterns` key in the saved JSON lets you see exactly what the miner found, independently of what the LLM said.

#### `call_gemini_api(prompt, ...)` — LLM call

- Configures `google.generativeai` with the API key from `.env`.
- Calls `model.generate_content()` requesting `response_mime_type = "application/json"` — this forces Gemini to return valid JSON.
- Sets `max_output_tokens = 65536` to avoid truncation on large schemas.
- Logs `finish_reason` and token usage when verbose.
- If `res.text` raises (e.g. due to safety filtering or truncation), falls back to manually concatenating `parts` from the candidate.

#### `extract_json(text)` — JSON rescue logic

The LLM doesn't always return clean JSON. Three fallback strategies:
1. Strip markdown code fences (` ```json ... ``` `) and try `json.loads()`.
2. Find the largest balanced `{...}` substring by counting brace depth, then try parsing that.
3. Walk backwards from the end of the string, trimming characters and trying to close open brackets with common closers like `}`, `]}`, etc. — rescues simple truncation cases.

#### `build_profile_text_from_patterns(patterns)` — compact text format

Converts the mined pattern dict into a structured text the LLM reads. Format:

```
# Legend: N=node, E=canonical edge, E*=subset edge, K=pattern keys, P=properties; /M=MANDATORY, /O=OPTIONAL.

## NODE TYPES
N: {Person}
  K: {id, name, birthDate}
  K: {id, name}
  P: birthDate:STRING/O, id:INTEGER/M, name:STRING/M

## EDGE TYPES
E: {Person}-[:KNOWS]->{Person}  card=1..1:0..N
  K: {weight}
  P: weight:DOUBLE/O
```

This is deliberately compact (one line per entity) to minimize token usage while keeping all information the LLM needs to reconstruct the schema.

---

### `src/pg_schema_llm/io/neo4j_io.py` (the mining engine)

This is the most complex module. It runs all Cypher queries against Neo4j and computes all statistical properties.

#### Type inference: `_infer_value_type(v)`

Called on each individual property value. Returns one of: `BOOLEAN`, `INTEGER`, `DOUBLE`, `DATE`, `STRING`, `LIST`.

Priority order (from most specific to most general):
1. Python `bool` → `BOOLEAN`
2. Python `int` → `INTEGER`
3. Python `float` → `DOUBLE`
4. Python `list` → `LIST`
5. String that parses as `true`/`false` → `BOOLEAN`
6. String that parses as integer → `INTEGER`
7. String that parses as float → `DOUBLE`
8. String matching date patterns (YYYY-MM-DD etc.) → `DATE`
9. Anything else → `STRING`

#### `_resolve_property_type(type_counts: Counter)` 

After scanning all values for a property, picks the **most general** type observed. The rank order is: `BOOLEAN < INTEGER < DOUBLE < DATE < STRING < LIST`. This is conservative — if even one value is a STRING in an otherwise INTEGER column, the whole property becomes STRING.

#### The six Cypher queries in `mine_patterns(driver)`

**Query 1 — Node pattern scan:**
```cypher
MATCH (n)
RETURN labels(n) AS labels, keys(n) AS props, count(*) AS cnt
```
Groups all nodes by their exact label set + property key set. Returns the count of nodes per group. This tells us every distinct `(L, K)` node pattern and how common each is.

**Query 2 — Full node property type scan (one per node label set):**
```cypher
MATCH (n) WHERE size(labels(n)) = 2 AND 'Person' IN labels(n) AND 'Employee' IN labels(n)
RETURN properties(n) AS props
```
Streams every node of a given label set and inspects each value. Uses `fetch_size=1000` so the driver fetches rows in batches — the whole result never lives in memory at once. This is O(n) time and bounded memory regardless of graph size. Progress is printed every 50,000 nodes.

**Query 3 — Edge pattern scan:**
```cypher
MATCH (a)-[r]->(b)
RETURN type(r) AS rt, labels(a) AS sl, labels(b) AS tl, keys(r) AS props, count(*) AS cnt
```
Groups all edges by `(relationship type, source label set, target label set, edge property key set)`. This identifies every canonical edge type.

**Query 4 & 5 — Cardinality (one pair per canonical edge type):**
```cypher
-- Out-degree side (source perspective):
MATCH (a) WHERE <exact src labels>
OPTIONAL MATCH (a)-[r:REL_TYPE]->(b) WHERE <exact tgt labels>
WITH a, count(r) AS od
RETURN min(od) AS mn, max(od) AS mx, count(a) AS total

-- In-degree side (target perspective):
MATCH (b) WHERE <exact tgt labels>
OPTIONAL MATCH (a)-[r:REL_TYPE]->(b) WHERE <exact src labels>
WITH b, count(r) AS id
RETURN min(id) AS mn, max(id) AS mx, count(b) AS total
```
`OPTIONAL MATCH` is critical — without it, nodes with zero edges wouldn't appear in the result, and `min` would never be 0. With it, a source node that participates in no edges of this type contributes `od = 0`, so `min_out = 0` correctly. The result is formatted as `"0..N : 1..1"` etc.

**Query 6 — Full edge property type scan (one per canonical edge type):**
Same streaming pattern as Query 2 but for edges.

#### Cardinality formatting: `_format_cardinality(min_out, max_out, min_in, max_in)`

```python
def side(lo, hi):
    lo_s = "0" if lo == 0 else "1"
    hi_s = "1" if hi <= 1 else "N"
    return f"{lo_s}..{hi_s}"
return f"{side(min_out, max_out)} : {side(min_in, max_in)}"
```

Examples:
- Every source sends exactly one edge, some targets receive many → `"1..1 : 0..N"`
- Optional on both sides, many-to-many → `"0..N : 0..N"`

#### Mandatory/Optional determination

For each property on a node type, the fill count is compared to the total node count:
```python
constraint = "MANDATORY" if fill == total else "OPTIONAL"
fill_ratio = round(fill / total, 4)
```
Fill is computed by summing pattern counts across all patterns that include the property. If every node has the property, it's mandatory.

#### Subset edge expansion

When `expand_edge_subsets=True`, after mining canonical edges, the code generates all non-empty subsets of the source and target label sets for each edge. For example, if the canonical edge is `{Person, Employee}-[:WORKS_AT]->{Company, Organization}`, the subsets would include:
- `{Person}-[:WORKS_AT]->{Company}`
- `{Person}-[:WORKS_AT]->{Organization}`
- `{Employee}-[:WORKS_AT]->{Company}`
- etc.

These are marked `is_canonical: False` and have `cardinality: null`. They exist so the LLM (and the evaluator) can match against partial label sets, which sometimes appear in the GT when the schema was written at a less granular level.

---

### `src/pg_schema_llm/llm/prompts.py` (the prompt)

The prompt is a single function `build_inference_prompt(profile_text)` that wraps the profile in a large instruction block.

#### Structure of the prompt

**Part 1 — Role and framing:**  
Tells Gemini it is a "Senior Property Graph Schema Architect" receiving results of an exhaustive (not sampled) pattern mining pass. Emphasizes: *the data is the source of truth, do not clean it up or compensate for sampling error*.

**Part 2 — Format legend:**  
Explains the compact line-based format (`N:`, `E:`, `E*:`, `K:`, `P:`) so Gemini can read the profile text correctly.

**Part 3 — The actual profile text** (inserted from miner output).

**Part 4 — Theoretical framework:**  
Defines Node Pattern `(L, K)` and Edge Pattern `(L, K, R)` formally, so Gemini understands the semantics.

**Part 5 — Hard rules (R1–R7), non-negotiable:**
- **R1**: Never rename labels. Copy them verbatim.
- **R2**: Node `name` must be one of its actual labels.
- **R3**: Edge names must be copied verbatim from the relationship type.
- **R4**: Include `source_labels` and `target_labels` arrays on every edge.
- **R5**: Don't filter or remove properties. Don't collapse patterns.
- **R6**: Drop Neo4j import-only columns (`:START_ID`, `:END_ID`, `:TYPE`, `:LABEL`).
- **R7**: Copy cardinality verbatim; set null for non-canonical edges.

**Part 6 — Output JSON schema:**  
Exact expected JSON structure with typed fields.

The rules exist because without them, Gemini tends to rename things, invent cleaner names, drop uncommon properties, or merge edge types — all of which reduce evaluation scores.

---

## 6. Stage 3 — Evaluation / Comparison

### High-Level Purpose
Given two JSON schemas (GT and inferred), compute how well the inferred one matches the ground truth. The evaluation follows PG-HIVE (EDBT 2026) definitions of node/edge patterns.

### Files Involved
| File | Role |
|---|---|
| `scripts/compare.py` | Entry point |
| `src/pg_schema_llm/pipeline/compare.py` | All matching and metric computation logic |

---

### `scripts/compare.py` (entry point)

```
python scripts/compare.py fib25
# or with custom paths:
python scripts/compare.py fib25 --gt path/to/gt.json --inf path/to/inf.json
```

Constructs default paths and calls `run_compare(gt_path, inf_path)`.

---

### `src/pg_schema_llm/pipeline/compare.py` (the evaluator)

#### Schema normalisation: `_normalise_schema(schema)`

Both the GT and inferred schemas go through the same normalisation before comparison. This handles format differences between the two:

| GT uses | Inferred uses | Normaliser handles |
|---|---|---|
| `"type": "Long"` | `"data_type": "INTEGER"` | `_norm_type()` maps all aliases |
| `"mandatory": true/false` | `"constraint": "MANDATORY"/"OPTIONAL"` | `_prop_info()` unifies both |
| topology blocks with `allowed_sources`/`allowed_targets` | `source_labels`/`target_labels` per edge | cross-product expansion |
| `"start_node"/"end_node"` strings | label arrays | `_node_label_key_for_name()` resolves |

After normalisation, everything is keyed by `frozenset`:
- Nodes by `LabelKey = frozenset[str]` (their label set)
- Edges by `EdgeTypeKey = (rel_name, src_LabelKey, tgt_LabelKey)`

This makes matching an O(1) set intersection rather than a string-matching problem.

#### The 7 comparison sections

**Section 1 — Node Type matching**

```python
matched_nodes = gt_node_lks & inf_node_lks   # set intersection on frozensets
only_gt  = gt_node_lks - inf_node_lks         # missed by inference
only_inf = inf_node_lks - gt_node_lks         # invented by inference
```

Metrics:
- Precision = |matched| / |inferred|
- Recall = |matched| / |GT|
- F1 = 2·P·R / (P+R)

This is an **exact label-set match**. If the GT has `{Person}` and the inferred has `{person}` (different case), they do not match — the hard rules in the prompt exist precisely to prevent this.

**Section 2 — Node Pattern matching (Definition 3.5)**

A GT node pattern `(L, K_gt)` is "covered" by the inferred schema if:
- The inferred schema has the same label set `L`
- AND `K_gt ⊆ K_inf` — the GT property key set is a subset of what was inferred

This is lenient: inferring extra properties doesn't hurt recall, but missing properties does.

**Section 3 — Node Property Accuracy**

For each matched node type (same label set in both GT and inferred), for each property that appears in both:
- Compare `constraint` (MANDATORY/OPTIONAL) → constraint accuracy
- Compare `data_type` (STRING, INTEGER, etc.) → data type accuracy

Both are reported as simple accuracy fractions.

**Section 4 — Edge Type matching**

Identical logic to node types but keyed by `(rel_name, src_frozenset, tgt_frozenset)`. An edge matches only if **all three components** match exactly. This is the hardest metric to satisfy because the source/target label sets must also match, not just the relationship name.

This is why edge scores drop first: the node mapping must be perfect for edges to match.

**Section 5 — Edge Pattern matching (Definition 3.6)**

Same subset-coverage logic as Section 2 but for edge property keys.

**Section 6 — Edge Property Accuracy**

Same as Section 3 but for edge properties.

**Section 7 — Cardinality Accuracy**

For matched edges that have a GT cardinality string, checks if the inferred cardinality string matches exactly after normalisation (via `_normalize_cardinality()`). GT `.pgs` files don't declare cardinality, so this is always `null` in GT and this section typically reports `N/A`.

#### Final MACRO F1

```python
macro_f1 = mean([node_type_f1, node_pattern_f1, edge_type_f1, edge_pattern_f1])
```

This single number is the headline accuracy figure. It equally weights node-level and edge-level performance at both the type and pattern granularity.

---

## 7. Entry-Point Scripts

### `scripts/run_pipeline.py` — full pipeline runner

```
python scripts/run_pipeline.py fib25
python scripts/run_pipeline.py fib25 --skip-gt        # if GT already extracted
python scripts/run_pipeline.py fib25 --skip-infer     # if inference already done
python scripts/run_pipeline.py fib25 --skip-compare   # just run the first two steps
```

Runs all three stages in sequence via `subprocess.run()`. Stops and exits non-zero if any stage fails. Sets `PYTHONPATH=src` in the subprocess environment so the `pg_schema_llm` package is always importable.

### `scripts/inspect_gt.py` — quick JSON inspector

```
python scripts/inspect_gt.py 03_outputs/schemas/ground_truth/fib25/gt_fib25.json
```

A debugging utility. Loads a JSON file and prints:
- All top-level keys
- Which key contains edge definitions (tries `edge_types`, `relationships`, `edges`, `relationship_types`)
- The count of edges found
- A sample edge object

Useful for sanity-checking output files after extraction.

---

## 8. Data Flow Summary

```
.pgs file
    │
    │ _strip_comments()
    │ split on ";"
    │ _parse_node_stmt()  →  node dicts
    │ _parse_edge_stmt()  →  edge dicts (cross-product of | alternatives)
    ▼
gt_<dataset>.json
    {"node_types": [...], "edge_types": [...]}


Neo4j database
    │
    │  Query 1: MATCH (n) RETURN labels, keys, count  →  node patterns
    │  Query 2: stream properties(n) per label set    →  type counts (full scan)
    │  Query 3: MATCH (a)-[r]->(b) RETURN type,labels,keys,count  →  edge patterns
    │  Query 4-5: OPTIONAL MATCH cardinality per canonical edge
    │  Query 6: stream properties(r) per canonical edge  →  type counts
    │
    │  _resolve_property_type()  →  most general type
    │  fill_ratio == 1.0 → MANDATORY, else OPTIONAL
    │  _format_cardinality()  →  "a..b : c..d"
    │  subset expansion (Python, no DB)
    ▼
patterns dict
    {"node_types": [...], "edge_types": [...]}
    │
    │  build_profile_text_from_patterns()
    │    N: {Label}  K: {props}  P: prop:TYPE/M, ...
    │    E: {Src}-[:REL]->{Tgt}  card=...
    ▼
profile_text (compact token-efficient string)
    │
    │  build_inference_prompt(profile_text)
    │    Role + Format legend + Profile + Theory + Rules + Output schema
    ▼
Gemini 2.5 Flash API
    │
    │  response_mime_type="application/json"
    │  max_output_tokens=65536
    ▼
raw JSON response
    │
    │  extract_json()  (3-layer fallback: direct / balanced-brace / truncation-rescue)
    ▼
inf_<dataset>.json
    {"node_types": [...], "edge_types": [...], "_mined_patterns": {...}}


gt_<dataset>.json  +  inf_<dataset>.json
    │
    │  _normalise_schema()  (unify type aliases, constraint formats, topology blocks)
    │
    │  Section 1: node label-set intersection         → Node Type P/R/F1
    │  Section 2: pattern subset coverage             → Node Pattern P/R/F1
    │  Section 3: per-property constraint + type acc  → Node Property accuracy
    │  Section 4: edge (rel, src_lk, tgt_lk) match   → Edge Type P/R/F1
    │  Section 5: edge pattern subset coverage        → Edge Pattern P/R/F1
    │  Section 6: edge property constraint + type acc → Edge Property accuracy
    │  Section 7: cardinality string equality         → Cardinality accuracy
    │
    │  Macro F1 = mean(Node Type F1, Node Pattern F1, Edge Type F1, Edge Pattern F1)
    ▼
Terminal output + CompareResult dataclass
```

---

## 9. Results and What They Mean

| Dataset | Macro F1 | Notes |
|---|---|---|
| STARWARS | ~96% | Small, clean, few types |
| POLE | ~96% | Crime investigation dataset |
| LDBC | ~93% | Social network benchmark |
| FIB25 | ~70% | FlyWire neuroscience connectome |
| MB6 | ~66% | Most complex structure |

Macro F1 is the average of four F1 scores:
- **Node Type F1**: Did we find the right node types (exact label sets)?
- **Node Pattern F1**: Did we infer the right property keys on each node type?
- **Edge Type F1**: Did we find the right edges with the right source/target node types?
- **Edge Pattern F1**: Did we infer the right properties on each edge type?

A score of 96% on STARWARS means the LLM almost perfectly reproduced a human-written schema for that graph, just from mining its data.

---

## 10. Why Results Differ Across Datasets

**STARWARS / POLE (high scores):** These datasets have clean, semantically clear schemas with few node types, short label names, and simple edge topology. The LLM easily understands the data and the hard rules prevent it from drifting.

**LDBC (good score):** The social network benchmark has many node types and edges but they follow standard social graph conventions. The LLM has likely seen similar schemas in training data.

**FIB25 / MB6 (lower scores):** These are scientific/biological datasets with unusual naming conventions (e.g., neuroscience terminology), complex multi-label nodes, and high-cardinality topologies. The edge topology is where most points are lost:

1. **Node mapping drives edge matching.** Even if the LLM perfectly reproduces a node's property list, if it outputs a slightly different label set (or the normalizer can't align it), all edges touching that node fail to match too.

2. **GT topology is sometimes more aggregated.** The GT `.pgs` may declare a single edge between abstract node types, while the miner sees many canonical combinations due to multi-labeling. Cross-product expansion helps but doesn't fully solve this.

3. **LLM semantic drift.** Without the strict hard rules, Gemini would rename edges or merge similar ones. The rules prevent most of this but not all — especially on datasets with novel vocabulary.
