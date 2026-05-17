# Session Changes — PG-HIVE Evaluation Alignment

Four changes were made to bring the pipeline's evaluation logic into alignment with the PG-HIVE paper (Sideri et al., EDBT 2026).

---

## 1. Cardinality Accuracy — Always 0 Bug Fixed

### Problem

Ground-truth JSON files are extracted from `.pgs` schema files. Those files describe the *shape* of the graph (node types, edge types, property constraints) but carry **no cardinality information** — every edge's `cardinality` field is `null`. The old comparison logic in `compare.py` only evaluated cardinality when the GT had an explicit non-null value:

```python
if gt_c:          # always False → card_tot stays 0 forever
    card_tot += 1
```

This meant cardinality accuracy was **always 0.0** (or reported as "N/A"), regardless of how well the LLM inferred cardinalities.

### Root Cause

Cardinality is not encoded in `.pgs` files. The real ground-truth cardinality must come from the *actual graph* — which `mine_patterns()` in `neo4j_io.py` already computes correctly via OPTIONAL MATCH queries (giving `0..1`, `0..N`, `1..1`, `1..N` per side). These mined cardinalities are stored in the `_mined_patterns` key that `infer_schema.py` attaches to every inferred schema JSON.

### Fix

**File:** `src/pg_schema_llm/pipeline/compare.py`

After loading the inferred JSON, a lookup table `mined_card` is built from `_mined_patterns`:

```python
mined_card: Dict[EdgeTypeKey, str] = {}
for et in inf_raw.get("_mined_patterns", {}).get("edge_types", []):
    if et.get("is_canonical") and et.get("cardinality"):
        ...
        mined_card[(rel, src_lk, tgt_lk)] = _normalize_cardinality(et["cardinality"])
```

In section 7 (Cardinality), the reference is now:

```python
ref_c = gt_c if gt_c is not None else mined_card.get(ek)
```

- If the GT JSON has an explicit cardinality (future-proofed for GT files that do), it wins.
- Otherwise the mined cardinality — computed directly from OPTIONAL MATCH over the real graph — is used as the reference.

The LLM prompt (rule R7) already instructs the model to copy cardinalities verbatim from the mined profile, so this gives a fair and meaningful accuracy score.

### Effect

For the POLE dataset, 17 edge types are now evaluated and all score correctly. Cardinality accuracy went from N/A (always 0) to a real score.

---

## 2. Case-Insensitive Label Matching

### Problem

Neo4j label names are case-sensitive at the database level, but in practice datasets sometimes mix casing — e.g. `"Person"` in the ground-truth `.pgs` file vs `"person"` in the mined patterns, or across different dataset dumps. The old `_lk()` function preserved case, so these would never match:

```python
# Before
def _lk(labels) -> LabelKey:
    return frozenset(str(l) for l in (labels or []))
```

A GT type `{Person}` and an inferred type `{person}` would appear as two separate non-matching types, incorrectly penalising precision and recall.

### Fix

**File:** `src/pg_schema_llm/pipeline/compare.py`

`_lk()` now lowercases every label before building the frozenset:

```python
def _lk(labels) -> LabelKey:
    # Normalize to lowercase so "Person" and "person" collapse to the same key.
    return frozenset(str(l).lower() for l in (labels or []))
```

Because `_lk()` is the single entry point for all label-set construction throughout the file — used for node types, edge source/target labels, pattern keys, and the mined cardinality lookup — this single change makes **all** comparisons case-insensitive with no other modifications needed.

### Effect

Labels that differ only by casing are treated as identical during comparison. A GT type `{Person}` now correctly matches an inferred type `{person}`, eliminating false mismatches caused by inconsistent capitalisation across schema sources.

---

## 3. Label Backup and Unlabeled-Data Evaluation (`label_noise.py`)

### Background

The PG-HIVE paper (Section 5.2) evaluates schema discovery under three **label availability scenarios**: 100%, 50%, and 0% of node labels retained. For 50% and 0%, nodes have their labels removed before schema discovery runs. After discovery, the paper assigns the **majority original label** of each cluster's members as the cluster's type, then computes F1 against the labeled ground truth.

The key insight is that this lets you evaluate schema discovery even when the inference was done without labels — the stored original labels serve as the evaluation oracle.

### What the Script Does

**File:** `scripts/label_noise.py`

The script has five sub-commands:

| Command | Purpose |
|---|---|
| `backup` | Snapshot every node's current labels into `_orig_labels` (list property) and `_orig_label_concat` (joined string). Adds a Neo4j marker label `OriginalLabel` to all nodes. Idempotent. |
| `majority-assign <inferred.json>` | For each inferred node/edge type, query Neo4j for matching nodes, tally their `_orig_labels`, and rewrite the schema with majority-voted labels. Outputs a new JSON ready for `compare.py`. |
| `apply <pct>` | Randomly strip labels from `pct`% of nodes (seed-controlled for reproducibility). Simulates partial-label scenarios before running `infer.py`. |
| `restore` | Re-add `_orig_labels` to every stripped node. |
| `clean` | Remove all backup metadata and the `OriginalLabel` marker. |

#### `majority-assign` in detail

For a type with a **non-empty** label set (100% label scenario):
- Finds all Neo4j nodes with exactly that label set (excluding `OriginalLabel`).
- Tallies `_orig_labels` across those nodes using a `Counter`.
- The most frequent `_orig_labels` value becomes the type's labels in the output schema.
- In the 100% case this is a no-op (all nodes in a `Person` type have `_orig_labels = ["Person"]`).

For a type with an **empty** label set (0% label scenario):
- Finds nodes that have only `OriginalLabel` (their real labels were stripped).
- Further filters by matching property keys to narrow the cluster.
- Majority label assignment maps the discovered unlabeled cluster back to a GT type.

### OriginalLabel is Invisible to the Inference Pipeline

A critical correctness requirement: `OriginalLabel` must not appear in the mined schema. If it did, every node would appear to have an extra label, corrupting all label-set comparisons.

**File:** `src/pg_schema_llm/io/neo4j_io.py`

Three guards were added:

1. **`_label_key()` filters internal labels** before building the label frozenset:
   ```python
   _INTERNAL_LABELS = frozenset({"OriginalLabel"})

   def _label_key(label_list) -> Tuple[str, ...]:
       return tuple(sorted(l for l in label_list if l not in _INTERNAL_LABELS))
   ```

2. **`_build_label_match_exact()` excludes internal labels from the size check.** Without this, a `Person` node that also carries `OriginalLabel` would have `size(labels(n)) = 2`, and a query checking `size = 1` for `["Person"]` would miss it entirely. The fix counts only non-internal labels:
   ```cypher
   size([l IN labels(n) WHERE NOT l IN ['OriginalLabel'] | l]) = 1
   ```

3. **`_props_key()` and the streaming property scan skip internal properties** (`_orig_labels`, `_orig_label_concat`, `_label_stripped`) so they never appear in the inferred schema's property lists.

### Typical Workflow

```bash
# Baseline (100% labels)
python scripts/label_noise.py backup
python scripts/infer.py pole
python scripts/label_noise.py majority-assign \
    03_outputs/schemas/inferred/pole/inf_pole.json
python scripts/compare.py pole

# 50% label availability
python scripts/label_noise.py apply 50
python scripts/infer.py pole
python scripts/label_noise.py majority-assign \
    03_outputs/schemas/inferred/pole/inf_pole.json
python scripts/compare.py pole

python scripts/label_noise.py restore  # reset for next scenario
python scripts/label_noise.py clean    # done — remove all metadata
```

---

## 4. Instance-Weighted F1\* (PG-HIVE Definition)

### Problem

The original F1 calculation in `compare.py` treated every type equally — each distinct label set counted as 1 unit regardless of how many nodes it contained. A type with 500,000 nodes had exactly the same weight as a type with 3 nodes. This is a **type-count F1**, not the metric the paper defines.

The PG-HIVE paper uses a **majority-based F1\* Score** (Section 5.2, citing [68]):

> *"the correctness of a node/edge placement is determined based on whether its actual type matches the majority label(s) of its cluster"*

Each *instance* (node or edge relationship) is a vote. A missed type with 100,000 nodes hurts far more than a missed type with 10 nodes. The paper computes instance-weighted precision and recall:

```
TP = sum of instance counts for matched types
FP = sum of instance counts for inferred-only types
FN = sum of instance counts for GT-only types

P* = TP / (TP + FP)
R* = TP / (TP + FN)
F1* = 2 · P* · R* / (P* + R*)
```

This applies at all four levels: node types, node patterns, edge types, edge patterns.

### Where the Counts Come From

The `_mined_patterns` key attached to every inferred schema JSON already contains exact instance counts for every node type and pattern, and every canonical edge type and pattern. These counts come from full-scan Cypher queries over the actual Neo4j graph, making them the authoritative ground truth.

**File:** `src/pg_schema_llm/pipeline/compare.py`

After loading both JSON files, four lookup dictionaries are built:

```python
mined_node_cnt:     Dict[LabelKey, int]        # node type → instance count
mined_node_pat_cnt: Dict[NodePatternKey, int]  # (L, K) pattern → count
mined_edge_cnt:     Dict[EdgeTypeKey, int]      # edge type → count
mined_edge_pat_cnt: Dict[EdgePatternKey, int]  # (L, K, R) pattern → count
```

Each metric section then sums instance counts instead of counting types:

```python
# Node types — before (type count):
nt_p, nt_r, nt_f1 = _prf(len(matched_nodes), len(inf_node_lks), len(gt_node_lks))

# Node types — after (instance weighted):
nt_tp = sum(_node_inst(lk) for lk in matched_nodes)
nt_fp = sum(_node_inst(lk) for lk in only_inf)
nt_fn = sum(_node_inst(lk) for lk in only_gt)
nt_p, nt_r, nt_f1 = _prf_inst(nt_tp, nt_fp, nt_fn)
```

A fallback of 1 is used for any type that appears in the GT but has no mined count (e.g. a type defined in a `.pgs` file that has zero actual instances in the graph). This avoids zero-weight phantom types silently inflating recall.

### Effect

For datasets with highly unequal type sizes, the instance-weighted F1\* can differ significantly from the old type-count F1. A single large missed edge type now appropriately dominates the score, rather than being diluted by many small correctly-recovered types.

For the mb6 dataset the edge F1\* is 98.69%, reflecting 24,838 relationship instances belonging to GT edge types the LLM did not recover — a loss that would have been invisible at the type-count level if those types were a small fraction of the total type count.

The four instance-weighted F1\* scores feed into the **Macro F1\*** = mean(node type F1\*, node pattern F1\*, edge type F1\*, edge pattern F1\*), which is now the headline metric of the pipeline.
