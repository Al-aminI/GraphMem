# Memory Evolution

GraphMem's evolution system mimics how human memory works—consolidating, forgetting, and improving over time.

## The Evolution Cycle

```python
memory.evolve()  # Trigger evolution
```

This single call performs:

| Stage | Description | Human Equivalent |
|-------|-------------|------------------|
| 1. PageRank | Recalculate importance scores | Synaptic strengthening |
| 2. Decay | Fade unused memories | Forgetting curve |
| 3. Consolidation | Merge similar memories | Sleep consolidation |
| 4. Rehydration | Resolve conflicts | Memory updating |

---

## 1. PageRank Centrality

Identifies "hub" entities that are well-connected.

### Formula

```
PR(A) = (1-d) + d × Σ(PR(Ti)/C(Ti))

where:
  d = 0.85 (damping factor)
  Ti = pages linking to A
  C(Ti) = number of outbound links from Ti
```

### Example

```
Elon Musk ─────┬───→ Tesla ───→ Austin
   (HUB)       ├───→ SpaceX ───→ Hawthorne
   PR=1.00     └───→ Neuralink

PageRank Scores:
  Elon Musk (1.00) > Tesla (0.77) > SpaceX (0.77) > Austin (0.52)
```

Hub entities are:
- Retrieved more often
- Weighted higher in answers
- Protected from decay

---

## 2. Memory Decay

Unused memories fade over time, following the Ebbinghaus forgetting curve.

### Formula

```
importance(t) = importance_0 × e^(-λ × (t - last_access))

where:
  λ = decay rate (configurable)
  t = current time
  last_access = when the memory was last used
```

### Configuration

```python
config = MemoryConfig(
    decay_enabled=True,
    decay_half_life_days=30,  # How fast memories fade
)
```

### What Gets Decayed

| Importance Level | Decay Behavior |
|------------------|----------------|
| CRITICAL | Never decays |
| VERY_HIGH | Very slow decay |
| HIGH | Slow decay |
| MEDIUM | Normal decay |
| LOW | Fast decay |

### Accessing Prevents Decay

When you query something, it gets "accessed" and its decay timer resets:

```python
# Day 1: Ingest
memory.ingest("Alice works at Google")

# Day 30: Without access, this might have decayed
# But:
memory.query("Where does Alice work?")  # Resets decay!
```

---

## 3. Consolidation

Similar memories are merged into stronger, unified memories.

### How It Works

```python
# Multiple mentions of the same thing:
memory.ingest("User likes Python")
memory.ingest("User prefers Python for AI")
memory.ingest("User's favorite language is Python")
memory.ingest("User codes in Python daily")
memory.ingest("Python is user's go-to language")

# After evolution:
memory.evolve()

# Result: ONE strong memory about user + Python
# 80% reduction in storage, same information!
```

### Consolidation Threshold

```python
config = MemoryConfig(
    consolidation_threshold=0.85,  # 0.0 - 1.0
)
```

- **Higher** (0.9+): Only merge very similar entities
- **Lower** (0.7-): More aggressive merging

### LLM-Based Merging

GraphMem uses LLM to intelligently merge:

```
Before: "Dr. Chen", "Alexander Chen", "Alex Chen"
After:  "Dr. Alexander Chen" (aliases: ["Alex Chen"])
```

---

## 4. Conflict Resolution (Rehydration)

When facts conflict, newer information wins.

### Example

```python
# Initial fact
memory.ingest("The company has 1,000 employees.")

# Updated fact (later)
memory.ingest("The company now has 2,500 employees after expansion.")

# Evolution resolves the conflict
memory.evolve()

# Query returns the newer information
memory.query("How many employees?")  # → "2,500"
```

### Priority-Based Resolution

Facts have priorities based on:
1. **Recency**: Newer facts win
2. **Explicit markers**: "UPDATE:", "CORRECTION:", "AS OF 2024"
3. **Importance level**: Higher importance wins

---

## Importance Scoring

The overall importance formula combines multiple factors:

```
ρ(e) = w1·f1 + w2·f2 + w3·f3 + w4·f4

where:
  f1 = Temporal recency    (0.3) - Recent = important
  f2 = Access frequency    (0.3) - Used often = important  
  f3 = PageRank centrality (0.2) - Well-connected = important
  f4 = User signal         (0.2) - Explicit importance
```

### Setting Importance

```python
from graphmem import MemoryImportance

# Critical information (never decays)
memory.ingest(
    "Customer's allergies: peanuts, shellfish",
    importance=MemoryImportance.CRITICAL,
)

# Normal information (may decay)
memory.ingest(
    "Customer mentioned they like coffee",
    importance=MemoryImportance.LOW,
)
```

---

## When to Evolve

### Auto-Evolution

```python
config = MemoryConfig(
    auto_evolve=True,  # Evolve after each ingestion
)
```

### Manual Evolution

```python
# After batch ingestion
memory.ingest_batch(documents)
memory.evolve()

# On a schedule
import schedule
schedule.every().day.at("02:00").do(memory.evolve)
```

### Evolution Events

```python
events = memory.evolve()
for event in events:
    print(f"{event.evolution_type}: {event.description}")
    
# Output:
# CONSOLIDATION: Merged 3 mentions of "Anthropic" into 1 entity
# DECAY: Archived 2 outdated relationships
# SYNTHESIS: Created new community summary for "AI Companies"
```

---

## Best Practices

1. **Evolve regularly** - At least daily for active systems
2. **Set appropriate importance** - Critical data shouldn't decay
3. **Tune consolidation threshold** - Based on your domain
4. **Monitor evolution events** - Track what's being merged/decayed

