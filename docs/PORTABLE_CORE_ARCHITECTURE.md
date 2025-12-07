# GraphMem Portable Core Architecture

## 🎯 Executive Summary for Rust Porting

This document separates GraphMem into:
1. **CORE** - Pure algorithms (portable to Rust/WASM)
2. **ADAPTERS** - I/O interfaces (traits/ports)
3. **INFRASTRUCTURE** - External services (workers/HTTP)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           GraphMem Architecture                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        INFRASTRUCTURE LAYER                              │ │
│  │                 (Lightweight Workers / Axum / CloudRun)                 │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │ │
│  │  │   LLM    │  │ Embedding│  │  Neo4j   │  │  Redis   │               │ │
│  │  │ Provider │  │ Provider │  │  Store   │  │  Cache   │               │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │ │
│  └───────┼──────────────┼──────────────┼──────────────┼─────────────────────┘ │
│          │              │              │              │                       │
│          ▼              ▼              ▼              ▼                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         ADAPTER LAYER (Traits/Ports)                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────────┐ │ │
│  │  │ LLMAdapter   │  │EmbedAdapter  │  │StorePort  │  │ CachePort      │ │ │
│  │  │ .complete()  │  │.embed_text() │  │.save()    │  │ .get()         │ │ │
│  │  │ .chat()      │  │.embed_batch()│  │.load()    │  │ .set()         │ │ │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  └───────┬────────┘ │ │
│  └─────────┼─────────────────┼────────────────┼────────────────┼──────────┘ │
│            │                 │                │                │            │
│            ▼                 ▼                ▼                ▼            │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                              CORE LAYER                                  │ │
│  │                    (Pure Rust / WASM Portable)                          │ │
│  │                                                                          │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                      DATA STRUCTURES                             │   │ │
│  │  │  MemoryNode | MemoryEdge | MemoryCluster | Memory | Query       │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                          │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │ │
│  │  │ ENTITY         │  │ EVOLUTION      │  │ RETRIEVAL              │   │ │
│  │  │ RESOLUTION     │  │ ENGINE         │  │ ENGINE                 │   │ │
│  │  │                │  │                │  │                        │   │ │
│  │  │ • Token Match  │  │ • Decay        │  │ • Semantic Search      │   │ │
│  │  │ • Fuzzy Match  │  │ • Consolidate  │  │ • Graph Traversal      │   │ │
│  │  │ • Alias Index  │  │ • Importance   │  │ • Ranking              │   │ │
│  │  │ • Cosine Sim   │  │ • PageRank     │  │ • Context Assembly     │   │ │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘   │ │
│  │                                                                          │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │ │
│  │  │ CHUNKING       │  │ COMMUNITY      │  │ QUERY                  │   │ │
│  │  │                │  │ DETECTION      │  │ PROCESSING             │   │ │
│  │  │ • Semantic     │  │                │  │                        │   │ │
│  │  │ • Paragraph    │  │ • Louvain      │  │ • Multi-hop            │   │ │
│  │  │ • Sentence     │  │ • Modularity   │  │ • Aggregation          │   │ │
│  │  │ • Code         │  │ • Summaries    │  │ • Confidence           │   │ │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘   │ │
│  │                                                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Current Python Module Structure

```
graphmem/src/graphmem/
├── core/                    # 🎯 CORE - Port to Rust
│   ├── memory_types.py      # Data structures (MemoryNode, MemoryEdge, etc.)
│   ├── memory.py            # Main orchestrator (uses adapters)
│   └── exceptions.py        # Error types
│
├── graph/                   # 🎯 CORE - Port to Rust
│   ├── knowledge_graph.py   # LLM-dependent extraction (needs adapter)
│   ├── entity_resolver.py   # ✅ PURE LOGIC (Jaccard, fuzzy, cosine)
│   └── community_detector.py # Uses NetworkX (needs petgraph in Rust)
│
├── evolution/               # 🎯 CORE - Port to Rust
│   ├── importance_scorer.py # ✅ PURE LOGIC (formula ρ(e) = w1·f1 + w2·f2 + ...)
│   ├── decay.py             # ✅ PURE LOGIC (exponential decay math)
│   ├── consolidation.py     # ✅ PURE LOGIC (entity merging algorithms)
│   ├── rehydration.py       # ✅ PURE LOGIC (archive restoration)
│   └── memory_evolution.py  # Orchestrator (calls above)
│
├── context/                 # 🎯 CORE - Port to Rust
│   ├── chunker.py           # ✅ PURE LOGIC (boundary detection, overlap)
│   ├── context_engine.py    # Context assembly
│   └── extractors.py        # Uses external libs (not core)
│
├── retrieval/               # 🎯 CORE - Port to Rust
│   ├── semantic_search.py   # ✅ PURE LOGIC (cosine similarity, ranking)
│   ├── retriever.py         # ✅ PURE LOGIC (graph traversal)
│   └── query_engine.py      # LLM-dependent answer gen (needs adapter)
│
├── llm/                     # 🔌 ADAPTER - Traits Only
│   ├── providers.py         # HTTP calls to LLM APIs
│   └── embeddings.py        # HTTP calls to embedding APIs
│
├── stores/                  # 🔌 ADAPTER - Traits Only
│   ├── memory_store.py      # In-memory store interface
│   ├── neo4j_store.py       # Neo4j driver calls
│   ├── turso_store.py       # SQLite/libSQL calls
│   └── redis_cache.py       # Redis calls
│
└── ingestion/               # 🏭 INFRASTRUCTURE - Workers
    ├── pipeline.py          # Concurrent processing
    ├── batch_embedder.py    # Batch API calls
    ├── async_extractor.py   # Async LLM calls
    └── auto_scale.py        # Hardware detection
```

---

## 🧠 CORE LAYER - Pure Algorithms (Portable to Rust/WASM)

### 1. Data Structures (`core/memory_types.py`)

```rust
// Rust equivalent data structures

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub description: Option<String>,
    pub canonical_name: Option<String>,
    pub aliases: HashSet<String>,
    pub embedding: Option<Vec<f32>>,  // 1536 dimensions typically
    pub importance: MemoryImportance,
    pub state: MemoryState,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub accessed_at: DateTime<Utc>,
    pub access_count: u64,
    pub user_id: Option<String>,      // Multi-tenant isolation
    pub memory_id: Option<String>,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation_type: String,
    pub description: Option<String>,
    pub weight: f64,                   // 0.1 to 10.0
    pub confidence: f64,               // 0.0 to 1.0
    pub importance: MemoryImportance,
    pub state: MemoryState,
    pub valid_from: Option<DateTime<Utc>>,  // Temporal validity
    pub valid_until: Option<DateTime<Utc>>,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryImportance {
    Critical = 10,    // Never decay
    VeryHigh = 8,
    High = 6,
    Medium = 5,       // Default
    Low = 3,
    VeryLow = 1,
    Ephemeral = 0,    // Immediate decay candidate
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryState {
    Active,
    Consolidating,
    Decaying,
    Archived,
    Deleted,
}

#[derive(Clone, Debug)]
pub struct Memory {
    pub id: String,
    pub nodes: HashMap<String, MemoryNode>,
    pub edges: HashMap<String, MemoryEdge>,
    pub clusters: HashMap<i32, MemoryCluster>,
}

#[derive(Clone, Debug)]
pub struct MemoryCluster {
    pub id: i32,
    pub summary: String,
    pub entities: Vec<String>,
    pub edges: Vec<String>,
    pub coherence_score: f64,
    pub density: f64,
}
```

---

### 2. Importance Scoring Algorithm (`evolution/importance_scorer.py`)

**PURE MATH - No I/O dependencies**

```rust
/// Importance scoring formula from paper:
/// ρ(e) = w1·f1(e) + w2·f2(e) + w3·f3(e) + w4·f4(e)
/// 
/// Where:
/// - f1 = Temporal recency (exponential decay)
/// - f2 = Access frequency (logarithmic scaling)
/// - f3 = PageRank centrality (graph structure)
/// - f4 = User/explicit importance

pub struct ImportanceScorer {
    pub recency_weight: f64,    // w1 = 0.3
    pub frequency_weight: f64,  // w2 = 0.3
    pub pagerank_weight: f64,   // w3 = 0.2
    pub user_weight: f64,       // w4 = 0.2
    pub pagerank_damping: f64,  // 0.85 (Google's original)
}

impl ImportanceScorer {
    /// f1: Recency score using exponential decay
    /// Formula: exp(-0.693 * age_days / half_life_days)
    pub fn recency_score(&self, accessed_at: DateTime<Utc>, now: DateTime<Utc>) -> f64 {
        let age_days = (now - accessed_at).num_seconds() as f64 / 86400.0;
        let half_life_days = 30.0;
        (-0.693 * age_days / half_life_days).exp()
    }
    
    /// f2: Frequency score with logarithmic saturation
    /// Formula: min(1.0, log(1 + access_count) / log(1 + saturation_point))
    pub fn frequency_score(&self, access_count: u64) -> f64 {
        let saturation_point = 100.0;
        f64::min(1.0, (1.0 + access_count as f64).ln() / (1.0 + saturation_point).ln())
    }
    
    /// f3: PageRank centrality
    /// Uses power iteration: PR(u) = (1-d)/N + d * Σ(PR(v)/L(v))
    pub fn pagerank_score(
        &self,
        node_id: &str,
        nodes: &[MemoryNode],
        edges: &[MemoryEdge],
    ) -> f64 {
        // Build adjacency list
        let mut graph: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for edge in edges {
            graph.entry(edge.source_id.clone())
                 .or_default()
                 .push((edge.target_id.clone(), edge.weight * edge.confidence));
        }
        
        // Power iteration
        let n = nodes.len() as f64;
        let mut scores: HashMap<String, f64> = nodes.iter()
            .map(|n| (n.id.clone(), 1.0 / n))
            .collect();
        
        for _ in 0..100 {  // max iterations
            let mut new_scores = HashMap::new();
            for node in nodes {
                let mut rank = (1.0 - self.pagerank_damping) / n;
                for other in nodes {
                    if let Some(edges) = graph.get(&other.id) {
                        for (target, weight) in edges {
                            if target == &node.id {
                                let out_degree: f64 = edges.iter().map(|(_, w)| w).sum();
                                rank += self.pagerank_damping * scores[&other.id] * weight / out_degree;
                            }
                        }
                    }
                }
                new_scores.insert(node.id.clone(), rank);
            }
            scores = new_scores;
        }
        
        // Normalize to [0, 1]
        let max_score = scores.values().cloned().fold(0.0, f64::max);
        scores.get(node_id).copied().unwrap_or(0.0) / max_score.max(1e-10)
    }
    
    /// Combined importance score: ρ(e) = Σ(wi * fi)
    pub fn score_node(
        &self,
        node: &MemoryNode,
        all_nodes: &[MemoryNode],
        all_edges: &[MemoryEdge],
        now: DateTime<Utc>,
    ) -> f64 {
        let f1 = self.recency_score(node.accessed_at, now);
        let f2 = self.frequency_score(node.access_count);
        let f3 = self.pagerank_score(&node.id, all_nodes, all_edges);
        let f4 = node.importance as u8 as f64 / 10.0;
        
        let score = self.recency_weight * f1
                  + self.frequency_weight * f2
                  + self.pagerank_weight * f3
                  + self.user_weight * f4;
        
        (score * 10.0).clamp(0.0, 10.0)
    }
}
```

---

### 3. Memory Decay Algorithm (`evolution/decay.py`)

**PURE MATH - Exponential decay with importance weighting**

```rust
pub struct MemoryDecay {
    pub half_life_days: f64,           // 30.0 default
    pub archive_threshold: f64,         // 0.2
    pub delete_threshold: f64,          // 0.05
    pub min_importance_to_keep: MemoryImportance,
}

impl MemoryDecay {
    /// Calculate current strength of a node
    /// Strength = exp(-0.693 * age_days / effective_half_life)
    /// Where effective_half_life = base_half_life * importance_factor * access_factor
    pub fn calculate_strength(&self, node: &MemoryNode, now: DateTime<Utc>) -> f64 {
        let age_days = (now - node.accessed_at).num_seconds() as f64 / 86400.0;
        
        // Importance modifier: 0.5 to 1.0
        let importance_factor = 0.5 + (node.importance as u8 as f64 / 20.0);
        
        // Access count modifier: 0.5 to 1.0
        let access_factor = f64::min(1.0, 0.5 + (1.0 + node.access_count as f64).ln() / 10.0);
        
        // Effective half-life
        let effective_half_life = self.half_life_days * importance_factor * access_factor;
        
        // Exponential decay
        (-0.693 * age_days / effective_half_life).exp()
    }
    
    /// Apply decay to all nodes, returning evolution events
    pub fn apply_decay(&self, memory: &mut Memory, now: DateTime<Utc>) -> Vec<EvolutionEvent> {
        let mut events = vec![];
        
        for (node_id, node) in memory.nodes.iter_mut() {
            // Skip critical memories
            if node.importance == MemoryImportance::Critical {
                continue;
            }
            
            let strength = self.calculate_strength(node, now);
            
            if strength <= self.delete_threshold {
                node.state = MemoryState::Deleted;
                events.push(EvolutionEvent::pruning(node_id.clone()));
            } else if strength <= self.archive_threshold {
                if node.importance < self.min_importance_to_keep {
                    node.state = MemoryState::Archived;
                    events.push(EvolutionEvent::decay(node_id.clone()));
                }
            }
        }
        
        events
    }
}
```

---

### 4. Entity Resolution Algorithm (`graph/entity_resolver.py`)

**PURE STRING/VECTOR OPERATIONS - No I/O**

```rust
pub struct EntityResolver {
    pub similarity_threshold: f64,  // 0.85
    pub token_threshold: f64,       // 0.7
    pub fuzzy_threshold: f64,       // 0.92
    
    entity_index: HashMap<String, EntityCandidate>,
    alias_lookup: HashMap<String, String>,  // alias -> canonical_key
}

#[derive(Clone)]
pub struct EntityCandidate {
    pub name: String,
    pub entity_type: String,
    pub tokens: HashSet<String>,
    pub embedding: Option<Vec<f32>>,
    pub aliases: HashSet<String>,
    pub occurrences: u32,
}

impl EntityResolver {
    /// Tokenize entity name (remove stopwords)
    fn tokenize(&self, name: &str) -> HashSet<String> {
        let stopwords: HashSet<&str> = ["the", "a", "an", "of", "and", "inc", "ltd", "corp"]
            .into_iter().collect();
        
        name.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 2 && !stopwords.contains(t))
            .map(String::from)
            .collect()
    }
    
    /// Jaccard similarity between token sets
    fn token_similarity(&self, a: &HashSet<String>, b: &HashSet<String>) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count() as f64;
        let union = a.union(b).count() as f64;
        intersection / union
    }
    
    /// Levenshtein-based fuzzy similarity (SequenceMatcher ratio)
    fn fuzzy_similarity(&self, a: &str, b: &str) -> f64 {
        let a = a.to_lowercase();
        let b = b.to_lowercase();
        
        // Use longest common subsequence ratio
        let lcs = self.longest_common_subsequence(&a, &b);
        2.0 * lcs as f64 / (a.len() + b.len()) as f64
    }
    
    fn longest_common_subsequence(&self, a: &str, b: &str) -> usize {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let mut dp = vec![vec![0; b.len() + 1]; a.len() + 1];
        
        for i in 1..=a.len() {
            for j in 1..=b.len() {
                if a[i-1] == b[j-1] {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = dp[i-1][j].max(dp[i][j-1]);
                }
            }
        }
        dp[a.len()][b.len()]
    }
    
    /// Cosine similarity between embedding vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
        let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
    
    /// Check if two entities are similar enough to merge
    pub fn are_similar(&self, a: &EntityCandidate, b: &EntityCandidate) -> bool {
        // 1. Exact name match
        if a.name.to_lowercase() == b.name.to_lowercase() {
            return true;
        }
        
        // 2. Alias overlap
        if !a.aliases.is_disjoint(&b.aliases) {
            return true;
        }
        
        // 3. Name containment
        let a_lower = a.name.to_lowercase();
        let b_lower = b.name.to_lowercase();
        if a_lower.contains(&b_lower) || b_lower.contains(&a_lower) {
            return true;
        }
        
        // 4. Token similarity
        let token_sim = self.token_similarity(&a.tokens, &b.tokens);
        if token_sim >= self.token_threshold {
            return true;
        }
        
        // 5. Fuzzy string similarity
        let fuzzy_sim = self.fuzzy_similarity(&a.name, &b.name);
        if fuzzy_sim >= self.fuzzy_threshold {
            return true;
        }
        
        // 6. Embedding similarity (if available)
        if let (Some(emb_a), Some(emb_b)) = (&a.embedding, &b.embedding) {
            let emb_sim = self.cosine_similarity(emb_a, emb_b);
            if emb_sim >= self.similarity_threshold {
                return true;
            }
            // Combined: token + embedding
            if token_sim >= 0.6 && emb_sim >= 0.85 {
                return true;
            }
        }
        
        false
    }
    
    /// Resolve entities to canonical forms
    pub fn resolve(&mut self, nodes: Vec<MemoryNode>) -> Vec<MemoryNode> {
        let mut resolved = vec![];
        
        for node in nodes {
            let key = self.generate_key(&node.name);
            
            // Check if matches existing
            if let Some(existing_key) = self.find_match(&node) {
                // Merge
                let existing = self.entity_index.get_mut(&existing_key).unwrap();
                existing.aliases.insert(node.name.clone());
                existing.occurrences += 1;
            } else {
                // New entity
                let candidate = EntityCandidate {
                    name: node.name.clone(),
                    entity_type: node.entity_type.clone(),
                    tokens: self.tokenize(&node.name),
                    embedding: node.embedding.clone(),
                    aliases: [node.name.clone()].into_iter().collect(),
                    occurrences: 1,
                };
                self.entity_index.insert(key.clone(), candidate);
                self.alias_lookup.insert(node.name.to_lowercase(), key);
                resolved.push(node);
            }
        }
        
        resolved
    }
    
    fn generate_key(&self, name: &str) -> String {
        name.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .take(64)
            .collect()
    }
    
    fn find_match(&self, node: &MemoryNode) -> Option<String> {
        // Fast path: alias lookup
        if let Some(key) = self.alias_lookup.get(&node.name.to_lowercase()) {
            return Some(key.clone());
        }
        
        // Slow path: compare with all candidates
        let tokens = self.tokenize(&node.name);
        for (key, candidate) in &self.entity_index {
            if candidate.entity_type != node.entity_type {
                continue;
            }
            
            let new_candidate = EntityCandidate {
                name: node.name.clone(),
                entity_type: node.entity_type.clone(),
                tokens: tokens.clone(),
                embedding: node.embedding.clone(),
                aliases: HashSet::new(),
                occurrences: 1,
            };
            
            if self.are_similar(candidate, &new_candidate) {
                return Some(key.clone());
            }
        }
        
        None
    }
}
```

---

### 5. Document Chunking (`context/chunker.py`)

**PURE STRING OPERATIONS - No I/O**

```rust
pub struct DocumentChunker {
    pub chunk_size: usize,       // 1000 chars
    pub chunk_overlap: usize,    // 200 chars
    pub min_chunk_size: usize,   // 100 chars
    pub respect_sentences: bool,
    pub respect_paragraphs: bool,
}

pub struct DocumentChunk {
    pub id: String,
    pub content: String,
    pub chunk_index: usize,
    pub start_char: usize,
    pub end_char: usize,
}

impl DocumentChunker {
    /// Split text into semantic chunks
    pub fn chunk_text(&self, text: &str) -> Vec<DocumentChunk> {
        if text.trim().is_empty() {
            return vec![];
        }
        
        // Try paragraph-based chunking first
        if self.respect_paragraphs {
            let paragraphs: Vec<&str> = text.split("\n\n").collect();
            if paragraphs.len() > 1 {
                return self.chunk_by_units(&paragraphs, text);
            }
        }
        
        // Fall back to sentence-based
        if self.respect_sentences {
            let sentences = self.split_sentences(text);
            if sentences.len() > 1 {
                return self.chunk_by_units(&sentences, text);
            }
        }
        
        // Fall back to character-based
        self.chunk_by_characters(text)
    }
    
    fn split_sentences(&self, text: &str) -> Vec<&str> {
        // Regex: (?<=[.!?])\s+(?=[A-Z])
        let mut result = vec![];
        let mut start = 0;
        
        for (i, c) in text.char_indices() {
            if ".!?".contains(c) {
                // Look ahead for whitespace + uppercase
                if let Some(next) = text[i+1..].chars().next() {
                    if next.is_whitespace() {
                        if let Some(after_space) = text[i+2..].chars().next() {
                            if after_space.is_uppercase() {
                                result.push(&text[start..=i+1]);
                                start = i + 2;
                            }
                        }
                    }
                }
            }
        }
        if start < text.len() {
            result.push(&text[start..]);
        }
        result
    }
    
    fn chunk_by_units(&self, units: &[&str], _original: &str) -> Vec<DocumentChunk> {
        let mut chunks = vec![];
        let mut current = String::new();
        let mut current_start = 0;
        
        for unit in units {
            let unit = unit.trim();
            if unit.is_empty() {
                continue;
            }
            
            if current.len() + unit.len() + 1 > self.chunk_size && !current.is_empty() {
                // Save current chunk
                chunks.push(DocumentChunk {
                    id: uuid::Uuid::new_v4().to_string(),
                    content: current.trim().to_string(),
                    chunk_index: chunks.len(),
                    start_char: current_start,
                    end_char: current_start + current.len(),
                });
                
                // Overlap
                let overlap = self.get_overlap(&current);
                current_start = current_start + current.len() - overlap.len();
                current = if overlap.is_empty() {
                    unit.to_string()
                } else {
                    format!("{} {}", overlap, unit)
                };
            } else {
                current = if current.is_empty() {
                    unit.to_string()
                } else {
                    format!("{} {}", current, unit)
                };
            }
        }
        
        if !current.trim().is_empty() {
            chunks.push(DocumentChunk {
                id: uuid::Uuid::new_v4().to_string(),
                content: current.trim().to_string(),
                chunk_index: chunks.len(),
                start_char: current_start,
                end_char: current_start + current.len(),
            });
        }
        
        chunks
    }
    
    fn get_overlap(&self, text: &str) -> String {
        if text.len() <= self.chunk_overlap {
            return text.to_string();
        }
        
        let overlap = &text[text.len() - self.chunk_overlap..];
        
        // Try to start at sentence boundary
        if let Some(pos) = overlap.find(". ") {
            return overlap[pos+2..].to_string();
        }
        
        // Try word boundary
        if let Some(pos) = overlap.find(' ') {
            return overlap[pos+1..].to_string();
        }
        
        overlap.to_string()
    }
    
    fn chunk_by_characters(&self, text: &str) -> Vec<DocumentChunk> {
        let mut chunks = vec![];
        let mut start = 0;
        
        while start < text.len() {
            let mut end = (start + self.chunk_size).min(text.len());
            
            // Find good break point
            if end < text.len() {
                for i in (start + self.min_chunk_size..end).rev() {
                    let c = text.chars().nth(i - 1).unwrap();
                    if ".!?".contains(c) {
                        if text.chars().nth(i).map_or(true, |c| c.is_whitespace()) {
                            end = i;
                            break;
                        }
                    }
                }
            }
            
            let chunk_text = text[start..end].trim();
            if !chunk_text.is_empty() {
                chunks.push(DocumentChunk {
                    id: uuid::Uuid::new_v4().to_string(),
                    content: chunk_text.to_string(),
                    chunk_index: chunks.len(),
                    start_char: start,
                    end_char: end,
                });
            }
            
            start = if end < text.len() {
                end - self.chunk_overlap
            } else {
                end
            };
        }
        
        chunks
    }
}
```

---

### 6. Semantic Search (`retrieval/semantic_search.py`)

**PURE VECTOR OPERATIONS - Adapter for embeddings**

```rust
pub struct SemanticSearch {
    index: HashMap<String, Vec<f32>>,        // node_id -> embedding
    node_lookup: HashMap<String, MemoryNode>, // node_id -> node
    top_k: usize,
    min_similarity: f64,
}

impl SemanticSearch {
    /// Index nodes for search
    pub fn index_nodes(&mut self, nodes: &[MemoryNode]) {
        for node in nodes {
            if let Some(embedding) = &node.embedding {
                self.index.insert(node.id.clone(), embedding.clone());
                self.node_lookup.insert(node.id.clone(), node.clone());
            }
        }
    }
    
    /// Search with query embedding (embedding comes from adapter)
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: Option<usize>,
        min_similarity: Option<f64>,
    ) -> Vec<(MemoryNode, f64)> {
        let top_k = top_k.unwrap_or(self.top_k);
        let min_similarity = min_similarity.unwrap_or(self.min_similarity);
        
        let mut results: Vec<(MemoryNode, f64)> = self.index.iter()
            .filter_map(|(node_id, node_embedding)| {
                let similarity = self.cosine_similarity(query_embedding, node_embedding);
                
                if similarity >= min_similarity {
                    if let Some(node) = self.node_lookup.get(node_id) {
                        // Skip EPHEMERAL nodes (decayed)
                        if node.importance == MemoryImportance::Ephemeral {
                            return None;
                        }
                        
                        // Apply importance weighting
                        let importance_weight = node.importance as u8 as f64 / 10.0;
                        let combined_score = 0.7 * similarity + 0.3 * importance_weight;
                        
                        return Some((node.clone(), combined_score));
                    }
                }
                None
            })
            .collect();
        
        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
    
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
        let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }
}
```

---

### 7. Community Detection (`graph/community_detector.py`)

**USES GRAPH ALGORITHMS - Needs petgraph in Rust**

```rust
use petgraph::graph::UnGraph;
use petgraph::algo::kosaraju_scc;

pub struct CommunityDetector {
    max_cluster_size: usize,
    min_cluster_size: usize,
}

impl CommunityDetector {
    /// Detect communities using modularity-based algorithm
    /// (Louvain or Greedy Modularity equivalent)
    pub fn detect(
        &self,
        nodes: &[MemoryNode],
        edges: &[MemoryEdge],
    ) -> Vec<HashSet<String>> {
        // Build petgraph
        let mut graph = UnGraph::<&str, f64>::new_undirected();
        let mut node_indices = HashMap::new();
        
        for node in nodes {
            let idx = graph.add_node(&node.id);
            node_indices.insert(&node.id, idx);
        }
        
        for edge in edges {
            if let (Some(&src), Some(&tgt)) = (
                node_indices.get(&edge.source_id),
                node_indices.get(&edge.target_id),
            ) {
                graph.add_edge(src, tgt, edge.weight * edge.confidence);
            }
        }
        
        // For Rust, use community detection from petgraph or custom implementation
        // This is a simplified version using connected components
        // For production, implement Louvain algorithm
        self.greedy_modularity_communities(&graph, &node_indices)
    }
    
    fn greedy_modularity_communities(
        &self,
        graph: &UnGraph<&str, f64>,
        node_indices: &HashMap<&String, petgraph::graph::NodeIndex>,
    ) -> Vec<HashSet<String>> {
        // Simplified: use connected components as starting point
        // Real implementation should use modularity optimization
        
        let mut communities = vec![];
        let mut visited = HashSet::new();
        
        for &node_idx in node_indices.values() {
            if visited.contains(&node_idx) {
                continue;
            }
            
            let mut community = HashSet::new();
            let mut stack = vec![node_idx];
            
            while let Some(idx) = stack.pop() {
                if visited.insert(idx) {
                    community.insert(graph[idx].to_string());
                    
                    for neighbor in graph.neighbors(idx) {
                        if !visited.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
            
            if community.len() >= self.min_cluster_size {
                // Split large communities
                if community.len() > self.max_cluster_size {
                    for chunk in community.into_iter()
                        .collect::<Vec<_>>()
                        .chunks(self.max_cluster_size) 
                    {
                        communities.push(chunk.iter().cloned().collect());
                    }
                } else {
                    communities.push(community);
                }
            }
        }
        
        communities
    }
}
```

---

## 🔌 ADAPTER LAYER - Traits/Ports (Interface Definitions)

These are the interfaces that external services must implement:

```rust
/// LLM Adapter - For completion and chat
pub trait LLMAdapter: Send + Sync {
    fn complete(&self, prompt: &str, temperature: f64, max_tokens: usize) -> Result<String>;
    fn chat(&self, messages: Vec<Message>, temperature: f64, max_tokens: usize) -> Result<String>;
}

/// Embedding Adapter - For vector generation
pub trait EmbeddingAdapter: Send + Sync {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Storage Adapter - For persistence
pub trait StorageAdapter: Send + Sync {
    fn save_memory(&self, memory: &Memory) -> Result<()>;
    fn load_memory(&self, memory_id: &str, user_id: Option<&str>) -> Result<Option<Memory>>;
    fn delete_memory(&self, memory_id: &str) -> Result<()>;
    fn list_memories(&self) -> Result<Vec<String>>;
}

/// Cache Adapter - For caching
pub trait CacheAdapter: Send + Sync {
    fn get(&self, key: &str) -> Option<Vec<u8>>;
    fn set(&self, key: &str, value: &[u8], ttl_secs: u64) -> Result<()>;
    fn delete(&self, key: &str) -> Result<()>;
    fn invalidate_pattern(&self, pattern: &str) -> Result<()>;
}
```

---

## 🏭 INFRASTRUCTURE LAYER - Workers/Services

This layer handles I/O and should run on lightweight workers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INFRASTRUCTURE SERVICES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        INGESTION WORKERS                              │  │
│  │                    (Axum / CloudRun / Fluvio)                        │  │
│  │                                                                       │  │
│  │  HTTP Request                                                         │  │
│  │      ↓                                                                │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │  │
│  │  │ Chunker         │ →  │ Embedding API   │ →  │ LLM Extraction  │  │  │
│  │  │ (WASM Core)     │    │ (HTTP Client)   │    │ (HTTP Client)   │  │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘  │  │
│  │                              ↓                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                     MESSAGE QUEUE (Redpanda/Fluvio)             │ │  │
│  │  │   ingest_queue | embed_queue | extract_queue | evolve_queue    │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        QUERY WORKERS                                  │  │
│  │                                                                       │  │
│  │  HTTP Request                                                         │  │
│  │      ↓                                                                │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │  │
│  │  │ Semantic Search │ →  │ Graph Traversal │ →  │ LLM Answer Gen  │  │  │
│  │  │ (WASM Core)     │    │ (WASM Core)     │    │ (HTTP Client)   │  │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        EVOLUTION WORKERS                              │  │
│  │                                                                       │  │
│  │  Scheduled / Triggered                                                │  │
│  │      ↓                                                                │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │  │
│  │  │ Decay Engine    │    │ Consolidation   │    │ Community Det   │  │  │
│  │  │ (WASM Core)     │    │ (WASM + LLM)    │    │ (WASM Core)     │  │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow - End to End

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INGESTION FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Document                                                                   │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────┐  CORE    ┌─────────────┐  ADAPTER   ┌─────────────────┐   │
│  │ DocumentChunker │───────→│ Chunks      │──────────→│ EmbeddingAdapter │   │
│  │ (Pure Rust) │          └─────────────┘           │ (HTTP Call)      │   │
│  └─────────────┘                                     └────────┬────────┘   │
│                                                                │            │
│                                                                ▼            │
│  ┌─────────────┐  ADAPTER ┌─────────────┐  CORE      ┌─────────────────┐   │
│  │ LLMAdapter  │←─────────│ Chunks+Emb  │──────────→│ EntityResolver  │   │
│  │ (HTTP Call) │          └─────────────┘           │ (Pure Rust)     │   │
│  └──────┬──────┘                                     └────────┬────────┘   │
│         │                                                      │            │
│         ▼                                                      ▼            │
│  ┌─────────────┐  CORE    ┌─────────────┐  ADAPTER   ┌─────────────────┐   │
│  │ Entities    │──────────│ Memory      │──────────→│ StorageAdapter  │   │
│  │ + Edges     │          │ (Graph)     │           │ (Neo4j/Turso)   │   │
│  └─────────────┘          └─────────────┘           └─────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            QUERY FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Query                                                                      │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────┐  ADAPTER ┌─────────────┐  CORE      ┌─────────────────┐   │
│  │ EmbeddingAdapter │──────│ Query+Emb   │──────────→│ SemanticSearch  │   │
│  │ (HTTP Call) │          └─────────────┘           │ (Pure Rust)     │   │
│  └─────────────┘                                     └────────┬────────┘   │
│                                                                │            │
│                                                                ▼            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     GRAPH TRAVERSAL (CORE)                           │   │
│  │                                                                       │   │
│  │   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │   │
│  │   │ Initial     │────→│ Multi-hop   │────→│ Context Assembly    │   │   │
│  │   │ Retrieval   │     │ Expansion   │     │ + Ranking           │   │   │
│  │   └─────────────┘     └─────────────┘     └─────────────────────┘   │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                           │                                  │
│                                           ▼                                  │
│  ┌─────────────┐  ADAPTER ┌─────────────┐         ┌─────────────────────┐  │
│  │ LLMAdapter  │←─────────│ Context     │────────→│ Answer + Confidence │  │
│  │ (HTTP Call) │          │ + Query     │         └─────────────────────┘  │
│  └─────────────┘          └─────────────┘                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVOLUTION FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Memory                                                                     │
│      │                                                                       │
│      ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        PARALLEL EXECUTION                              │ │
│  │                                                                         │ │
│  │  ┌───────────────────────┐  ┌───────────────────────┐                 │ │
│  │  │     DECAY ENGINE      │  │    CONSOLIDATION      │                 │ │
│  │  │      (CORE)           │  │    (CORE + LLM)       │                 │ │
│  │  │                       │  │                       │                 │ │
│  │  │ • Strength calc       │  │ • Entity grouping     │                 │ │
│  │  │ • State transitions   │  │ • LLM duplicate check │                 │ │
│  │  │ • Archive/delete      │  │ • Merge nodes         │                 │ │
│  │  │ • Priority conflict   │  │ • Edge reinforcement  │                 │ │
│  │  └───────────────────────┘  └───────────────────────┘                 │ │
│  │                                                                         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────┐  CORE      ┌─────────────┐  ADAPTER  ┌─────────────────┐  │
│  │ Community   │───────────→│ Updated     │─────────→│ StorageAdapter  │  │
│  │ Detection   │            │ Memory      │          │ (Persist)       │  │
│  └─────────────┘            └─────────────┘          └─────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🦀 Rust Crate Structure

```
graphmem-core/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── types/              # Data structures
│   │   ├── mod.rs
│   │   ├── node.rs         # MemoryNode
│   │   ├── edge.rs         # MemoryEdge
│   │   ├── cluster.rs      # MemoryCluster
│   │   ├── memory.rs       # Memory
│   │   └── query.rs        # MemoryQuery, MemoryResponse
│   │
│   ├── evolution/          # Pure algorithms
│   │   ├── mod.rs
│   │   ├── importance.rs   # ImportanceScorer
│   │   ├── decay.rs        # MemoryDecay
│   │   ├── consolidation.rs # Consolidation (pure parts)
│   │   └── rehydration.rs
│   │
│   ├── graph/              # Graph algorithms
│   │   ├── mod.rs
│   │   ├── entity_resolver.rs
│   │   ├── community.rs    # Uses petgraph
│   │   └── traversal.rs    # Multi-hop
│   │
│   ├── retrieval/          # Search algorithms
│   │   ├── mod.rs
│   │   ├── semantic_search.rs
│   │   ├── ranking.rs
│   │   └── context.rs
│   │
│   ├── chunking/           # Text processing
│   │   ├── mod.rs
│   │   ├── document.rs
│   │   ├── markdown.rs
│   │   └── code.rs
│   │
│   └── adapters/           # Trait definitions only
│       ├── mod.rs
│       ├── llm.rs          # trait LLMAdapter
│       ├── embeddings.rs   # trait EmbeddingAdapter
│       ├── storage.rs      # trait StorageAdapter
│       └── cache.rs        # trait CacheAdapter
│
└── tests/
    ├── entity_resolver_tests.rs
    ├── importance_tests.rs
    ├── decay_tests.rs
    └── search_tests.rs
```

---

## 🎯 Porting Priority

### Phase 1: Pure Core (No I/O)
1. **types/** - Data structures (serde derives for JSON compat)
2. **chunking/** - Document chunker
3. **evolution/importance.rs** - Scoring formula
4. **evolution/decay.rs** - Decay math
5. **graph/entity_resolver.rs** - Token/fuzzy/cosine matching

### Phase 2: Graph Algorithms
6. **graph/community.rs** - Use petgraph
7. **graph/traversal.rs** - BFS/DFS for multi-hop
8. **retrieval/semantic_search.rs** - Vector search
9. **retrieval/ranking.rs** - Score combination

### Phase 3: LLM-Dependent (Keep in Workers)
10. **extraction.rs** - LLM calls for entity extraction
11. **answer_gen.rs** - LLM calls for answer generation
12. **consolidation.rs** (LLM parts) - Smart entity matching

---

## 🔗 Integration with Workers

```rust
// In Axum worker
use graphmem_core::{DocumentChunker, EntityResolver, SemanticSearch, ImportanceScorer};
use graphmem_core::adapters::{LLMAdapter, EmbeddingAdapter};

// Infrastructure implementations
struct OpenAIEmbeddings { client: reqwest::Client, api_key: String }
struct AzureOpenAI { client: reqwest::Client, endpoint: String }

impl EmbeddingAdapter for OpenAIEmbeddings {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // HTTP call to OpenAI
    }
}

impl LLMAdapter for AzureOpenAI {
    fn complete(&self, prompt: &str, temp: f64, max: usize) -> Result<String> {
        // HTTP call to Azure
    }
}

// Handler
async fn ingest(State(state): State<AppState>, body: String) -> Result<Json<Memory>> {
    // 1. Pure core: chunk
    let chunks = state.chunker.chunk_text(&body);
    
    // 2. Adapter: get embeddings
    let embeddings = state.embeddings.embed_batch(&chunks.iter().map(|c| c.content.clone()).collect::<Vec<_>>()).await?;
    
    // 3. Adapter: LLM extraction
    let entities = state.llm.extract_entities(&body).await?;
    
    // 4. Pure core: resolve entities
    let resolved = state.resolver.resolve(entities);
    
    // 5. Pure core: build memory
    let memory = Memory::new().with_nodes(resolved);
    
    // 6. Adapter: persist
    state.storage.save_memory(&memory).await?;
    
    Ok(Json(memory))
}
```

---

## 📋 Summary

| Layer | What | Where to Run | Port to Rust? |
|-------|------|--------------|---------------|
| **CORE** | Data structures, algorithms | WASM | ✅ Yes - First priority |
| **ADAPTERS** | Trait definitions | Anywhere | ✅ Yes - Interface only |
| **INFRASTRUCTURE** | HTTP clients, storage drivers | Axum/CloudRun | ⚠️ Rust implementation (not WASM) |

The **CORE** is ~70% of the value and is pure computation:
- Entity resolution (Jaccard, fuzzy, cosine)
- Importance scoring (PageRank, decay formulas)
- Memory decay (exponential math)
- Document chunking (string parsing)
- Semantic search (vector math)
- Graph traversal (BFS/DFS)

The **INFRASTRUCTURE** is I/O bound and should run on workers:
- LLM API calls
- Embedding API calls
- Database operations
- Message queue operations

