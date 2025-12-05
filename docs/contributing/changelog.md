# Changelog

All notable changes to GraphMem are documented here.

## [1.8.3] - 2024-12-05

### Added
- Comprehensive Agent Building Guide (`docs/AGENT_GUIDE.md`)
- Distributed Infrastructure Guide (`docs/DISTRIBUTED_INFRASTRUCTURE.md`)
- Better document validation error messages
- Shows type and preview of invalid documents

### Fixed
- Fixed batch ingestion validation for invalid documents

## [1.8.2] - 2024-12-05

### Added
- LLM-based entity consolidation for more accurate merging
- Priority-based conflict resolution in memory decay
- Fact order priority parsing from input text
- Concurrent evolution processing
- Full source chunk storage and retrieval

### Improved
- Enhanced extraction prompt for more exhaustive knowledge capture
- Better handling of aliases during entity resolution
- Improved community detection summaries

## [1.8.1] - 2024-12-04

### Added
- LLM-based temporal extraction
- LLM-based decay reasoning
- Exhaustive community summaries
- Continuous evolution in benchmarks

### Fixed
- Various indentation errors in evolution modules

## [1.8.0] - 2024-12-04

### Added
- Coreference resolution (second LLM pass)
- Alias extraction from text
- Temporal validity extraction from text
- Hybrid retrieval with alias matching
- LLM-based answer checking in benchmarks

### Improved
- Entity resolution with alias support
- Query engine prompts for alias/temporal awareness

## [1.7.0] - 2024-12-03

### Added
- MemoryAgentBench evaluation support
- Synthetic benchmark for GraphMem features
- Safe concurrent query processing

### Improved
- Evolution pipeline efficiency
- Batch ingestion performance

## [1.6.0] - 2024-12-02

### Added
- RAGAS-style evaluation metrics
- Skip-ingestion flag for evaluations
- Debug logging for predicted vs expected answers

### Fixed
- MultiHopRAG dataset handling
- Evaluation accuracy calculations

## [1.5.0] - 2024-12-01

### Added
- Comprehensive evaluation metrics
- Token efficiency tracking
- Memory growth monitoring
- Cost per query calculation

### Improved
- Evaluation script structure
- Profiling logs for bottleneck tracking

## [1.4.0] - 2024-11-30

### Added
- High-performance ingestion pipeline
- Batch embedding generation
- Async LLM extraction
- Auto-scaling worker configuration
- Benchmark utility for speedup measurement

### Improved
- Rate limit handling with infinite retry
- Worker configuration for different hardware

## [1.3.0] - 2024-11-29

### Added
- Turso (libSQL) storage backend
- Native vector search for Turso
- SQLite persistence support

### Improved
- Storage backend abstraction
- README with storage decision guide

## [1.2.0] - 2024-11-28

### Added
- PageRank centrality for importance scoring
- Temporal validity for relationships
- Point-in-time queries

### Improved
- Evolution engine with PageRank integration
- Query engine with temporal awareness

## [1.1.0] - 2024-11-27

### Added
- Multi-tenant isolation
- Redis caching layer
- User-scoped cache invalidation

### Improved
- Data isolation architecture
- Cache key structure

## [1.0.0] - 2024-11-25

### Initial Release
- Core GraphMem functionality
- Knowledge graph extraction
- Memory evolution (decay, consolidation)
- Semantic search
- Community detection
- Neo4j and in-memory storage

