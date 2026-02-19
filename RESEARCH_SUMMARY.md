# Hierarchical Adaptive Encoding for LLM Agent Communication

## Research Problem
LLM agents communicating with natural language face token inefficiency. Existing compression approaches either:
- Lose semantic fidelity (traditional compression)
- Don't scale to longer messages (simple summarization)
- Require training data (ML-based approaches)

## Novel Contributions

### 1. Two-Tier Hierarchical Architecture
- **Tier 1 (Summary)**: High-level structured metadata (intent, target, key metrics)
- **Tier 2 (Sections)**: Detailed content sections preserving full information
- Enables both compression and semantic preservation

### 2. Importance-Weighted Preservation
- Analyzes message structure to identify section importance (critical/high/medium/low)
- Preserves MORE detail in critical sections, compresses less important sections
- Adaptive compression depth based on content importance

### 3. Iterative Refinement with Semantic Feedback
- Semantic judge evaluates decoded output vs original
- Identifies missing concepts and information loss
- Encoder re-encodes with focus on missing information
- Closed-loop optimization until target similarity reached

### 4. Multi-Pass Encoding Strategy
- **Pass 1**: Structure analysis and importance assessment
- **Pass 2**: Importance-weighted encoding
- **Pass 3+**: Iterative refinement based on semantic feedback

## Results

### Long Message Test (1763 tokens)
- **Original**: 1763 tokens
- **Compressed**: 1295 tokens (73.5% of original)
- **Semantic Similarity**: 80.6% (target: 80%)
- **Iterations**: 2
- **Sections Identified**: 8 (3 critical, 2 high, 3 medium importance)

### Iteration Breakdown
| Iteration | Tokens | Compression | Similarity |
|-----------|--------|-------------|------------|
| 1         | 778    | 44.1%       | 79.7%      |
| 2         | 1295   | 73.5%       | 80.6%      | ✅

### Key Insight
First iteration compressed too aggressively (44% → 79.7% similarity). Feedback loop identified missing critical information. Second iteration relaxed compression on critical sections, achieving target similarity.

## Technical Architecture

```
Input Message (1763 tokens)
    ↓
[Structure Analysis]
    ↓
Importance Assessment
- 3 critical sections
- 2 high sections  
- 3 medium sections
    ↓
[Importance-Weighted Encoding]
    ↓
Compressed Signal (778 tokens)
    ↓
[Decode & Judge]
    ↓
Similarity: 79.7% ❌
    ↓
[Feedback Analysis]
Missing: specific numbers, deadlines, technical details
    ↓
[Refinement Encoding]
Focus on critical sections
    ↓
Compressed Signal (1295 tokens)
    ↓
[Decode & Judge]
    ↓
Similarity: 80.6% ✅
```

## Advantages Over Existing Approaches

### vs Traditional Compression (gzip, etc)
- ✅ Preserves semantic meaning
- ✅ Human-readable intermediate format
- ✅ Structured for agent processing

### vs Simple Summarization
- ✅ Scales to long messages (>1500 tokens)
- ✅ Preserves critical details
- ✅ Adaptive compression based on importance

### vs ML-Based Compression (BART, T5, etc)
- ✅ No training data required
- ✅ Analytical/algorithmic approach
- ✅ Explainable (can see what's preserved/compressed)
- ✅ Iterative refinement with feedback

### vs Graph-Based Approaches
- ✅ Preserves narrative structure
- ✅ Works for both structured data AND narrative text
- ✅ Maintains conversational context

## Implementation Details

### Components
1. **HierarchicalAdaptiveEncoder**: Main encoding logic with multi-pass strategy
2. **SemanticJudge**: Embedding-based similarity evaluation (sentence-transformers)
3. **MSPDecoder**: Structured signal → natural language reconstruction
4. **GroqClient**: LLM inference (Llama 3.3 70B)

### Key Prompts
- Structure extraction prompt (identifies sections and importance)
- Importance-weighted encoding prompt (preserves detail based on importance)
- Refinement prompt (addresses missing information from feedback)

### No Training Required
- Uses LLM prompting for extraction and encoding
- Analytical importance assessment (no ML training)
- Embedding-based similarity (pre-trained model)

## Future Work

1. **Adaptive Section Granularity**: Automatically determine optimal section breakdown
2. **Cross-Message Context**: Preserve context across multi-turn conversations
3. **Domain-Specific Importance**: Learn importance patterns for specific domains
4. **Streaming Encoding**: Process very long messages in chunks
5. **Multi-Agent Optimization**: Optimize for specific agent pairs

## Reproducibility

```bash
# Install dependencies
poetry install

# Run test
poetry run python test_hierarchical_long_message.py

# Results saved to: hierarchical_test_results.json
```

## Code Location
- Main implementation: `src/minimal_signaling/encoding/hierarchical_adaptive_encoder.py`
- Test script: `test_hierarchical_long_message.py`
- Results: `hierarchical_test_results.json`

---

**Key Takeaway**: By combining hierarchical structure, importance-weighted preservation, and iterative semantic feedback, we achieve 80%+ semantic similarity while compressing long messages, without requiring training data or losing narrative structure.
