# Hierarchical Adaptive Encoding - Architecture Diagram

## Complete System Architecture with Three-Phase Process

```mermaid
graph TB
    subgraph "Agent A"
        A[Natural Language Message<br/>2209 tokens]
    end
    
    subgraph "HIERARCHICAL ADAPTIVE ENCODER"
        subgraph "PHASE 1: Structure Analysis"
            B[Structure Analyzer<br/>LLM]
            B --> C[Section Identification]
            C --> D[Importance Rating<br/>critical/high/medium/low]
            D --> E[Key Concepts Extraction]
            E --> F[Importance Map]
        end
        
        subgraph "PHASE 2: Organic Compression"
            F --> G[Importance-Weighted<br/>Compression Rules]
            G --> H1[Critical: Preserve ALL facts]
            G --> H2[High: Keep main points + details]
            G --> H3[Medium: Summarize with key facts]
            G --> H4[Low: Brief summary]
            H1 --> I[Compressed Encoder<br/>LLM]
            H2 --> I
            H3 --> I
            H4 --> I
        end
        
        I --> J[MinimalSignal JSON<br/>615 tokens]
    end
    
    subgraph "DECODER"
        J --> K[MSP Decoder<br/>LLM]
        K --> L[Decoded Natural Language<br/>1164 tokens]
    end
    
    subgraph "EVALUATION & REFINEMENT"
        L --> M[Semantic Judge<br/>Sentence Embeddings<br/>all-MiniLM-L6-v2]
        A --> M
        M --> N[Cosine Similarity<br/>Calculation]
        N --> O{Similarity<br/>>= 80%?}
        
        O -->|No| P[Loss Analyzer<br/>LLM]
        P --> Q[Identify Specific<br/>Missing Facts]
        Q --> R[MISSING: $1.2M revenue<br/>MISSING: 4:00 AM deadline<br/>MISSING: Sarah Martinez<br/>etc.]
        R --> S[Prioritize by<br/>Section Importance]
        S --> T[Feedback Loop]
        T --> I
        
        O -->|Yes| U[Success]
    end
    
    subgraph "Agent B"
        U --> V[Final Decoded Message<br/>1164 tokens<br/>47.3% compression<br/>91.5% similarity]
    end
    
    style A fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style V fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    style F fill:#fff4e1,stroke:#f57c00,stroke-width:2px
    style J fill:#f0f0f0,stroke:#424242,stroke-width:2px
    style L fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style O fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style U fill:#4caf50,stroke:#1b5e20,stroke-width:3px
    style H1 fill:#ffcdd2,stroke:#c62828,stroke-width:2px
    style H2 fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    style H3 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style H4 fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    style R fill:#ffccbc,stroke:#d84315,stroke-width:2px
```

## Key Components:

**PHASE 1 - Structure Analysis:**
- Analyzes message structure before compression
- Identifies logical sections
- Rates importance (critical/high/medium/low)
- Extracts key concepts per section

**PHASE 2 - Organic Compression:**
- Applies differential compression based on importance
- No hard compression targets
- Natural compression emerges from redundancy removal
- Preserves critical facts while summarizing less important content

**PHASE 3 - Iterative Refinement:**
- Decodes signal and evaluates semantic similarity
- If < 80% similarity, identifies SPECIFIC missing facts
- Re-encodes with explicit instructions to add missing information
- Repeats until target similarity achieved (typically 2-3 iterations)

**Novel Contributions:**
1. Importance-weighted compression (not all content treated equally)
2. Organic compression without forced ratios
3. Specific loss analysis with precise feedback
4. Three-phase separation of concerns (analyze → compress → refine)

**Results:**
- Large messages: 47% compression, 91% similarity
- Preserves all critical information (numbers, names, dates)
- Converges in 2-3 iterations
