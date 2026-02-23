# Architecture Diagrams for Hierarchical Adaptive Encoding

## 1. Overall System Architecture

```mermaid
graph TB
    A[Agent A<br/>Natural Language Message] --> B[Hierarchical Adaptive Encoder]
    B --> C[MinimalSignal<br/>JSON Protocol]
    C --> D[Decoder]
    D --> E[Agent B<br/>Natural Language Message]
    
    D --> F[Semantic Judge<br/>Sentence Embeddings]
    A --> F
    F --> G{Similarity >= 80%?}
    
    G -->|Yes| H[Success<br/>Communication Complete]
    G -->|No| I[Loss Analyzer<br/>Identify Missing Facts]
    I --> J[Feedback Loop]
    J --> B
    
    style A fill:#e1f5ff
    style E fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#f0f0f0
    style D fill:#fff4e1
    style F fill:#e8f5e9
    style G fill:#fff3e0
    style H fill:#c8e6c9
    style I fill:#ffccbc
    style J fill:#ffccbc
```

## 2. Three-Phase Hierarchical Encoding Process

```mermaid
graph TB
    subgraph "PHASE 1: Structure Analysis"
        A[Input Message<br/>2209 tokens] --> B[LLM Analysis]
        B --> C[Section Identification]
        C --> D[Importance Rating<br/>critical/high/medium/low]
        D --> E[Key Concepts Extraction]
        E --> F[Importance Map<br/>JSON Output]
    end
    
    subgraph "PHASE 2: Organic Compression"
        F --> G[Importance-Weighted<br/>Compression Rules]
        G --> H1[Critical Sections<br/>Preserve ALL facts]
        G --> H2[High Sections<br/>Keep main points + details]
        G --> H3[Medium Sections<br/>Summarize with key facts]
        G --> H4[Low Sections<br/>Brief summary]
        H1 --> I[MinimalSignal JSON<br/>615 tokens]
        H2 --> I
        H3 --> I
        H4 --> I
    end
    
    subgraph "PHASE 3: Iterative Refinement"
        I --> J[Decode to NL<br/>1164 tokens]
        J --> K[Semantic Judge<br/>Calculate Similarity]
        K --> L{Similarity >= 80%?}
        L -->|No| M[Loss Analyzer<br/>Identify Specific Missing Facts]
        M --> N[Re-encode with<br/>Missing Facts]
        N --> J
        L -->|Yes| O[Final Output<br/>91.5% similarity]
    end
    
    style A fill:#e1f5ff
    style F fill:#fff4e1
    style I fill:#f0f0f0
    style O fill:#c8e6c9
    style M fill:#ffccbc
```

## 3. Comparison: Baseline vs Graph-Based vs Hierarchical Adaptive

```mermaid
graph TB
    subgraph "Baseline Approach"
        B1[Message] --> B2[Single-Pass Encoder<br/>'Compress by 30%']
        B2 --> B3[MinimalSignal]
        B3 --> B4[Decoder]
        B4 --> B5[Semantic Judge]
        B5 --> B6{Good?}
        B6 -->|No| B7[Vague Feedback<br/>'Some details missing']
        B7 --> B2
        B6 -->|Yes| B8[Done]
    end
    
    subgraph "Graph-Based Approach (Failed)"
        G1[Message] --> G2[Entity Extraction<br/>spaCy NLP]
        G2 --> G3[Build Knowledge Graph<br/>Nodes + Edges]
        G3 --> G4[Graph Compression<br/>PageRank Pruning]
        G4 --> G5[Graph Reconstruction<br/>Template Generation]
        G5 --> G6[Unnatural Text<br/>65% similarity ❌]
    end
    
    subgraph "Hierarchical Adaptive (Current)"
        H1[Message] --> H2[Structure Analysis<br/>Importance Rating]
        H2 --> H3[Organic Compression<br/>Importance-Weighted]
        H3 --> H4[MinimalSignal]
        H4 --> H5[Decoder]
        H5 --> H6[Semantic Judge]
        H6 --> H7{Good?}
        H7 -->|No| H8[Specific Feedback<br/>'MISSING: $1.2M revenue']
        H8 --> H3
        H7 -->|Yes| H9[Done<br/>91.5% similarity ✅]
    end
    
    style B8 fill:#fff3e0
    style G6 fill:#ffcdd2
    style H9 fill:#c8e6c9
```

## 4. Detailed Phase 1: Structure Analysis

```mermaid
flowchart LR
    A[Input Message] --> B[LLM with<br/>Structure Analysis Prompt]
    
    B --> C1[Section 1:<br/>Executive Summary]
    B --> C2[Section 2:<br/>Timeline]
    B --> C3[Section 3:<br/>Root Cause]
    B --> C4[Section 4:<br/>Impact]
    B --> C5[Section 5:<br/>Actions]
    B --> C6[Section 6:<br/>Lessons]
    B --> C7[Section 7:<br/>Recommendations]
    
    C1 --> D1[CRITICAL<br/>revenue, outage, cause]
    C2 --> D2[HIGH<br/>timestamps, actions]
    C3 --> D3[CRITICAL<br/>bug, testing, monitoring]
    C4 --> D4[HIGH<br/>users, revenue, tickets]
    C5 --> D5[HIGH<br/>fixes, deadlines, owners]
    C6 --> D6[MEDIUM<br/>best practices]
    C7 --> D7[MEDIUM<br/>future improvements]
    
    D1 --> E[Importance Map<br/>JSON Output]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    D6 --> E
    D7 --> E
    
    style A fill:#e1f5ff
    style E fill:#fff4e1
    style D1 fill:#ffcdd2
    style D3 fill:#ffcdd2
    style D2 fill:#ffe0b2
    style D4 fill:#ffe0b2
    style D5 fill:#ffe0b2
    style D6 fill:#fff9c4
    style D7 fill:#fff9c4
```

## 5. Detailed Phase 2: Importance-Weighted Compression

```mermaid
graph TB
    A[Importance Map] --> B{Section Importance}
    
    B -->|CRITICAL| C1[Compression Rule:<br/>Preserve ALL facts]
    B -->|HIGH| C2[Compression Rule:<br/>Keep main points + details]
    B -->|MEDIUM| C3[Compression Rule:<br/>Summarize with key facts]
    B -->|LOW| C4[Compression Rule:<br/>Brief summary]
    
    C1 --> D1[Example:<br/>'$1.2M revenue loss,<br/>2.3M users affected,<br/>4h 37m outage']
    C2 --> D2[Example:<br/>'14:23 UTC - alerts triggered,<br/>response time 120ms → 3400ms']
    C3 --> D3[Example:<br/>'Lessons: monitoring critical,<br/>testing must reflect production']
    C4 --> D4[Example:<br/>'Recommendations include<br/>hiring SRE team']
    
    D1 --> E[MinimalSignal JSON]
    D2 --> E
    D3 --> E
    D4 --> E
    
    E --> F[Compression Ratio:<br/>Original: 2209 tokens<br/>Signal: 615 tokens<br/>Decoded: 1164 tokens<br/>47.3% compression]
    
    style C1 fill:#ffcdd2
    style C2 fill:#ffe0b2
    style C3 fill:#fff9c4
    style C4 fill:#f1f8e9
    style F fill:#c8e6c9
```

## 6. Detailed Phase 3: Iterative Refinement Loop

```mermaid
flowchart TD
    A[MinimalSignal<br/>Iteration 1] --> B[Decoder]
    B --> C[Decoded Text<br/>1164 tokens]
    C --> D[Semantic Judge<br/>Sentence Embeddings]
    E[Original Text<br/>2209 tokens] --> D
    
    D --> F[Cosine Similarity<br/>79.5%]
    F --> G{>= 80%?}
    
    G -->|No| H[Loss Analyzer]
    H --> I[Compare Original vs Decoded]
    I --> J[Identify Missing Facts:<br/>• MISSING: $340K SLA credits<br/>• MISSING: 490 of 500 connections<br/>• MISSING: 23 job instances<br/>• MISSING: 4:00 AM rollback deadline<br/>• MISSING: Monday 9:00 AM post-mortem]
    
    J --> K[Prioritize by Importance:<br/>Critical section losses first]
    K --> L[Re-encode with Feedback]
    L --> M[MinimalSignal<br/>Iteration 2]
    
    M --> N[Decoder]
    N --> O[Decoded Text<br/>1164 tokens]
    O --> P[Semantic Judge]
    E --> P
    P --> Q[Cosine Similarity<br/>91.5%]
    Q --> R{>= 80%?}
    R -->|Yes| S[Success!<br/>Final Output]
    
    style A fill:#f0f0f0
    style F fill:#ffccbc
    style J fill:#ffccbc
    style M fill:#f0f0f0
    style Q fill:#c8e6c9
    style S fill:#4caf50
```

## 7. Message Length Impact on Compression Strategy

```mermaid
graph TB
    A[Input Message] --> B{Message Length?}
    
    B -->|< 800 tokens| C[Short Message]
    B -->|800-1500 tokens| D[Medium Message]
    B -->|> 1500 tokens| E[Long Message]
    
    C --> C1[Characteristics:<br/>• Minimal redundancy<br/>• Dense content<br/>• JSON overhead dominates]
    C1 --> C2[Baseline Approach:<br/>✅ 85% similarity<br/>✅ 30% compression]
    C1 --> C3[Hierarchical Approach:<br/>❌ Expansion<br/>✅ 80% similarity]
    
    D --> D1[Characteristics:<br/>• Some redundancy<br/>• Mixed content density<br/>• Balanced overhead]
    D1 --> D2[Baseline Approach:<br/>⚠️ 75% similarity<br/>✅ 30% compression]
    D1 --> D3[Hierarchical Approach:<br/>✅ 85% similarity<br/>✅ 40% compression]
    
    E --> E1[Characteristics:<br/>• Significant redundancy<br/>• Structured content<br/>• Importance varies]
    E1 --> E2[Baseline Approach:<br/>❌ 69% similarity<br/>✅ 30% compression]
    E1 --> E3[Hierarchical Approach:<br/>✅ 91% similarity<br/>✅ 47% compression]
    
    style C2 fill:#c8e6c9
    style C3 fill:#ffcdd2
    style D2 fill:#fff9c4
    style D3 fill:#c8e6c9
    style E2 fill:#ffcdd2
    style E3 fill:#4caf50
```

## 8. Semantic Similarity vs Compression Trade-off

```mermaid
graph LR
    subgraph "Baseline Approach"
        B1[High Compression<br/>40%] -.->|Linear trade-off| B2[Low Similarity<br/>65%]
        B3[Medium Compression<br/>30%] -.-> B4[Medium Similarity<br/>75%]
        B5[Low Compression<br/>20%] -.-> B6[High Similarity<br/>85%]
    end
    
    subgraph "Hierarchical Adaptive"
        H1[High Compression<br/>47%] -.->|Non-linear trade-off| H2[High Similarity<br/>91%]
        H3[Medium Compression<br/>40%] -.-> H4[High Similarity<br/>87%]
        H5[Low Compression<br/>30%] -.-> H6[High Similarity<br/>92%]
    end
    
    style H1 fill:#4caf50
    style H2 fill:#4caf50
    style B1 fill:#ffcdd2
    style B2 fill:#ffcdd2
```

## 9. Data Flow: Original Message to Final Output

```mermaid
sequenceDiagram
    participant A as Agent A
    participant E as Encoder
    participant S as Structure Analyzer
    participant C as Compressor
    participant D as Decoder
    participant J as Judge
    participant L as Loss Analyzer
    participant B as Agent B
    
    A->>E: Natural Language Message<br/>(2209 tokens)
    E->>S: Analyze Structure
    S->>S: Identify Sections<br/>Rate Importance<br/>Extract Key Concepts
    S->>E: Importance Map
    
    E->>C: Compress with Importance Rules
    C->>C: Apply Differential Compression<br/>CRITICAL: Keep all facts<br/>HIGH: Keep main points<br/>MEDIUM: Summarize<br/>LOW: Brief summary
    C->>E: MinimalSignal JSON (615 tokens)
    
    E->>D: Decode Signal
    D->>J: Decoded Text (1164 tokens)
    J->>J: Calculate Similarity<br/>Original vs Decoded
    J->>E: 79.5% similarity ❌
    
    E->>L: Analyze Loss
    L->>L: Identify Missing Facts<br/>MISSING: $340K SLA credits<br/>MISSING: 4:00 AM deadline<br/>etc.
    L->>E: Specific Feedback
    
    E->>C: Re-compress with Missing Facts
    C->>E: Updated MinimalSignal (641 tokens)
    E->>D: Decode Signal
    D->>J: Decoded Text (1164 tokens)
    J->>J: Calculate Similarity
    J->>E: 91.5% similarity ✅
    
    E->>B: Final Decoded Message<br/>(1164 tokens, 47.3% compression)
```

## 10. Component Architecture

```mermaid
classDiagram
    class HierarchicalAdaptiveEncoder {
        +GroqClient groq_client
        +SemanticJudge judge
        +MSPDecoder decoder
        +TiktokenTokenizer tokenizer
        +int max_iterations
        +float target_similarity
        +encode_with_refinement(text) HierarchicalEncodingResult
        -_analyze_structure(text) str
        -_encode_hierarchical(text, importance) MinimalSignal
        -_encode_with_feedback(text, importance, feedback) MinimalSignal
        -_analyze_loss(original, decoded) str
        -_extract_missing_concepts(feedback, importances) List
    }
    
    class SemanticJudge {
        +SentenceTransformer model
        +float threshold
        +evaluate(original, decoded) JudgeResult
        -_calculate_similarity(text1, text2) float
    }
    
    class MSPDecoder {
        +GroqClient groq_client
        +decode(signal, style) str
        -_format_signal(signal) str
    }
    
    class MinimalSignal {
        +str version
        +str intent
        +str target
        +str priority
        +dict summary
        +List~ContentSection~ sections
        +List constraints
        +dict state
        +str encoding_strategy
    }
    
    class ContentSection {
        +str title
        +str content
        +str importance
    }
    
    class HierarchicalEncodingResult {
        +str original_text
        +int original_tokens
        +MinimalSignal final_signal
        +str final_decoded
        +float final_similarity
        +int iterations
        +List~HierarchicalRefinementStep~ refinement_history
        +bool converged
    }
    
    HierarchicalAdaptiveEncoder --> SemanticJudge
    HierarchicalAdaptiveEncoder --> MSPDecoder
    HierarchicalAdaptiveEncoder --> MinimalSignal
    HierarchicalAdaptiveEncoder --> HierarchicalEncodingResult
    MinimalSignal --> ContentSection
```

## 11. Prompt Engineering Flow

```mermaid
graph TB
    subgraph "Structure Analysis Prompt"
        A1[System Prompt:<br/>Analyze message structure] --> A2[User Input:<br/>Full message text]
        A2 --> A3[LLM Output:<br/>JSON with sections,<br/>importance ratings,<br/>key concepts]
    end
    
    subgraph "Hierarchical Encoding Prompt"
        B1[System Prompt:<br/>Compress with importance rules<br/>+ Importance Map] --> B2[User Input:<br/>Full message text]
        B2 --> B3[LLM Output:<br/>MinimalSignal JSON<br/>with compressed sections]
    end
    
    subgraph "Refinement Prompt"
        C1[System Prompt:<br/>Re-encode with feedback<br/>+ Importance Map<br/>+ Missing Concepts] --> C2[User Input:<br/>Full message text]
        C2 --> C3[LLM Output:<br/>Updated MinimalSignal JSON<br/>with missing facts added]
    end
    
    subgraph "Loss Analysis Prompt"
        D1[System Prompt:<br/>Identify missing information] --> D2[User Input:<br/>Original text<br/>+ Decoded text]
        D2 --> D3[LLM Output:<br/>List of specific<br/>missing facts]
    end
    
    A3 --> B1
    B3 --> D1
    D3 --> C1
    C3 --> E{Similarity<br/>>= 80%?}
    E -->|No| D1
    E -->|Yes| F[Done]
    
    style A3 fill:#e1f5ff
    style B3 fill:#fff4e1
    style C3 fill:#fff4e1
    style D3 fill:#ffccbc
    style F fill:#c8e6c9
```

## 12. Performance Comparison Matrix

```mermaid
graph TB
    subgraph "Baseline Approach"
        B[Performance Matrix]
        B --> B1[Small Messages: ✅✅✅]
        B --> B2[Medium Messages: ⚠️⚠️]
        B --> B3[Large Messages: ❌❌]
        B --> B4[Fact Preservation: ⚠️]
        B --> B5[Natural Language: ✅✅]
        B --> B6[Convergence Speed: ⚠️]
    end
    
    subgraph "Graph-Based Approach"
        G[Performance Matrix]
        G --> G1[Small Messages: ❌]
        G --> G2[Medium Messages: ❌]
        G --> G3[Large Messages: ❌]
        G --> G4[Fact Preservation: ⚠️]
        G --> G5[Natural Language: ❌❌]
        G --> G6[Convergence Speed: N/A]
    end
    
    subgraph "Hierarchical Adaptive"
        H[Performance Matrix]
        H --> H1[Small Messages: ❌❌]
        H --> H2[Medium Messages: ✅✅]
        H --> H3[Large Messages: ✅✅✅]
        H --> H4[Fact Preservation: ✅✅✅]
        H --> H5[Natural Language: ✅✅✅]
        H --> H6[Convergence Speed: ✅✅]
    end
    
    style B1 fill:#c8e6c9
    style B2 fill:#fff9c4
    style B3 fill:#ffcdd2
    style G1 fill:#ffcdd2
    style G2 fill:#ffcdd2
    style G3 fill:#ffcdd2
    style G5 fill:#ffcdd2
    style H1 fill:#ffcdd2
    style H2 fill:#c8e6c9
    style H3 fill:#4caf50
    style H4 fill:#4caf50
    style H5 fill:#4caf50
    style H6 fill:#c8e6c9
```

---

## How to Use These Diagrams

1. **For Thesis:** Copy the mermaid code blocks into your thesis document. Most markdown processors and thesis templates support mermaid diagrams.

2. **For Presentations:** Use online tools like:
   - https://mermaid.live/ (render and export as PNG/SVG)
   - https://mermaid.ink/ (generate image URLs)

3. **For Documentation:** These diagrams are already in markdown format and will render in GitHub, GitLab, and most documentation platforms.

4. **Key Diagrams for Your Prof:**
   - Diagram 2: Three-Phase Process (shows the core innovation)
   - Diagram 3: Comparison (shows why your approach is better)
   - Diagram 7: Message Length Impact (explains the trade-offs)
   - Diagram 9: Data Flow (shows the complete system)
