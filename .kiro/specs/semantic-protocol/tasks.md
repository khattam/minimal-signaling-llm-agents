# Implementation Plan

- [x] 1. Define MSP Schema and Data Models

  - [x] 1.1 Create `protocol.py` with MinimalSignal Pydantic model


    - Define all fields: intent, target, params, constraints, state, priority, trace_id, timestamp
    - Add field validators and JSON serialization
    - _Requirements: 1.1, 1.2, 1.4, 1.5_
  - [ ]* 1.2 Write property test for MSP schema validity
    - **Property 1: MSP Schema Validity**
    - **Validates: Requirements 1.1, 1.2, 1.4, 1.5**
  - [ ]* 1.3 Write property test for MSP JSON round-trip
    - **Property 8: MSP JSON Round-Trip**


    - **Validates: Requirements 1.2**
  - [x] 1.4 Create result models (PipelineResult, PipelineMetrics, JudgeResult)


    - _Requirements: 5.1, 5.2_


- [ ] 2. Implement Groq Client with Rate Limiting
  - [ ] 2.1 Create `groq_client.py` with async Groq wrapper
    - Implement rate limiter for 30 req/min free tier
    - Add JSON mode support for structured outputs
    - _Requirements: 2.2, 8.1, 8.5_
  - [x] 2.2 Add configuration loading from environment variables

    - Support GROQ_API_KEY env var
    - Model selection via config
    - _Requirements: 8.4, 8.6_
  - [ ]* 2.3 Write unit tests for rate limiter
    - _Requirements: 8.5_

- [x] 3. Implement MSP Encoder

  - [x] 3.1 Create `encoder.py` with MSPEncoder class


    - Define system prompt for structured extraction
    - Implement encode() method with JSON mode
    - _Requirements: 2.1, 2.3_
  - [ ]* 3.2 Write property test for encoder produces valid MSP
    - **Property 2: Encoder Produces Valid MSP**
    - **Validates: Requirements 2.1**
  - [x] 3.3 Add error handling for empty input and API failures


    - _Requirements: 2.4_

- [x] 4. Implement MSP Decoder

  - [x] 4.1 Create `decoder.py` with MSPDecoder class


    - Define system prompt for natural language generation
    - Implement decode() method with style parameter
    - _Requirements: 3.1, 3.3_
  - [ ]* 4.2 Write property test for decoder produces non-empty output
    - **Property 3: Decoder Produces Non-Empty Output**
    - **Validates: Requirements 3.1**
  - [ ]* 4.3 Write property test for decoder preserves MSP content
    - **Property 4: Decoder Preserves MSP Content**
    - **Validates: Requirements 3.5**
  - [x] 4.4 Add error handling for invalid MSP and API failures


    - _Requirements: 3.4_

- [ ] 5. Checkpoint - Ensure encoder/decoder tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Semantic Judge

  - [x] 6.1 Create `semantic_judge.py` with SemanticJudge class


    - Load sentence-transformers model (all-MiniLM-L6-v2)
    - Implement evaluate() with cosine similarity
    - _Requirements: 4.1, 4.2_
  - [ ]* 6.2 Write property test for judge score bounds
    - **Property 5: Judge Score Bounds**
    - **Validates: Requirements 4.1, 4.3, 4.4**
  - [x] 6.3 Add threshold-based pass/fail logic


    - _Requirements: 4.4_
  - [x] 6.4 Add detailed metrics output


    - _Requirements: 4.5_

- [x] 7. Implement Pipeline Orchestrator


  - [x] 7.1 Create `msp_pipeline.py` with MSPPipeline class


    - Wire together encoder, decoder, judge
    - Implement process() method
    - _Requirements: 2.1, 3.1, 4.1_
  - [ ]* 7.2 Write property test for metrics computation
    - **Property 6: Metrics Computation Correctness**
    - **Validates: Requirements 5.1, 5.2, 5.4**
  - [ ]* 7.3 Write property test for round-trip semantic preservation
    - **Property 7: Round-Trip Semantic Preservation**
    - **Validates: Requirements 4.1, 3.5**
  - [x] 7.4 Add event emission for real-time updates


    - _Requirements: 6.2, 6.3_

- [ ] 8. Checkpoint - Ensure pipeline tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Update Dashboard for MSP Visualization
  - [x] 9.1 Update backend server to use MSPPipeline
    - Replace old compression pipeline with MSP pipeline
    - Add new API endpoints for MSP processing
    - _Requirements: 6.1_
  - [x] 9.2 Update frontend to display MSP JSON
    - Show original NL, MSP JSON (formatted), decoded NL side-by-side
    - _Requirements: 6.1, 6.2_
  - [x] 9.3 Add semantic fidelity score display
    - Show similarity score with visual indicator
    - _Requirements: 6.3_
  - [x] 9.4 Add token count and compression metrics display
    - _Requirements: 6.4_

- [ ] 10. Implement Evaluation Harness
  - [ ] 10.1 Create `evaluation.py` with batch processing
    - Support JSON test datasets
    - Compute aggregate metrics
    - _Requirements: 7.1, 7.2_
  - [ ] 10.2 Add comparison report generation
    - MSP vs raw NL baseline
    - _Requirements: 7.3_
  - [ ] 10.3 Create sample test dataset
    - Diverse agent message types
    - _Requirements: 7.5_

- [ ] 11. Final Checkpoint - Full system validation
  - Ensure all tests pass, ask the user if questions arise.
