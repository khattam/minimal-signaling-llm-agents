# Implementation Plan

- [x] 1. Set up core data models and interfaces




  - [ ] 1.1 Create data models module with SemanticKey, CompressionResult, ExtractionResult, JudgeResult dataclasses
    - Define KeyType enum with INSTRUCTION, STATE, GOAL, CONTEXT, CONSTRAINT


    - Implement Pydantic models for validation


    - _Requirements: 3.3, 4.1, 4.2_


  - [x] 1.2 Write property test for semantic key serialization round-trip






    - **Property 8: Semantic key serialization round-trip**


    - **Validates: Requirements 4.3**






  - [ ] 1.3 Create abstract base classes for Compressor, SemanticKeyExtractor, Judge, Tokenizer
    - Define method signatures matching design document


    - _Requirements: 2.1_







  - [ ] 1.4 Write property test for schema validation
    - **Property 9: Schema validation rejects invalid input**

    - **Validates: Requirements 4.4**

- [x] 2. Implement configuration system


  - [x] 2.1 Create MediatorConfig Pydantic model with validation

    - Support token_budget, recursion_limit, stage toggles
    - Add dashboard config section

    - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.6_
  - [x] 2.2 Implement YAML config loader with error handling




    - Load from file path, validate against schema
    - _Requirements: 6.1, 6.7_


  - [x] 2.3 Write property test for configuration validation


    - **Property 12: Configuration validation**







    - **Validates: Requirements 6.7**



- [x] 3. Implement tokenization layer


  - [x] 3.1 Create TiktokenTokenizer implementation




    - Use tiktoken library for accurate token counting
    - Support configurable encoding


    - _Requirements: 1.1_

  - [ ] 3.2 Write property test for token count consistency
    - **Property 1: Token count consistency**




    - **Validates: Requirements 1.1**



- [ ] 4. Checkpoint - Ensure all tests pass







  - Ensure all tests pass, ask the user if questions arise.



- [ ] 5. Implement Stage 1: Compression Engine
  - [x] 5.1 Create DistilBARTCompressor implementation




    - Load model from HuggingFace transformers
    - Implement compress method with summarization

    - _Requirements: 2.2_
  - [ ] 5.2 Create CompressionEngine with recursive compression logic
    - Implement compress_to_budget with recursion tracking





    - Handle termination conditions (budget met, limit reached, no improvement)
    - _Requirements: 1.2, 1.3, 1.4_
  - [x] 5.3 Write property test for compression reduces tokens


    - **Property 2: Compression reduces or maintains token count**
    - **Validates: Requirements 1.2**
  - [ ] 5.4 Write property test for recursive compression termination
    - **Property 3: Recursive compression termination**
    - **Validates: Requirements 1.3, 1.4**
  - [ ] 5.5 Write property test for compression result completeness
    - **Property 4: Compression result completeness**
    - **Validates: Requirements 1.5**

- [ ] 6. Implement Stage 2: Semantic Key Extraction
  - [ ] 6.1 Create PlaceholderExtractor for initial development
    - Parse text for key patterns deterministically
    - Return valid ExtractionResult
    - _Requirements: 3.1, 3.2_
  - [ ] 6.2 Write property test for extraction result schema conformance
    - **Property 7: Extraction result schema conformance**
    - **Validates: Requirements 3.2, 3.4**
  - [ ] 6.3 Write property test for extraction follows compression
    - **Property 6: Extraction follows compression**
    - **Validates: Requirements 3.1**

- [ ] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement Judge verification layer
  - [ ] 8.1 Create PlaceholderJudge implementation
    - Return configurable pass/fail with confidence
    - Support issue detection stub
    - _Requirements: 5.2, 5.3_
  - [ ] 8.2 Write property test for judge result completeness
    - **Property 11: Judge result completeness**
    - **Validates: Requirements 5.2, 5.3**
  - [ ] 8.3 Write property test for judge invocation follows configuration
    - **Property 10: Judge invocation follows configuration**
    - **Validates: Requirements 5.1, 5.4**

- [x] 9. Implement Mediator orchestration



  - [ ] 9.1 Create Mediator class with pipeline orchestration
    - Wire together compression, extraction, judge

    - Handle stage enable/disable based on config
    - _Requirements: 8.1, 8.2, 8.3_
  - [ ] 9.2 Write property test for compressor interface substitutability
    - **Property 5: Compressor interface substitutability**

    - **Validates: Requirements 2.3**
  - [x] 9.3 Write property test for end-to-end pipeline integrity



    - **Property 14: End-to-end pipeline integrity**
    - **Validates: Requirements 8.1-8.4**

- [x] 10. Implement trace logging


  - [ ] 10.1 Create TraceLogger with JSONL output
    - Record all pipeline stages
    - Include timestamps, token counts, compression ratios



    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [ ] 10.2 Write property test for trace record completeness
    - **Property 13: Trace record completeness**
    - **Validates: Requirements 7.1-7.5**

- [ ] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement event system for real-time updates
  - [ ] 12.1 Create EventEmitter and PipelineEvent types
    - Define event payloads for each pipeline stage
    - Support async event emission
    - _Requirements: 10.2, 11.3_
  - [ ] 12.2 Integrate event emission into Mediator pipeline
    - Emit events at each stage transition
    - Include relevant data in payloads
    - _Requirements: 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [ ] 13. Implement WebSocket server
  - [ ] 13.1 Create WebSocketServer with broadcast capability
    - Handle client connections
    - Broadcast events to all connected clients
    - _Requirements: 13.2_
  - [ ] 13.2 Integrate WebSocket server with EventEmitter
    - Subscribe to pipeline events
    - Forward events to WebSocket clients
    - _Requirements: 13.2_

- [ ] 14. Implement Dashboard backend
  - [ ] 14.1 Create FastAPI-based dashboard server
    - Serve static files for frontend
    - Handle WebSocket upgrade
    - Expose API for sending test messages
    - _Requirements: 13.1, 13.5_
  - [ ] 14.2 Create dashboard launch command
    - Single command to start mediator + dashboard
    - _Requirements: 13.5_

- [ ] 15. Implement Dashboard frontend
  - [ ] 15.1 Set up React + TypeScript frontend with Vite
    - Configure build tooling
    - Set up WebSocket client
    - _Requirements: 13.3_
  - [ ] 15.2 Create pipeline architecture visualization component
    - Render flow diagram: Agent A → Stage 1 → Stage 2 → Judge → Agent B
    - Show enabled/disabled states
    - Animate active stage
    - _Requirements: 10.1, 11.1, 11.2, 11.3_
  - [ ] 15.3 Create message display component
    - Show current message text at each stage
    - Display token counts
    - _Requirements: 10.3_
  - [ ] 15.4 Create compression progress component
    - Show real-time compression ratio
    - Display timeline of compression passes
    - _Requirements: 10.4, 10.6_
  - [ ] 15.5 Create semantic keys display component
    - Render keys in structured format
    - Color-code by key type
    - _Requirements: 10.5_
  - [ ] 15.6 Create metrics panel component
    - Show token budget and usage
    - Display compression ratio bar
    - Show pass count vs limit
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  - [ ] 15.7 Write property test for metrics calculation accuracy
    - **Property 15: Metrics calculation accuracy**
    - **Validates: Requirements 12.1-12.3**
  - [ ] 15.8 Implement dark mode support
    - Toggle between light and dark themes
    - _Requirements: 13.4_

- [ ] 16. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Create minimal demo
  - [ ] 17.1 Create demo script with sample long message
    - Message that exceeds default token budget
    - _Requirements: 9.1_
  - [ ] 17.2 Implement demo runner with console output
    - Display compression progress
    - Show final semantic keys
    - Print compression statistics
    - _Requirements: 9.2, 9.3, 9.4, 9.5_
  - [ ] 17.3 Integrate trace logging into demo
    - Write trace file on completion
    - _Requirements: 9.6_

- [ ] 18. Final integration and polish
  - [ ] 18.1 Create CLI entry point with commands
    - `demo` - run minimal demo
    - `serve` - start dashboard server
    - `process` - process a single message
    - _Requirements: 9.1, 13.5_
  - [ ] 18.2 Update README with usage instructions
    - Quick start guide
    - Configuration documentation
    - Dashboard screenshots
    - _Requirements: 9.1_

- [ ] 19. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
