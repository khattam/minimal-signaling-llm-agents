# Requirements Document

## Introduction

This document specifies the requirements for the Mediated Minimal-Signaling Architecture, a research prototype that enforces a communication bottleneck between LLM agents. The system transforms natural language messages through a two-stage pipeline (compression + semantic key extraction) with optional verification, enabling study of minimal information exchange required for agent coordination.

The architecture addresses three problems with unrestricted natural language communication:
1. Verbosity (expensive, context window consumption)
2. Ambiguity (cross-model interpretation drift)
3. Instability (shared-belief drift over multiple turns)

## Glossary

- **Mediator**: The central component that processes messages between Agent A and Agent B through the compression and semantic key extraction pipeline
- **Stage 1 (Compression)**: The first transformation stage that reduces message token count using learned summarization
- **Stage 2 (Semantic Key Extraction)**: The second transformation stage that converts compressed text into structured symbolic keys
- **Semantic Key**: A small symbolic unit following a schema (e.g., `INSTRUCTION: update`, `STATE: confidence=high`)
- **Token Budget**: The maximum number of tokens allowed for a compressed message
- **Recursion Limit**: The maximum number of compression passes allowed
- **Judge**: An optional verification layer that evaluates semantic key fidelity
- **Trace**: A structured log artifact recording all pipeline stages for analysis

## Requirements

### Requirement 1: Message Compression (Stage 1)

**User Story:** As a researcher, I want the mediator to compress verbose agent messages to fit within a token budget, so that I can study coordination under bandwidth constraints.

#### Acceptance Criteria

1. WHEN the Mediator receives a message from Agent A THEN the Mediator SHALL measure the message token count using a configurable tokenizer
2. WHEN the message token count exceeds the configured token budget THEN the Mediator SHALL invoke the compression model to reduce the message length
3. WHILE the compressed message exceeds the token budget AND the recursion count is below the recursion limit THEN the Mediator SHALL re-compress the message recursively
4. WHEN the recursion limit is reached OR compression yields no meaningful reduction THEN the Mediator SHALL terminate compression and proceed with the current result
5. WHEN compression completes THEN the Mediator SHALL output the final compressed text, original token count, final token count, number of compression passes, and a step-by-step compression log

### Requirement 2: Compression Model Abstraction

**User Story:** As a researcher, I want to swap different compression models without changing the pipeline, so that I can experiment with various summarization approaches.

#### Acceptance Criteria

1. THE Mediator SHALL define a Compressor interface with a compress method accepting text input and returning compressed text
2. THE Mediator SHALL provide a DistilBART-based implementation of the Compressor interface as the default
3. WHEN a different Compressor implementation is configured THEN the Mediator SHALL use that implementation without code changes to the pipeline

### Requirement 3: Semantic Key Extraction (Stage 2)

**User Story:** As a researcher, I want compressed messages converted to structured semantic keys, so that I can study stable symbolic representations across different LLMs.

#### Acceptance Criteria

1. WHEN Stage 1 compression completes THEN the Mediator SHALL pass the compressed text to the Semantic Key Extractor
2. THE Semantic Key Extractor SHALL produce a list of semantic keys conforming to a defined schema
3. THE semantic key schema SHALL support key types including INSTRUCTION, STATE, GOAL, CONTEXT, and CONSTRAINT
4. WHEN extraction completes THEN the Mediator SHALL output the semantic keys, schema version, and raw extractor output for debugging

### Requirement 4: Semantic Key Schema

**User Story:** As a researcher, I want semantic keys to follow a consistent schema, so that I can reliably parse and analyze agent communication patterns.

#### Acceptance Criteria

1. THE Semantic Key schema SHALL define each key as having a type field and a value field
2. THE schema SHALL include a version identifier for tracking schema evolution
3. WHEN serializing semantic keys THEN the Mediator SHALL produce valid JSON conforming to the schema
4. WHEN deserializing semantic keys THEN the Mediator SHALL validate input against the schema and report validation errors

### Requirement 5: Judge Verification Layer

**User Story:** As a researcher, I want optional verification that semantic keys faithfully represent the original message, so that I can detect information loss or hallucination.

#### Acceptance Criteria

1. WHEN the Judge is enabled in configuration THEN the Mediator SHALL invoke the Judge after semantic key extraction
2. WHEN the Judge evaluates semantic keys THEN the Judge SHALL return a pass/fail status and a confidence score
3. WHEN the Judge detects issues THEN the Judge SHALL return a list of detected problems including missing critical information, hallucinated keys, or contradictions
4. WHEN the Judge is disabled in configuration THEN the Mediator SHALL skip verification and proceed directly to output

### Requirement 6: Configuration System

**User Story:** As a researcher, I want to configure pipeline parameters via a config file, so that I can run experiments with different settings without code changes.

#### Acceptance Criteria

1. THE Mediator SHALL load configuration from a YAML file at startup
2. THE configuration SHALL support setting the token budget as a positive integer
3. THE configuration SHALL support setting the recursion limit as a positive integer
4. THE configuration SHALL support enabling or disabling Stage 1 compression
5. THE configuration SHALL support enabling or disabling Stage 2 semantic key extraction
6. THE configuration SHALL support enabling or disabling the Judge verification layer
7. WHEN configuration values are invalid THEN the Mediator SHALL report clear validation errors at startup

### Requirement 7: Trace Logging

**User Story:** As a researcher, I want detailed trace logs of each pipeline run, so that I can analyze compression behavior and debug issues.

#### Acceptance Criteria

1. WHEN a message passes through the Mediator THEN the Mediator SHALL create a trace record
2. THE trace record SHALL include the original message metadata including timestamp and token count
3. THE trace record SHALL include each compression step with input tokens, output tokens, and compression ratio
4. THE trace record SHALL include the extracted semantic keys
5. WHEN the Judge is enabled THEN the trace record SHALL include the Judge result
6. WHEN the pipeline completes THEN the Mediator SHALL write the trace record to a JSONL file in the configured trace directory

### Requirement 8: End-to-End Message Flow

**User Story:** As a researcher, I want a complete message flow from Agent A to Agent B through the mediator, so that I can demonstrate the full architecture.

#### Acceptance Criteria

1. WHEN Agent A sends a message THEN the Mediator SHALL receive and process the message through the configured pipeline stages
2. WHEN all pipeline stages complete THEN the Mediator SHALL deliver semantic keys and minimal metadata to Agent B
3. THE metadata delivered to Agent B SHALL include the schema version and compression statistics
4. WHEN any pipeline stage fails THEN the Mediator SHALL log the error and report failure to the caller

### Requirement 9: Minimal Demo Scenario

**User Story:** As a researcher, I want a runnable demo that proves the architecture works, so that I can validate the implementation before running experiments.

#### Acceptance Criteria

1. THE demo SHALL accept a sample message that exceeds the configured token budget
2. WHEN the demo runs THEN the Mediator SHALL compress the message through one or more passes until it fits the budget
3. WHEN compression completes THEN the Mediator SHALL extract semantic keys from the compressed message
4. WHEN the Judge is enabled THEN the demo SHALL include Judge evaluation in the output
5. WHEN the demo completes THEN the demo SHALL display the semantic keys and compression statistics
6. WHEN the demo completes THEN the demo SHALL write a trace log file that can be inspected

### Requirement 10: Real-Time Visualization Dashboard

**User Story:** As a researcher, I want a visual dashboard showing the pipeline architecture and message progression in real-time, so that I can understand and debug the system behavior intuitively.

#### Acceptance Criteria

1. THE Visualization Dashboard SHALL display an interactive architecture diagram showing Agent A, Mediator stages, and Agent B
2. WHEN a message enters the pipeline THEN the Dashboard SHALL highlight the active stage with visual feedback
3. THE Dashboard SHALL display the current message text at each stage with token count indicators
4. WHEN compression occurs THEN the Dashboard SHALL show a real-time progress indicator with compression ratio
5. WHEN semantic keys are extracted THEN the Dashboard SHALL display the keys in a structured, readable format
6. THE Dashboard SHALL show a timeline view of compression passes with before/after token counts
7. WHEN the Judge evaluates keys THEN the Dashboard SHALL display the pass/fail status and any detected issues

### Requirement 11: Dashboard Architecture View

**User Story:** As a researcher, I want to see the full pipeline architecture visually, so that I can understand the data flow and system structure at a glance.

#### Acceptance Criteria

1. THE Dashboard SHALL render a flow diagram showing the complete pipeline: Agent A → Stage 1 → Stage 2 → Judge → Agent B
2. THE flow diagram SHALL use visual indicators to show enabled versus disabled stages based on configuration
3. WHEN a stage is processing THEN the Dashboard SHALL animate the connection between stages to show data flow
4. THE Dashboard SHALL display configuration parameters for each stage in a collapsible panel

### Requirement 12: Dashboard Metrics Panel

**User Story:** As a researcher, I want to see compression metrics and statistics in real-time, so that I can monitor system performance during experiments.

#### Acceptance Criteria

1. THE Dashboard SHALL display a metrics panel showing current token budget and usage
2. THE metrics panel SHALL show compression ratio as both a percentage and a visual bar
3. THE metrics panel SHALL display the number of compression passes used versus the limit
4. WHEN multiple messages are processed THEN the Dashboard SHALL show aggregate statistics including average compression ratio and total messages processed
5. THE Dashboard SHALL update metrics in real-time as messages flow through the pipeline

### Requirement 13: Dashboard Technology Stack

**User Story:** As a researcher, I want the dashboard built with modern web technologies, so that it is maintainable and provides a polished user experience.

#### Acceptance Criteria

1. THE Dashboard SHALL be implemented as a web application accessible via browser
2. THE Dashboard backend SHALL communicate with the Mediator via WebSocket for real-time updates
3. THE Dashboard SHALL use a modern frontend framework for responsive, interactive visualization
4. THE Dashboard SHALL support dark mode for comfortable extended use
5. THE Dashboard SHALL be launchable with a single command from the project root
