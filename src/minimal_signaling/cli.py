"""CLI entry point for the minimal-signaling pipeline."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Minimal Signaling Pipeline - Research prototype for LLM agent communication"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the dashboard server")
    serve_parser.add_argument(
        "--host", default="localhost", help="Host to bind to (default: localhost)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8080, help="Port to listen on (default: 8080)"
    )
    serve_parser.add_argument(
        "--config", help="Path to config file"
    )
    serve_parser.add_argument(
        "--real-compressor", action="store_true",
        help="Use real DistilBART compressor (slower but more realistic)"
    )
    
    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run a minimal demo")
    demo_parser.add_argument(
        "--config", help="Path to config file"
    )
    demo_parser.add_argument(
        "--message", help="Custom message to process"
    )
    demo_parser.add_argument(
        "--real-compressor", action="store_true",
        help="Use real DistilBART compressor"
    )
    
    # process command
    process_parser = subparsers.add_parser("process", help="Process a single message")
    process_parser.add_argument(
        "message", help="Message to process"
    )
    process_parser.add_argument(
        "--config", help="Path to config file"
    )
    process_parser.add_argument(
        "--output", help="Output file for trace (default: stdout)"
    )
    
    args = parser.parse_args()
    
    if args.command == "serve":
        run_serve(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "process":
        run_process(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_serve(args):
    """Run the dashboard server."""
    from .server import create_dashboard_server
    
    print(f"Starting dashboard server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    server = create_dashboard_server(
        config_path=args.config,
        use_real_compressor=args.real_compressor
    )
    server.run(host=args.host, port=args.port)


def run_demo(args):
    """Run a minimal demo."""
    from .config import MediatorConfig, CompressionConfig, SemanticKeysConfig, JudgeConfig
    from .mediator import Mediator
    from .extraction import PlaceholderExtractor
    from .tokenization import TiktokenTokenizer
    from .judge import PlaceholderJudge
    from .trace import TraceLogger
    from .interfaces import Compressor
    
    print("=" * 60)
    print("Minimal Signaling Pipeline - Demo")
    print("=" * 60)
    
    # Default demo message
    default_message = """
    INSTRUCTION: Analyze the quarterly sales data and prepare a comprehensive report.
    STATE: The data has been collected from all regional offices.
    GOAL: Identify trends, anomalies, and provide actionable recommendations.
    CONTEXT: This is for the Q4 board meeting presentation.
    CONSTRAINT: The report must be completed by end of week.
    Additional notes: Focus on year-over-year comparisons and highlight any
    significant changes in customer behavior patterns. Include visualizations
    where appropriate and ensure all data is properly validated before inclusion.
    """
    
    message = args.message or default_message
    
    # Load or create config
    if args.config:
        config = MediatorConfig.from_yaml(args.config)
    else:
        config = MediatorConfig(
            compression=CompressionConfig(
                enabled=True,
                token_budget=30,
                max_recursion=5
            ),
            semantic_keys=SemanticKeysConfig(enabled=True),
            judge=JudgeConfig(enabled=True)
        )
    
    # Create components
    tokenizer = TiktokenTokenizer()
    
    if args.real_compressor:
        from .compression import DistilBARTCompressor
        compressor = DistilBARTCompressor()
        print("Using DistilBART compressor (this may take a moment to load...)")
    else:
        class MockCompressor(Compressor):
            def compress(self, text: str) -> str:
                if not text.strip():
                    return text
                words = text.split()
                return " ".join(words[: max(1, len(words) // 2)])
        compressor = MockCompressor()
        print("Using mock compressor (use --real-compressor for DistilBART)")
    
    extractor = PlaceholderExtractor()
    judge = PlaceholderJudge()
    
    # Create mediator
    mediator = Mediator(
        config=config,
        compressor=compressor,
        extractor=extractor,
        tokenizer=tokenizer,
        judge=judge
    )
    
    # Process message
    print("\n--- Input Message ---")
    print(message.strip())
    
    original_tokens = tokenizer.count_tokens(message)
    print(f"\nOriginal tokens: {original_tokens}")
    print(f"Token budget: {config.compression.token_budget}")
    
    print("\n--- Processing ---")
    result = mediator.process(message)
    
    # Display results
    print("\n--- Results ---")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_ms:.2f}ms")
    
    if result.compression:
        print(f"\nCompression:")
        print(f"  Original tokens: {result.compression.original_tokens}")
        print(f"  Final tokens: {result.compression.final_tokens}")
        print(f"  Compression ratio: {result.compression.total_ratio:.2%}")
        print(f"  Passes: {result.compression.passes}")
        print(f"\n  Compressed text:")
        print(f"  {result.compression.compressed_text[:200]}...")
    
    if result.extraction:
        print(f"\nExtracted Keys ({len(result.extraction.keys)}):")
        for key in result.extraction.keys:
            print(f"  [{key.type.value}] {key.value}")
    
    if result.judge:
        print(f"\nJudge Result:")
        print(f"  Passed: {result.judge.passed}")
        print(f"  Confidence: {result.judge.confidence:.2%}")
        if result.judge.issues:
            print(f"  Issues: {result.judge.issues}")
    
    # Save trace
    trace_logger = TraceLogger()
    trace_file = trace_logger.log_trace_from_result(
        original_text=message,
        original_tokens=original_tokens,
        result=result,
        config=config
    )
    print(f"\nTrace saved to: {trace_file}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


def run_process(args):
    """Process a single message."""
    import json
    from .config import MediatorConfig, CompressionConfig, SemanticKeysConfig, JudgeConfig
    from .mediator import Mediator
    from .extraction import PlaceholderExtractor
    from .tokenization import TiktokenTokenizer
    from .judge import PlaceholderJudge
    from .interfaces import Compressor
    
    # Load or create config
    if args.config:
        config = MediatorConfig.from_yaml(args.config)
    else:
        config = MediatorConfig(
            compression=CompressionConfig(enabled=True, token_budget=50, max_recursion=5),
            semantic_keys=SemanticKeysConfig(enabled=True),
            judge=JudgeConfig(enabled=True)
        )
    
    # Create components (use mock compressor for speed)
    tokenizer = TiktokenTokenizer()
    
    class MockCompressor(Compressor):
        def compress(self, text: str) -> str:
            if not text.strip():
                return text
            words = text.split()
            return " ".join(words[: max(1, len(words) // 2)])
    
    mediator = Mediator(
        config=config,
        compressor=MockCompressor(),
        extractor=PlaceholderExtractor(),
        tokenizer=tokenizer,
        judge=PlaceholderJudge()
    )
    
    # Process
    result = mediator.process(args.message)
    
    # Output
    output = {
        "success": result.success,
        "original_tokens": tokenizer.count_tokens(args.message),
        "final_tokens": result.compression.final_tokens if result.compression else None,
        "compression_ratio": result.compression.total_ratio if result.compression else None,
        "passes": result.compression.passes if result.compression else 0,
        "keys": [
            {"type": k.type.value, "value": k.value}
            for k in (result.extraction.keys if result.extraction else [])
        ],
        "judge_passed": result.judge.passed if result.judge else None,
        "duration_ms": result.duration_ms
    }
    
    output_str = json.dumps(output, indent=2)
    
    if args.output:
        Path(args.output).write_text(output_str)
        print(f"Output written to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
