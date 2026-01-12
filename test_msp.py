"""Quick test script for MSP pipeline."""
import os
import asyncio

# Load from .env file
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.msp_pipeline import MSPPipeline
from minimal_signaling.msp_config import MSPConfig

async def test_pipeline():
    print("=" * 60)
    print("MSP Pipeline Test")
    print("=" * 60)
    
    # Test message - more verbose to show compression benefit
    test_message = """Hello there! I hope this message finds you well. I wanted to reach out to you today because I have an important request that I believe requires your immediate attention and expertise. 

As you may already be aware, our company has been collecting quarterly sales data from all of our regional offices across the country, including the Northeast, Southeast, Midwest, Southwest, and West Coast divisions. 

What I need from you is a comprehensive analysis of this data. Specifically, I am looking for you to identify any significant trends that have emerged over the past year, as well as any anomalies or outliers that might warrant further investigation. This analysis is critically important because we will be presenting the findings at our upcoming Q4 board meeting, where senior leadership will be making key strategic decisions based on your insights.

Please note that there is a firm deadline for this work - the report absolutely must be completed and ready for review by this Friday at the latest. Given the importance of this deliverable and the tight timeline, I am marking this as a high priority task that should take precedence over other work items.

Thank you so much for your help with this matter. Please let me know if you have any questions or need any clarification on the requirements."""
    
    print(f"\n📝 Original Message ({len(test_message)} chars):")
    print(test_message)
    print()
    
    # Initialize pipeline
    config = MSPConfig.from_env()
    pipeline = MSPPipeline(config=config)
    
    print("🔄 Processing through MSP pipeline...")
    print()
    
    # Process
    result = await pipeline.process(test_message, style="professional")
    
    # Show results
    print("📦 MSP Signal (JSON):")
    print(f"  intent: {result.signal.intent}")
    print(f"  target: {result.signal.target}")
    print(f"  params: {result.signal.params}")
    print(f"  constraints: {result.signal.constraints}")
    print(f"  state: {result.signal.state}")
    print(f"  priority: {result.signal.priority}")
    print()
    
    print("📤 Decoded Output:")
    print(result.decoded_text)
    print()
    
    print("📊 Metrics:")
    print(f"  Original tokens: {result.metrics.original_tokens}")
    print(f"  Signal tokens:   {result.metrics.signal_tokens}")
    print(f"  Decoded tokens:  {result.metrics.decoded_tokens}")
    print(f"  Compression:     {result.metrics.compression_ratio:.1%}")
    print(f"  Latency:         {result.metrics.latency_ms:.0f}ms")
    print()
    
    print("⚖️ Semantic Judge:")
    print(f"  Passed:     {result.judge.passed}")
    print(f"  Similarity: {result.judge.similarity_score:.1%}")
    print(f"  Issues:     {result.judge.issues}")
    print()
    
    print("=" * 60)
    print("✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
