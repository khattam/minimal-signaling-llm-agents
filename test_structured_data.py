"""Test graph compression with structured data (not narrative emails)."""

import asyncio
from dotenv import load_dotenv
from src.minimal_signaling.groq_client import GroqClient
from src.minimal_signaling.encoding.graph_based.iterative_graph_pipeline import IterativeGraphPipeline
from src.minimal_signaling.tokenization import TiktokenTokenizer

load_dotenv()

# Structured data - API monitoring report (perfect for graph compression)
STRUCTURED_MESSAGE = """
System Health Report - Production Environment
Generated: 2024-11-15 14:30:00 UTC
Report ID: SHR-2024-1115-001

API Endpoint Performance Metrics:
- /api/v1/users: 1,247,893 requests, 234ms avg latency, 99.97% success rate, 0.03% error rate (mostly 429 rate limits)
- /api/v1/orders: 892,441 requests, 456ms avg latency, 99.89% success rate, 0.11% error rate (timeout issues)
- /api/v1/products: 2,103,567 requests, 123ms avg latency, 99.99% success rate, 0.01% error rate
- /api/v1/payments: 445,223 requests, 678ms avg latency, 99.95% success rate, 0.05% error rate (Stripe API delays)
- /api/v1/analytics: 334,112 requests, 1,234ms avg latency, 98.76% success rate, 1.24% error rate (database query timeouts)

Database Performance:
- Primary PostgreSQL: 89% CPU utilization, 12.3GB memory usage, 2,341 active connections, 45ms avg query time
- Read Replica 1: 67% CPU utilization, 8.7GB memory usage, 1,892 active connections, 38ms avg query time
- Read Replica 2: 71% CPU utilization, 9.1GB memory usage, 2,014 active connections, 41ms avg query time
- Redis Cache: 34% memory utilization (17GB/50GB), 99.8% hit rate, 234,567 keys, 12ms avg latency

Infrastructure Metrics:
- EC2 Instances: 47 running (32 web servers, 12 worker nodes, 3 database servers), $12,340/day cost
- Load Balancer: 3.2M requests/hour, 234GB data transfer, 99.99% uptime
- S3 Storage: 2.3TB used, 45M objects, $890/month cost
- CloudFront CDN: 12.4TB data transfer, 234M requests, 98.7% cache hit rate, $2,340/month cost

Error Analysis:
- 500 Internal Server Errors: 1,234 occurrences, primarily in /api/v1/analytics endpoint, caused by database connection pool exhaustion
- 429 Rate Limit Errors: 3,456 occurrences, affecting /api/v1/users endpoint, triggered by automated scrapers from IP range 203.45.67.0/24
- 503 Service Unavailable: 567 occurrences, during deployment window 13:00-13:15 UTC, affected all endpoints
- 504 Gateway Timeout: 890 occurrences, in /api/v1/orders endpoint, caused by slow third-party shipping API integration

Security Events:
- Failed Login Attempts: 23,456 attempts, 89% from known bot IPs, blocked by rate limiter
- SQL Injection Attempts: 234 attempts, all blocked by WAF, originating from 12 unique IPs
- DDoS Mitigation: 3 incidents, largest was 45Gbps attack, successfully mitigated by CloudFlare, lasted 23 minutes
- Suspicious API Key Usage: 12 API keys flagged for unusual geographic access patterns, 3 keys revoked

Resource Utilization Trends:
- CPU Usage: increased 23% week-over-week, peak at 89% during 14:00-16:00 UTC
- Memory Usage: increased 15% week-over-week, currently at 78% capacity
- Disk I/O: increased 34% week-over-week, primary bottleneck in analytics queries
- Network Bandwidth: increased 28% week-over-week, driven by increased CDN usage

Alerts Triggered:
- High CPU Alert: 12 instances, threshold 85%, max duration 45 minutes
- Database Connection Pool Alert: 8 instances, threshold 90% utilization, max duration 23 minutes  
- Disk Space Alert: 3 instances, threshold 80% full, affected log servers
- SSL Certificate Expiry Warning: 2 certificates expiring in 14 days (api-staging.example.com, cdn-backup.example.com)

Recommendations:
- Scale database connection pool from 2,500 to 3,500 connections to prevent exhaustion
- Add 2 additional read replicas to distribute analytics query load
- Implement query result caching for /api/v1/analytics endpoint to reduce database load
- Upgrade EC2 instances from t3.large to t3.xlarge for web servers experiencing high CPU
- Block IP range 203.45.67.0/24 at firewall level to prevent rate limit abuse
- Renew SSL certificates for api-staging.example.com and cdn-backup.example.com before expiry
- Investigate slow shipping API integration, consider implementing timeout and fallback mechanism
- Review and optimize top 10 slowest database queries identified in analytics endpoint
"""

async def main():
    tokenizer = TiktokenTokenizer()
    token_count = tokenizer.count_tokens(STRUCTURED_MESSAGE)
    
    print(f"="*80)
    print(f"TESTING GRAPH COMPRESSION WITH STRUCTURED DATA")
    print(f"="*80)
    print(f"\nMessage type: System Health Report (structured data)")
    print(f"Token count: {token_count} tokens")
    print(f"Goal: 30% compression (→ ~{int(token_count * 0.7)} tokens) with 80% similarity\n")
    
    client = GroqClient()
    pipeline = IterativeGraphPipeline(
        groq_client=client,
        target_similarity=0.80,
        initial_entropy_target=0.90,
        max_iterations=3
    )
    
    result = await pipeline.compress(STRUCTURED_MESSAGE)
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    for i, iter_result in enumerate(result.iterations, 1):
        print(f"Iteration {i}:")
        print(f"  Nodes: {iter_result.nodes_kept}/{iter_result.total_nodes}")
        print(f"  Tokens: {iter_result.decoded_tokens} ({iter_result.compression_ratio:.1%})")
        print(f"  Similarity: {iter_result.similarity_score:.1%}")
        print()
    
    print(f"Final: {result.final_tokens} tokens ({result.final_compression:.1%}) - Similarity: {result.final_similarity:.1%}")
    print(f"Target: ~{int(token_count * 0.7)} tokens (70%) - Similarity: 80%")
    
    if result.final_similarity >= 0.80:
        print(f"\n✅ SUCCESS! Achieved target similarity!")
    else:
        print(f"\n⚠️  Below target similarity")
    
    # Save results
    pipeline.save_results(result, "structured_data_results")
    
    print(f"\n📄 Decoded message preview:")
    print(result.final_message[:500])
    print("...")

if __name__ == "__main__":
    asyncio.run(main())
