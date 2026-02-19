"""Test hierarchical adaptive encoding on long message."""
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from minimal_signaling.tokenization import TiktokenTokenizer
from minimal_signaling.groq_client import GroqClient
from minimal_signaling.semantic_judge import SemanticJudge
from minimal_signaling.msp_decoder import MSPDecoder
from minimal_signaling.encoding.hierarchical_adaptive_encoder import HierarchicalAdaptiveEncoder

# Same long message from test_long_message.py
LONG_MESSAGE = """Hi team, I wanted to provide a comprehensive update on the Q4 2024 product roadmap 
and address several critical issues that have emerged over the past two weeks. This is going to be 
a detailed update, so please read through carefully as there are action items for multiple teams.

First, regarding the authentication service migration that we started on November 1st. The migration 
from Auth0 to our in-house OAuth2 implementation is currently 65% complete. We've successfully migrated 
12,000 enterprise users and 450,000 consumer accounts. However, we're experiencing some concerning issues 
with the session management layer. Specifically, users are reporting that they're being logged out 
unexpectedly after 15-20 minutes of activity, even though our session timeout is configured for 8 hours. 
The engineering team has traced this to a Redis cluster configuration issue where the TTL values aren't 
being properly propagated across all 6 nodes in the cluster.

Sarah from DevOps is working on this with AWS support, but we need to make a decision by Friday about 
whether to roll back the migration for the remaining 35% of users or push forward with a hotfix. The 
rollback would cost us approximately 3 weeks of development time and $75K in AWS migration costs that 
we've already incurred. The hotfix approach is riskier but could be deployed by Monday if we dedicate 
the entire backend team to it this weekend.

Second, the mobile app performance issues that were escalated last week. Our analytics show that the 
iOS app is crashing for 8.3% of users on devices running iOS 17.2 or later. This is a critical issue 
because iOS 17.2+ represents 67% of our active mobile user base. The crash logs indicate a memory leak 
in the image caching layer, specifically in how we're handling WebP format images. The mobile team has 
identified the root cause - we're using an outdated version of the SDWebImage library (version 5.12) 
that has a known memory management bug with WebP images larger than 2MB.

The fix is straightforward - upgrade to SDWebImage 5.18 - but this requires updating our entire image 
pipeline and re-testing all image-related features. QA estimates this will take 4-5 days of testing. 
We need to decide if we want to fast-track this through our normal release process or do an emergency 
patch release. An emergency release would get the fix to users by Wednesday but bypasses our standard 
beta testing phase with our 5,000 beta users.

Third, the data warehouse migration to Snowflake. We're currently running dual systems - our legacy 
PostgreSQL data warehouse and the new Snowflake instance - which is costing us $12,000 per month in 
redundant infrastructure. The original plan was to complete the migration by October 31st, but we're 
now looking at a December 15th completion date. The delay is primarily due to the complexity of 
migrating our custom ETL pipelines. We have 47 different data pipelines, and only 23 have been 
successfully migrated and validated so far.

The data engineering team is requesting 2 additional contractors for 6 weeks to accelerate the migration. 
The cost would be approximately $45,000, but it would save us roughly $36,000 in redundant infrastructure 
costs over the next 3 months. Additionally, the longer we run dual systems, the higher the risk of data 
inconsistencies between the two warehouses, which could impact our business intelligence dashboards and 
executive reporting.

Fourth, regarding the customer support ticket backlog. We're currently sitting at 2,847 open tickets, 
which is 340% higher than our target of 700 tickets. The average response time has increased from 4 hours 
to 18 hours, and customer satisfaction scores have dropped from 4.2/5.0 to 3.1/5.0 over the past month. 
This is directly impacting our renewal rates - we've seen a 12% increase in churn among customers who 
had support tickets open for more than 48 hours.

The support team is requesting approval to hire 5 additional support engineers immediately. At $65K 
annual salary each, this would be $325K in additional headcount costs. However, the alternative is 
potentially losing high-value customers. Our analysis shows that each 1% increase in churn costs us 
approximately $2.3M in annual recurring revenue.

Fifth, the security audit findings from our SOC 2 Type II audit. The auditors identified 8 medium-severity 
findings and 2 high-severity findings that need to be remediated before we can receive our certification. 
The high-severity findings are: (1) insufficient access logging for production database access, and 
(2) lack of automated security scanning in our CI/CD pipeline. We need SOC 2 certification to close 
several enterprise deals worth a combined $4.5M in ARR.

The security team has a remediation plan that would take 6 weeks to implement fully. However, the auditors 
have indicated that if we can address the 2 high-severity findings within 2 weeks, they would be willing 
to issue a conditional certification that would allow us to proceed with the enterprise deals while we 
work on the medium-severity findings.

Finally, budget planning for Q1 2025. Finance is requesting that all department heads submit their Q1 
budget proposals by November 20th. Please include detailed justifications for any headcount increases, 
new tools or services, and capital expenditures over $10,000. The executive team will review all proposals 
during the week of November 27th, and final budgets will be approved by December 6th.

Additionally, I want to address the infrastructure costs that have been escalating. Our AWS bill for 
October was $287,000, which is 23% higher than our budgeted amount of $233,000. The primary drivers are: 
increased EC2 usage in our production environment (up 31% month-over-month), higher data transfer costs 
due to our expanded CDN usage (up $18K), and the dual data warehouse setup I mentioned earlier. We need 
to implement better cost monitoring and optimization strategies. I'm asking all engineering teams to 
review their resource usage and identify opportunities for optimization.

On the product side, we're seeing strong traction with the new analytics dashboard that launched in 
September. Usage has grown to 12,400 daily active users, which is 156% of our initial target. Customer 
feedback has been overwhelmingly positive (4.7/5.0 average rating), and we're seeing increased engagement 
across the board. However, this success is putting strain on our backend infrastructure, which ties back 
to the AWS cost increases I mentioned. We need to scale our infrastructure to support this growth while 
keeping costs under control.

The marketing team has also requested budget for a major campaign in Q1 targeting enterprise customers 
in the healthcare and financial services verticals. The proposed budget is $450,000, which would be our 
largest marketing investment to date. Early projections suggest this could generate 200-250 qualified 
leads and potentially $8-12M in pipeline. However, we need to ensure our sales team has the capacity 
to handle this influx of leads, which may require hiring 2-3 additional sales engineers.

Action items:
1. Engineering leads: Provide recommendation on auth migration (rollback vs hotfix) by Thursday EOD
2. Mobile team: Submit iOS crash fix timeline and release strategy by Wednesday 10 AM
3. Data engineering: Provide detailed cost-benefit analysis for contractor request by Friday
4. Support team: Present hiring plan and interim solutions for ticket backlog by Tuesday
5. Security team: Prioritize high-severity findings and provide 2-week remediation plan by Monday
6. All department heads: Submit Q1 2025 budget proposals by November 20th
7. Engineering teams: Conduct AWS cost optimization review and submit findings by November 25th
8. Sales leadership: Assess capacity for Q1 marketing campaign and provide hiring recommendations

Please let me know if you have any questions or concerns about any of these items. We'll have a 
leadership sync on Friday at 2 PM to discuss the most critical decisions. I know this is a lot to 
digest, but these are all important initiatives that will set us up for success in 2025.

Thanks,
Alex"""


async def main():
    tokenizer = TiktokenTokenizer()
    token_count = tokenizer.count_tokens(LONG_MESSAGE)
    
    print(f"Message token count: {token_count} tokens")
    print(f"Target: 80% semantic similarity with 30-50% compression")
    
    # Initialize components
    groq = GroqClient()
    judge = SemanticJudge(threshold=0.80)
    decoder = MSPDecoder(groq)
    
    # Create hierarchical encoder
    encoder = HierarchicalAdaptiveEncoder(
        groq_client=groq,
        judge=judge,
        decoder=decoder,
        max_iterations=5,
        target_similarity=0.80
    )
    
    # Run encoding
    result = await encoder.encode_with_refinement(LONG_MESSAGE)
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"✅ Success: {result.converged}")
    print(f"🔄 Iterations: {result.iterations}")
    print(f"📊 Similarity: {result.final_similarity:.1%} (target: 80%)")
    print(f"🗜️  Compression: {result.signal_tokens/result.original_tokens:.1%}")
    print(f"📝 Tokens: {result.original_tokens} → {result.signal_tokens}")
    
    print(f"\n📈 Iteration Progress:")
    for step in result.refinement_history:
        print(f"  Iter {step.iteration}: "
              f"similarity={step.similarity_score:.1%}, "
              f"tokens={step.signal_tokens} ({step.signal_tokens/result.original_tokens:.1%})")
    
    # Show section breakdown
    print(f"\n📑 Final Signal Structure:")
    print(f"  Intent: {result.final_signal.intent}")
    print(f"  Target: {result.final_signal.target}")
    print(f"  Sections: {len(result.final_signal.sections)}")
    for i, sec in enumerate(result.final_signal.sections, 1):
        sec_tokens = tokenizer.count_tokens(sec.content)
        print(f"    {i}. {sec.title} ({sec.importance}): {sec_tokens} tokens")
    
    # Show decoded output sample
    print(f"\n📄 Decoded Output (first 500 chars):")
    print(f"{result.final_decoded[:500]}...")
    
    # Save results
    import json
    output = {
        "success": result.converged,
        "iterations": result.iterations,
        "original_tokens": result.original_tokens,
        "final_tokens": result.signal_tokens,
        "compression_ratio": result.signal_tokens / result.original_tokens,
        "final_similarity": result.final_similarity,
        "target_similarity": 0.80,
        "sections": len(result.final_signal.sections),
        "iteration_history": [
            {
                "iteration": step.iteration,
                "similarity": step.similarity_score,
                "tokens": step.signal_tokens,
                "compression": step.signal_tokens / result.original_tokens
            }
            for step in result.refinement_history
        ]
    }
    
    with open("hierarchical_test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to hierarchical_test_results.json")


if __name__ == "__main__":
    asyncio.run(main())
