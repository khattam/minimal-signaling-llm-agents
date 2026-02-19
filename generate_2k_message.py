"""Generate a 2k token message for testing."""
from minimal_signaling.tokenization import TiktokenTokenizer

# Generate a realistic 2k token message
message_2k = """Subject: Q4 2025 Product Roadmap Review and Strategic Planning Session - Comprehensive Analysis and Recommendations

Dear Executive Leadership Team and Product Stakeholders,

I am writing to provide a comprehensive overview of our Q4 2025 product roadmap review, strategic planning outcomes, and critical recommendations that require immediate executive decision-making. This analysis synthesizes feedback from 47 customer interviews, 12 competitive analysis reports, internal team assessments across 8 departments, and market research data covering 23 industry segments.

EXECUTIVE SUMMARY AND KEY FINDINGS:

After conducting an extensive 6-week strategic review process involving cross-functional teams from Product, Engineering, Sales, Marketing, Customer Success, and Finance, we have identified significant opportunities and challenges that will shape our product strategy for the next 18 months. Our analysis reveals that while our current product portfolio has achieved 127% year-over-year growth in active users (reaching 2.3 million monthly active users), we are facing increasing competitive pressure from 5 major competitors who have collectively raised $890 million in funding over the past 12 months.

The most critical finding is that our product-market fit score has declined from 8.7/10 to 7.2/10 over the past quarter, primarily due to feature gaps in enterprise collaboration tools, mobile experience deficiencies, and integration limitations with popular third-party platforms. Customer churn has increased from 3.2% to 4.8% monthly, representing approximately $2.4 million in annual recurring revenue at risk.

DETAILED MARKET ANALYSIS AND COMPETITIVE LANDSCAPE:

Our competitive analysis reveals several concerning trends. Competitor A has launched an AI-powered workflow automation feature that has gained significant traction, with 67% of their enterprise customers adopting it within the first 3 months. This feature directly addresses a pain point that 43% of our customers have explicitly requested in feedback surveys. Competitor B has secured partnerships with 8 major enterprise software vendors, creating a comprehensive ecosystem that makes their platform significantly more attractive for large organizations with complex integration requirements.

Market research indicates that the total addressable market for our product category is expected to grow from $12.4 billion to $28.7 billion by 2028, representing a compound annual growth rate of 32%. However, our current market share of 4.2% is projected to decline to 2.8% if we do not address the identified gaps in our product offering. The fastest-growing segment is mid-market companies (100-1000 employees), which currently represents only 18% of our customer base but accounts for 41% of industry growth.

Customer feedback analysis from 847 support tickets, 234 feature requests, and 156 cancellation interviews reveals consistent themes. The top 5 requested features are: (1) Advanced analytics and reporting dashboards with customizable metrics (mentioned by 67% of enterprise customers), (2) Mobile app feature parity with desktop version (mentioned by 54% of all users), (3) Real-time collaboration tools including video conferencing and screen sharing (mentioned by 48% of team accounts), (4) API rate limit increases and webhook reliability improvements (mentioned by 39% of developer users), and (5) Single sign-on (SSO) support for additional identity providers beyond the current 3 we support (mentioned by 71% of enterprise prospects).

TECHNICAL INFRASTRUCTURE AND SCALABILITY ASSESSMENT:

Our engineering team has conducted a comprehensive technical debt assessment and infrastructure scalability review. Current findings indicate that our monolithic architecture is becoming a significant bottleneck for feature development velocity. The average time to deploy a new feature has increased from 2.3 weeks to 4.7 weeks over the past year, primarily due to tight coupling between system components and inadequate test coverage (currently at 62% compared to industry standard of 85%).

Database performance analysis reveals that our primary PostgreSQL cluster is operating at 78% capacity during peak hours, with query response times degrading by 340% when concurrent user count exceeds 45,000. Our current infrastructure can support approximately 3.5 million monthly active users before requiring significant architectural changes, giving us only 6-8 months of runway at current growth rates.

The technical team estimates that migrating to a microservices architecture would require 18-24 months and approximately $3.2 million in engineering resources, but would reduce feature deployment time by 65% and improve system reliability from current 99.2% uptime to target 99.9% uptime. This migration would also enable independent scaling of different system components, reducing infrastructure costs by an estimated 35% at scale.

PROPOSED PRODUCT ROADMAP FOR NEXT 18 MONTHS:

Based on our analysis, I am proposing a three-phase product roadmap that addresses the most critical gaps while positioning us for sustainable long-term growth.

PHASE 1 (Months 1-6): Foundation and Quick Wins
- Launch mobile app v2.0 with feature parity to desktop (estimated 4 months, 3 mobile engineers)
- Implement advanced analytics dashboard with 15 pre-built report templates (estimated 3 months, 2 backend engineers, 1 data analyst)
- Add SSO support for 5 additional identity providers including Okta, Azure AD, and Google Workspace (estimated 2 months, 2 backend engineers)
- Improve API rate limits from 1,000 to 10,000 requests per hour for enterprise plans (estimated 1 month, 1 backend engineer)
- Launch customer success program with dedicated account managers for accounts over $50k ARR (estimated 2 months, hire 3 CSMs)

Expected outcomes: Reduce churn from 4.8% to 3.5%, increase enterprise conversion rate from 12% to 18%, improve product-market fit score from 7.2 to 8.0.

PHASE 2 (Months 7-12): Competitive Differentiation
- Develop AI-powered workflow automation engine with 25 pre-built automation templates (estimated 6 months, 4 ML engineers, 2 product designers)
- Build real-time collaboration suite including video conferencing, screen sharing, and co-editing (estimated 5 months, 5 frontend engineers, 3 backend engineers)
- Launch marketplace for third-party integrations with initial 15 partner integrations (estimated 4 months, 2 platform engineers, 1 partnerships manager)
- Implement advanced permission management and audit logging for enterprise compliance (estimated 3 months, 2 backend engineers)
- Expand international presence with localization for 8 additional languages (estimated 4 months, 2 engineers, 3 translators)

Expected outcomes: Increase market share from 4.2% to 5.8%, grow enterprise segment from 23% to 35% of revenue, achieve 99.5% uptime SLA.

PHASE 3 (Months 13-18): Platform and Ecosystem
- Begin microservices migration starting with authentication and billing services (estimated 6 months, 6 backend engineers, 1 DevOps engineer)
- Launch developer platform with comprehensive API documentation, SDKs for 5 programming languages, and sandbox environment (estimated 5 months, 3 platform engineers, 1 technical writer)
- Implement predictive analytics using machine learning to provide proactive insights (estimated 4 months, 3 ML engineers)
- Build white-label solution for enterprise customers (estimated 5 months, 4 engineers, 1 product designer)
- Establish strategic partnerships with 3 major enterprise software vendors (estimated ongoing, 1 partnerships director)

Expected outcomes: Achieve 99.9% uptime, reduce infrastructure costs by 25%, grow developer ecosystem to 500+ third-party integrations, increase enterprise deal size by 45%.

RESOURCE REQUIREMENTS AND BUDGET ALLOCATION:

To execute this roadmap, we require significant investment in engineering talent, infrastructure, and go-to-market resources. The detailed breakdown is as follows:

Engineering: Hire 23 additional engineers across specializations (8 backend, 6 frontend, 4 ML/AI, 3 mobile, 2 platform) at an estimated cost of $4.8 million annually including salaries, benefits, and recruiting fees.

Infrastructure: Increase cloud infrastructure budget from current $180k/month to $320k/month to support growth and new features, totaling $1.68 million additional annual spend.

Product and Design: Hire 3 product managers and 2 product designers at estimated cost of $850k annually.

Customer Success: Build customer success team of 8 CSMs and 1 director at estimated cost of $1.2 million annually.

Marketing and Sales: Increase marketing budget by $2.4 million annually to support product launches and expand sales team by 5 enterprise account executives at cost of $1.1 million annually.

Total estimated investment: $12.03 million over 18 months.

RISK ASSESSMENT AND MITIGATION STRATEGIES:

Several significant risks could impact successful execution of this roadmap:

1. Talent Acquisition Risk: Current competitive market for engineering talent, particularly ML/AI specialists. Mitigation: Partner with 3 technical recruiting firms, offer competitive compensation packages including equity, implement employee referral program with $10k bonuses.

2. Technical Execution Risk: Complexity of microservices migration while maintaining feature velocity. Mitigation: Adopt incremental migration approach, maintain dedicated team for legacy system support, implement comprehensive testing and monitoring.

3. Market Timing Risk: Competitors may launch similar features before us. Mitigation: Prioritize features with highest customer impact, consider strategic acquisitions to accelerate capability development, maintain close customer relationships for early feedback.

4. Resource Constraints: Budget limitations may force prioritization trade-offs. Mitigation: Implement phased approach with clear success metrics, secure additional funding if needed, focus on revenue-generating features first.

5. Customer Adoption Risk: New features may not achieve expected adoption rates. Mitigation: Conduct extensive beta testing with 50+ customers per feature, implement comprehensive onboarding and training programs, gather continuous feedback and iterate rapidly.

FINANCIAL PROJECTIONS AND ROI ANALYSIS:

Based on our analysis and proposed roadmap, we project the following financial outcomes over the next 18 months:

Revenue: Increase from current $32 million ARR to $58 million ARR (81% growth), driven by reduced churn, increased enterprise conversion, and expansion revenue from existing customers.

Customer Metrics: Grow from 2.3 million to 4.2 million monthly active users, reduce churn from 4.8% to 2.9%, increase net revenue retention from 105% to 128%.

Market Position: Increase market share from 4.2% to 6.5%, establish leadership position in mid-market segment, achieve top 3 ranking in industry analyst reports.

Return on Investment: The $12.03 million investment is projected to generate $26 million in incremental ARR, representing a 2.16x return within 18 months and 4.8x return within 36 months.

RECOMMENDATIONS AND NEXT STEPS:

I strongly recommend that we proceed with the proposed three-phase roadmap with the following immediate actions:

1. Secure executive approval and budget allocation of $12.03 million for 18-month execution period by end of Q4 2025.

2. Begin immediate hiring process for critical engineering roles, with target of 15 hires completed within first 3 months.

3. Establish cross-functional roadmap execution team with dedicated program manager and weekly executive steering committee meetings.

4. Launch Phase 1 initiatives in January 2026 with clear success metrics and monthly progress reviews.

5. Conduct quarterly strategic reviews to assess market conditions, competitive landscape, and adjust roadmap priorities as needed.

The window of opportunity to address our competitive gaps and capture market share is narrowing. Our competitors are moving aggressively, and customer expectations are evolving rapidly. Delaying these investments will result in continued market share erosion, increased customer churn, and diminished ability to compete for enterprise deals.

I request an executive decision-making session within the next 2 weeks to review this proposal in detail, address any concerns, and secure commitment to move forward. I am available to provide additional analysis, financial modeling, or competitive intelligence as needed to support this decision.

This represents the most comprehensive and critical strategic initiative for our product organization, and successful execution will determine our competitive position and growth trajectory for the next 3-5 years.

Thank you for your consideration and leadership.

Best regards,
Chief Product Officer"""

tokenizer = TiktokenTokenizer()
token_count = tokenizer.count_tokens(message_2k)

print(f"Generated message:")
print(f"  Tokens: {token_count}")
print(f"  Characters: {len(message_2k)}")
print(f"\nMessage preview (first 500 chars):")
print(message_2k[:500])
print("\n...")

# Save to file
with open("message_2k.txt", "w") as f:
    f.write(message_2k)
print(f"\nSaved to message_2k.txt")
print(f"Target was 2000 tokens, got {token_count} tokens")
