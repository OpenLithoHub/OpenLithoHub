# Commercial Viability Assessment: Hosted Co-Design Platform

## Executive Summary

This document evaluates whether OpenLithoHub should pursue a commercial
hosted co-design layer on top of the open-source core. It is structured as
a demand-validation probe: customer profiles, value propositions, pricing
models, and a go/no-go framework.

## Target Customer Profiles

### Profile A: Fabless Semiconductor Startup

- **Size**: 10-50 engineers
- **Pain point**: No in-house OPC/ILT team, cannot justify a Synopsys
  CATS or Cadence Litho Analyzer license ($500K+/year)
- **Need**: Push-button mask optimization with reasonable quality
- **Budget**: $5K-20K/month for tools
- **Decision maker**: CAD/EDA team lead or VP Engineering

### Profile B: University Research Lab

- **Size**: 3-10 researchers
- **Pain point**: Need to benchmark novel ILT algorithms against
  baselines but lack lithography simulator infrastructure
- **Need**: Standardized benchmarking environment with leaderboard
- **Budget**: $500-2K/month (grant-funded)
- **Decision maker**: PI or postdoc leading the project

### Profile C: Established IDM / Foundry R&D

- **Size**: 100+ engineers
- **Pain point**: Internal tooling exists but is slow to iterate;
  need rapid prototyping for new process node exploration
- **Need**: Co-design sandbox that integrates with internal flows via API
- **Budget**: $20K-100K/month for targeted tools
- **Decision maker**: DFM group manager or lithography R&D director

### Profile D: Multiphysics Design Consultant

- **Size**: 1-5 engineers
- **Pain point**: Client projects span lithography + CFD + thermal;
  no single tool covers all
- **Need**: Co-design platform linking DiffCFD + OpenLithoHub + DiffNano
- **Budget**: $1K-5K/month
- **Decision maker**: Principal engineer / company owner

## Value Proposition: Hosted vs. Self-Hosted

| Dimension | Self-Hosted (OSS) | Hosted (Commercial) |
|-----------|-------------------|---------------------|
| Setup time | Hours-days (GPU drivers, deps) | Minutes (web UI / API key) |
| GPU hardware | Customer must provision | Included |
| Model selection | Manual config | Auto-tuned per process node |
| Multi-GPU scaling | Manual multiproc setup | Automatic load balancing |
| Monitoring | Roll your own | Dashboard with alerts |
| Support | Community only | SLA-backed, priority queue |
| Data privacy | Full control (on-prem) | Cloud-hosted (NDA available) |
| Cost | Hardware + engineer time | Subscription fee |

### Key Differentiator

The hosted layer must justify its cost against the self-hosted option.
The primary value is **time-to-first-result**: an engineer at a fabless
startup should get a optimized mask within 15 minutes of signing up,
versus 2-3 days of setting up self-hosted infrastructure.

## Pricing Model Options

### Option 1: Per-Optimization-Run

- $0.50-5.00 per optimization run (based on problem size)
- Good for: sporadic users, research labs
- Risk: unpredictable revenue; heavy users churn to self-hosted
- Unit economics: GPU time costs ~$1-3/hr on cloud; typical run = 1-10 min

### Option 2: Monthly Subscription (Tiered)

| Tier | Price | Included | Target |
|------|-------|----------|--------|
| Free | $0 | 10 runs/month, community support | Researchers, evaluation |
| Pro | $500/month | 500 runs/month, email support | Small teams |
| Team | $3,000/month | 5,000 runs/month, API access, priority support | Mid-size companies |
| Enterprise | Custom | Unlimited, on-prem option, SLA | IDMs, foundries |

- Good for: predictable revenue, budget planning
- Risk: underuse (customers pay but don't use) or overuse

### Option 3: GPU-Time Metered

- $2-4/hour of GPU compute consumed
- Good for: heavy users who want cost proportional to usage
- Risk: bill shock; hard to budget

### Recommendation

Start with **Option 2 (subscription)** because:
1. Predictable revenue enables capacity planning
2. Free tier drives adoption and benchmarking data
3. Enterprise tier captures high-value customers
4. Simpler billing than metered GPU time

## Customer Interview Questions

### For Fabless Startups (Profile A)

1. What is your current OPC/ILT workflow? (manual, commercial tool, none?)
2. How many mask optimizations do you run per month?
3. What is your tolerance for optimization quality vs. cost? (90% of
   commercial quality at 10% of cost?)
4. Would you trust a cloud-hosted service with your design data under NDA?
5. What is your target process node? (28nm, 14nm, 7nm, 5nm?)
6. How long does your current mask optimization take per run?

### For University Labs (Profile B)

1. What baselines do you currently compare against?
2. Do you need GPU access, or do you have your own?
3. Is reproducibility across papers important to you?
4. Would a public leaderboard drive adoption of your research?
5. What is your grant cycle for tool budget?

### For IDMs (Profile C)

1. Where does your internal tooling fall short?
2. What would a co-design sandbox need to integrate with? (GDS/OASIS in/out?
   KLayout bridge? Calibre DRC hooks?)
3. Is a hosted API acceptable, or do you require on-prem deployment?
4. What is your evaluation process and timeline for new tools?
5. What existing commercial tools would this displace or complement?

### For Consultants (Profile D)

1. What domains do your projects span? (litho + CFD? thermal + optics?)
2. Do clients require specific output formats? (GDS, OASIS, CIF?)
3. How do you currently handle multiphysics optimization?
4. Would API-based access fit your workflow?

## Decision Criteria

### Go Signals (Invest in Commercial Layer)

- [ ] 5+ potential customers interviewed, 3+ express willingness to pay
- [ ] At least one customer commits to a pilot ($1K+ MRR)
- [ ] Self-hosted setup takes >4 hours for a new user (it currently does)
- [ ] Competitor pricing leaves a gap we can fill at >70% margin
- [ ] OpenLithoHub benchmarking data has commercial value (leaderboard, process node coverage)

### No-Go Signals (Stay Open-Source Only)

- [ ] All interviewed customers prefer self-hosted for data privacy
- [ ] No one willing to pay >$500/month
- [ ] Internal tooling at target customers already covers our use case
- [ ] Engineering effort to build hosted layer > 6 months of team time
- [ ] GPU cloud costs make unit economics negative at viable price points

### Key Metrics to Track

| Metric | Target (6 months) | Measurement |
|--------|-------------------|-------------|
| Interview count | 20+ potential customers | CRM log |
| Conversion rate (trial -> paid) | >10% | Billing system |
| Monthly active users (free tier) | 500+ | Analytics |
| Paying customers | 10+ | Billing system |
| MRR | $10K+ | Finance |
| Time-to-first-result | <15 min | Product analytics |
| NPS | >40 | Survey |

## Go/No-Go Decision Framework

```
                    Interview Results
                          |
              +-----------+-----------+
              |                       |
         Positive signal         Negative signal
         (3+ willing to pay)    (none willing to pay)
              |                       |
         Build MVP                Stay OSS-only
         (3 months)               (no commercial layer)
              |
         Pilot with 3 customers
         (3 months)
              |
         +----+----+
         |         |
     Pilot       Pilot
     succeeds    fails
     (>3 paying) (<3 paying)
         |         |
     Full launch  Sunset pilot
     (hire sales) (refocus on OSS)
```

### Phase Gate Schedule

1. **Month 1-2**: Customer interviews (20+), pricing research
2. **Month 3**: Go/No-Go decision point
3. **Month 3-5**: MVP build (if Go) -- web UI, API, billing, GPU autoscaling
4. **Month 6-8**: Pilot with 3-5 customers
5. **Month 9**: Scale decision -- continue, pivot, or sunset

### Minimum Viable Product Scope

If we proceed, the MVP should include:

- Web UI for mask upload + optimization
- REST API for programmatic access
- 3 model options (Neural-ILT, GAN-OPC, Surrogate-ILT)
- Basic billing (Stripe integration)
- Results dashboard with quality metrics
- GDS/OASIS download

### Estimated Investment

| Item | Cost (3 months) |
|------|----------------|
| 2 engineers (full-stack + ML infra) | $60K |
| GPU cloud (development + pilot) | $5K |
| Stripe/billing setup | $2K |
| Domain, SSL, monitoring | $1K |
| **Total** | **$68K** |

Break-even at $10K MRR = ~7 months after launch (assuming steady 70% gross margin).
