# Decision Brief — Customer Intelligence and Anomaly Investigation

## Executive decision

**Decision owner:** Fraud operations / customer operations lead  
**Decision:** Use the platform to prioritize human review and investigate connected customer, device and IP entities. Do not treat anomaly scores as fraud probabilities or use them as automatic enforcement decisions.

## Operational question

Which customer behaviours and shared entities deserve analyst attention, and what evidence should an investigator inspect before deciding whether a case needs escalation?

## Evidence and workflow

- Customer profiles combine behavioural features with K-Means segments.
- Anomaly candidates are ranked using Isolation Forest percentile (60%) and LOF percentile (40%).
- A user-device-IP graph supports connected-entity exploration, component analysis, betweenness and community inspection.
- The published investigation graph is bounded to up to 100,000 edges; customer purchase timelines retain up to 50 recent transactions per customer.
- The historical class label is excluded from customer feature generation, segmentation, behavioural risk and unsupervised anomaly scoring.

These are workflow scale and methodology facts, not evidence of fraud-detection precision, prevented loss or reduced investigation time. No labelled ground truth or operational case outcomes are claimed for the current dashboard.

## Recommended operating workflow

1. Review aggregate anomaly and segment patterns.
2. Open a candidate's Customer 360 profile and inspect the underlying behavioural indicators.
3. Explore shared device/IP connections and graph context.
4. Compare the candidate with peers in the same cohort.
5. Record the evidence and have a trained reviewer make the final decision.
6. Track review outcomes to build a labelled evaluation set for future precision@K, analyst yield and time-to-resolution measurement.

## Metrics to collect before claiming business impact

- Precision@K and confirmed-case yield from adjudicated investigation outcomes.
- False-positive rate by customer segment.
- Median analyst review time and time-to-resolution.
- Confirmed loss or prevented loss only where a defensible counterfactual exists.
- Stability of anomaly volumes and scores over time.

## Stakeholder handoff

For a fraud-operations/product review, agree on alert capacity, escalation thresholds, review reason codes, analyst feedback capture and a process for appeal. A shadow evaluation should precede any workflow integration.

## Collaboration and provenance

This is an independently developed portfolio project using a public dataset. The operations stakeholder above is the intended user persona; it is **not a claim of client delivery, team deployment, or external stakeholder sign-off**.

**Source of truth:** repository [README](../README.md) and research workflow artifacts.