# RULES FREEZE — proposed v2

- Freeze date: 2026-10-18
- Status: FROZEN FOR THE NEXT EXPERIMENTAL RUNS
- Frozen configuration: `config/business_rules.json`
- SHA-256 of frozen configuration: a9fa1fbccaa933f754221afcc5aa301bcd981c8caa1b126cb7b6820509ec5b02
- Baseline configuration retained: `config/business_rules_original.json`

## What freezing means

For subsequent Big Data experiments, do not change rule definitions, counting scope, timing boundary, notification recipient/channel, or thresholds in proposed v2. Machine Learning experiments may change independently, but must not change the frozen business rules. If a rule change is later required, create a separately named/versioned candidate (for example proposed v3), explain the reason, update the scenario tests, and compare outputs rather than silently modifying this frozen version.

This freeze is for experimental consistency, not bank approval or operational deployment. Proposed v2 includes interpretations that still need the business owner's confirmation.

## Frozen rule definitions

### R1 — customer notification

Create an SMS notification for a transaction that is a cash withdrawal outside the user's `home_country`, has status `APPROVED` or `DECLINED`, and is the user's first *foreign transaction of any type* for that business day. There is no minimum amount. An earlier foreign transaction of any type suppresses R1 for the current withdrawal. R1 is a customer notification, not a fraud verdict.

### R2 — cross-country review indicator

Create an email record for the Fraud team when a qualifying foreign cash withdrawal is associated with another transaction by the same user in a different country within an absolute time difference strictly less than 90 minutes. The other event may occur before or after the withdrawal and need not itself be a cash withdrawal; it may be domestic or foreign relative to the user's home country. The qualifying event must be a cash withdrawal outside `home_country`. An alert is recorded at most once per qualifying withdrawal. Exactly 90 minutes does not trigger the rule. Country-code difference alone is not proof of impossible travel or fraud.

### R3 — foreign cash-withdrawal velocity / cumulative amount

Create an email record for the Fraud team when a qualifying foreign cash withdrawal becomes the user's fifth foreign cash withdrawal on that business day, or the cumulative amount of foreign cash withdrawals reaches or exceeds IDR 6,000,000 equivalent including the current event. In proposed v2, both count and amount scope are foreign cash withdrawals only, not all transaction types. The count alert and amount-threshold alert each occur at most once per user per business day. Thresholds of 5 and IDR 6,000,000 are inherited from the supplied case, not universal thresholds established by public research.

## Frozen assumptions / open confirmations

- `event_time` is normalized to the same business timezone before processing.
- `amount_idr_equiv` is already converted to IDR equivalent; the rule engine does not convert currencies.
- In the current prototype, R3 may count declined/pending cash-withdrawal events because the original case does not specify status scope. If the rule is intended to count only dispensed cash, this requires an explicitly versioned change.
- R1, R2, and R3 are tested using synthetic scenarios. Test pass means implementation matches these assumptions; it does not show operational fraud-detection effectiveness.
- The Kaggle dataset does not contain all fields needed to evaluate these bank rules against Kaggle labels.
- The previous rule definition is retained in `config/business_rules_original.json`; proposed v2 is retained in `config/business_rules.json` and must not be silently edited during the next experiments.
