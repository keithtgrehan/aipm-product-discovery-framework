# Omission Scoring (Experiment 1 MVP)

For each benchmark item (`id`), evaluate model output against expected critical items.

## Missing rule

An item is **missing** when either of the following applies:

- A critical condition is not mentioned.
- A critical qualifier is not addressed.

## Metrics

- `omission_score = (missing_conditions + missing_qualifiers) / total_expected`
- `binary_fail = true` if any critical condition or qualifier is missing, else `false`

Where:

- `total_expected = len(critical_conditions) + len(critical_qualifiers)`
