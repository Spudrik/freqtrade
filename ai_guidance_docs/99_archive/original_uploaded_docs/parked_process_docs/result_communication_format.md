# Result Communication Format

Use this format when explaining FreqAI, direct-test, confluence, orderbook, structure, or feature-discovery results to the user.

## Core Rule

Explain the research as one-hour trader decision rows, not as machine-learning jargon.

Every result explanation should answer these questions in order:

1. Trader question:
   - What real market question are we asking?
   - Example: "When price breaks resistance with rising volume and weak sell walls, does it keep going up?"
2. What we looked at:
   - List the visible market conditions used as inputs.
   - Example: resistance nearby, volume rising, orderbook wall weakening, price breaking the range.
3. What we asked afterwards:
   - Explain the answer/label in plain language.
   - Example: "Did price keep going up over the next 6 hours?"
4. What happened:
   - State the result without relying on acronyms first.
   - Example: "This worked better than price alone in Q4, but did not hold up in the recent window."
5. Decision:
   - Use one of: worth more testing, reject for now, needs better features, needs more data, or promote to FreqAI.

## Plain-Language Translations

1. Feature:
   - "Something we knew at that hour."
   - Example: price near resistance, volume rising, orderbook wall disappearing.
2. Label:
   - "The answer sheet for what happened after that hour."
   - Example: price kept going up, breakout failed, price dropped 3%.
3. Entry:
   - Avoid this word unless discussing a real strategy.
   - Prefer "decision point" or "hour a trader might act."
4. Follow-through:
   - "The move kept going after it started."
5. Event scope:
   - "Only looking at rows where this kind of market situation was active."
6. AUC:
   - "How well the score ranked good outcomes above bad outcomes."
   - Always translate the number into: poor, weak, useful, strong, or too small to trust.
7. Top bucket:
   - "The rows the model liked most."
   - Explain whether those rows actually had more good outcomes than ordinary rows.

## Required Result Summary Shape

Use this shape before detailed metrics:

1. Trader question:
2. What we looked at:
3. What we asked afterwards:
4. Result in plain English:
5. Numbers that support it:
6. Verdict:
7. Next step:

## Avoid

1. Do not lead with profile names, AUC, AP, Spearman, model class, label names, or internal column names.
2. Do not say "post-break label" without explaining it as "after a break starts, did the move keep going or fail?"
3. Do not describe a result as good or bad without saying what a trader would have seen on the chart.
4. Do not collapse complex confluence into one indicator or one timeframe.

