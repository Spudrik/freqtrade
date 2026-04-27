# Explorer agent rules

Explorer objective:
Optimise selected strategy families or modes, validate against selected windows, and apply a challenger only when ExplorerAcceptanceScoreV1 improves and drawdown guards pass.

Normal Explorer workflow:
- Target type: family or mode
- Target selection: random or specific
- Search breadth: targeted or open
- Training windows
- Validation windows

Do not add new selection systems.
Do not add new scoring modes.
Do not reintroduce promotion gates.
Do not use trade count as an acceptance gate.
Do not use losing-window count as an acceptance gate.
Trade count and losing-window count are display-only diagnostics.
Drawdown is the primary hard risk limiter.
Root-cause first: fix mapping/config/state issues at source before adding fallback/workaround code.
Do not add runtime rescue logic by default; use one-time migration when needed.
If a workaround is truly unavoidable, stop and ask for explicit approval before adding it.
