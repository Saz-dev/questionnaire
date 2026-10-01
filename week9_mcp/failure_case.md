# Worker Failure Injection

Injected 500 error into coverage/exclusions worker for case c10-03.
Orchestrator behaviour: retried once, then degraded to partial answer using adjuster output only, without declaring coverage.
