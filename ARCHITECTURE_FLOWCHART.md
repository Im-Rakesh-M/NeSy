# NESO-DT Architecture Flowchart

This document describes the runtime flow of the `NeSy` digital twin code base.

## High-level flow

The system starts in `main.py`, initializes the digital twin orchestrator, loads models, connects to Neo4j, and starts a state sync loop that processes delivery and machine risk events.

## Mermaid flowchart

```mermaid
flowchart TD
    A[main.py] --> B[NESODigitalTwin init]
    B --> C[Neo4jConnection singleton]
    B --> D[Load neural models]
    B --> E[MessageBus]
    B --> F[Instantiate agents]

    F --> G[LogisticsAgent]
    F --> H[MaintenanceAgent]
    F --> I[InventoryAgent]
    F --> J[HRAgent]
    F --> K[RegulatoryAgent]
    F --> L[SOPAgent]
    F --> M[BOMAgent]

    A --> N[run_sync_loop()]
    N --> O[Query Neo4j for high-risk deliveries]
    N --> P[Query Neo4j for high-risk machines]

    O --> Q[process_delivery_event()]
    Q --> R[DeliveryForecaster.predict()]
    R --> S[Publish DELIVERY_RISK]
    S --> G
    S --> J
    S --> L
    S --> K

    P --> T[process_machine_event()]
    T --> U[MachineForecaster.predict()]
    U --> V[Publish MACHINE_RISK]
    V --> H
    V --> M
    V --> L
    V --> K

    G --> W[Logistics decision tree]
    W --> X{Delay & buffer check}
    X -->|Low risk| Y[MONITOR_ONLY]
    X -->|Buffer OK| Z[BUFFER_ALERT]
    X -->|Buffer insufficient| AA[CASCADE_ALERT / REROUTE]

    Z --> I
    Z --> M
    I --> AB[RESTOCK_DIRECTIVE or CASCADE_ALERT]

    H --> AC[Maintenance decision tree]
    AC --> AD{Failure probability / regulatory hours}
    AD --> AE[HALT_LINE / CASCADE_ALERT]
    AD --> AF[MANDATORY_MAINTENANCE / COMPLIANCE_BREACH]
    AD --> AG[SCHEDULE_MAINTENANCE / CASCADE_ALERT]
    AD --> AH[FLAG_MAINTENANCE_WINDOW]
    AD --> AI[CONTINUE_OPERATION]

    J --> AJ[Staffing & regulatory check]
    AJ --> AK[SHIFT_GAP or OVERTIME alert]
    AJ --> AL[COMPLIANT]
    AJ --> AM[COMPLIANCE_BREACH]

    L --> AN[SOP rule evaluation]
    AN --> AO[SOP_VETO or SOP_WARNING]
    AN --> AP[COMPLIANT]
    AO --> K

    M --> AQ[BOM impact assessment]
    AQ --> AR[AT_RISK graph relationships]
    AQ --> AS[CASCADE_ALERT if critical]

    AB --> K
    AL --> K
    AM --> K
    AE --> K
    AF --> K
    AG --> K
    AH --> K
    AS --> K

    K --> AT[Audit & compliance logging]
    AT --> AU[Neo4j AuditEntry nodes]
    AT --> AV[print_audit_summary()]

    subgraph Graph
        C
        O
        P
        AQ
        AT
    end

    E --> S
    E --> V
    E --> Z
    E --> AB
    E --> AK
    E --> AO
    E --> AS
```

## Legend

- `main.py` starts the orchestrator.
- `NESODigitalTwin` loads models, initializes the shared `MessageBus`, and starts the sync loop.
- `Neo4jConnection` is a singleton used by the orchestrator and all agents.
- Delivery and machine events are fetched from Neo4j and passed through neural predictors.
- The `MessageBus` publishes events to subscribed agents.
- Agents use `BaseAgent` helpers to query graph data, log audits, and publish follow-up messages.
- `RegulatoryAgent` and `SOPAgent` are interceptors that evaluate all relevant topics for compliance.
- Final outputs are audit entries written back to Neo4j and printed by `print_audit_summary()`.

## Notes

- The flowchart emphasizes the main runtime pipeline rather than every internal helper method.
- For more detail, inspect `agents/base_agent.py`, `agents/logistics_agent.py`, `agents/maintenance_agent.py`, `agents/inventory_agent.py`, `agents/hr_agent.py`, `agents/regulatory_agent.py`, `agents/sop_agent.py`, and `agents/bom_agent.py`.
