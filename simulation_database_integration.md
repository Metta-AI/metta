# Integrating Your Simulation Database with the Repository

## Overview

Your SQLAlchemy models are **excellent candidates for persistence** and align perfectly with the repository's existing data patterns. Here's how to integrate them effectively.

## Repository Database Patterns

### 1. PostgreSQL with Raw SQL Migrations
Your repository uses **PostgreSQL** with a sophisticated migration system:

```python
# From metta/app_backend/src/metta/app_backend/schema_manager.py
class SqlMigration(Migration):
    def __init__(self, version: int, description: str, sql_statements: List[LiteralString]):
        self._version = version
        self._description = description
        self._sql_statements = sql_statements
```

### 2. Prisma ORM Integration
The library component uses **Prisma** for Node.js/TypeScript:

```prisma
# From metta/library/prisma/schema.prisma
model User {
  id    String @id @default(cuid())
  name  String?
  email String? @unique
  // ... relationships
}
```

### 3. Agent State Persistence Patterns
Multiple agent systems already persist states:

- **CodeBot**: `BotState` and `WorkCycleRecord`
- **PyTorch Agents**: LSTM memory states with `get_memory()`/`set_memory()`
- **Policy Records**: Metadata and checkpoint management

## Recommended Integration Strategy

### Option 1: Extend Existing PostgreSQL Schema
**Best for consistency with current patterns**

1. **Create Migration Files** for your models:
```python
# simulation_models_migration_001.py
from metta.app_backend.schema_manager import SqlMigration

# Generate CREATE TABLE statements from your SQLAlchemy models
migration_001 = SqlMigration(
    version=1,
    description="Add simulation database schema",
    sql_statements=[
        """
        CREATE TABLE simulations (
            simulation_id VARCHAR(64) PRIMARY KEY,
            experiment_id VARCHAR(64),
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP NULL,
            status VARCHAR(50) DEFAULT 'pending',
            parameters JSONB NOT NULL,
            results_summary JSONB NULL,
            simulation_db_path VARCHAR(255) NOT NULL
        );
        """,
        # ... more CREATE TABLE statements
    ]
)
```

2. **Use Repository's Database Connection**:
```python
# Leverage existing connection patterns
from metta.app_backend.schema_manager import run_migrations_async
from psycopg import AsyncConnection

async def setup_simulation_db(conn: AsyncConnection):
    migrations = [migration_001, migration_002, ...]
    await run_migrations_async(conn, migrations)
```

### Option 2: Create Separate Simulation Database
**Best for isolation and specialized needs**

1. **Create Dedicated Database Connection**:
```python
# simulation_db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# SQLite for simplicity, or PostgreSQL for production
SQLALCHEMY_DATABASE_URL = "sqlite:///./simulation.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # Only for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)
```

2. **Integration with Repository Services**:
```python
# simulation_service.py
from .simulation_db import SessionLocal
from .models import Simulation, AgentStateModel

class SimulationService:
    def __init__(self):
        self.db = SessionLocal()

    def record_agent_state(self, agent_state_data: dict) -> AgentStateModel:
        """Record agent state with repository integration."""
        state = AgentStateModel(**agent_state_data)

        # Link to repository's experiment system
        if experiment_id := agent_state_data.get('experiment_id'):
            state.simulation_id = experiment_id

        self.db.add(state)
        self.db.commit()
        self.db.refresh(state)
        return state
```

## Data Flow Architecture

### Recommended Integration Pattern

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Simulation    │───▶│  Agent States    │───▶│   Repository    │
│   Engine        │    │  Database        │    │   Services      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Real-time     │    │   Analytics      │    │   Experiment    │
│   Monitoring    │    │   & Metrics      │    │   Management    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Integration Points

1. **Experiment Management**:
   - Link simulations to repository's experiment system
   - Use existing experiment metadata structure
   - Enable cross-simulation comparisons

2. **Metrics Integration**:
   - Feed simulation metrics into repository's analytics
   - Use existing metric collection patterns
   - Enable real-time monitoring

3. **Agent State Correlation**:
   - Connect agent states to repository's agent systems
   - Enable state comparison across different agent implementations
   - Support transfer learning between systems

## Implementation Steps

### Step 1: Database Setup
```python
# 1. Choose integration approach (extend existing vs separate DB)
# 2. Set up database connection using repository patterns
# 3. Create migration files for schema
```

### Step 2: Model Integration
```python
# 1. Adapt your SQLAlchemy models to repository conventions
# 2. Add foreign keys to link with existing tables
# 3. Implement repository service layer
```

### Step 3: Service Integration
```python
# 1. Create simulation service that integrates with repository
# 2. Implement data export/import with repository systems
# 3. Add monitoring and analytics hooks
```

### Step 4: Testing and Validation
```python
# 1. Create integration tests with repository test patterns
# 2. Validate data consistency across systems
# 3. Performance testing with repository load patterns
```

## Benefits of Integration

### Data Consistency
- **Unified Schema**: Single source of truth for agent states
- **Cross-System Queries**: Query across simulation and repository data
- **Audit Trail**: Complete history of agent state changes

### Analytics Capabilities
- **Comparative Analysis**: Compare different simulation runs
- **Performance Tracking**: Monitor agent performance over time
- **Pattern Recognition**: Identify successful agent strategies

### Scalability
- **Distributed Processing**: Leverage repository's distributed systems
- **Storage Optimization**: Use repository's data management patterns
- **Backup & Recovery**: Integrated with repository's backup systems

## Repository Alignment

Your models align well with existing patterns:

| Your Models           | Repository Equivalent | Integration Benefit         |
| --------------------- | --------------------- | --------------------------- |
| `AgentModel`          | `BotState` (CodeBot)  | Unified agent metadata      |
| `AgentStateModel`     | PyTorch memory states | Consistent state management |
| `SimulationStepModel` | Repository metrics    | Enhanced analytics          |
| `ExperimentModel`     | Existing experiments  | Seamless integration        |

## Conclusion

**Yes, absolutely persist this data!** Your SQLAlchemy models represent a sophisticated simulation database that will integrate well with the repository's existing patterns. The recommended approach is to extend the existing PostgreSQL schema using the repository's migration system for consistency and maintainability.

This integration will enable:
- **Unified data management** across simulation and repository systems
- **Advanced analytics** and comparison capabilities
- **Scalable storage** with repository's infrastructure
- **Cross-system insights** and transfer learning opportunities
