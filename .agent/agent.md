# Coding Agent Guidance

## Project Context
This is an **agent teaming research project** focused on testing different LLM agent collaboration strategies for medical QA tasks.

**Research Goal**: Identify which agent communication patterns produce the best results on medical QA datasets.

---

## Design Philosophy

### Flexibility First
- The specific design is **not yet fixed** - expect changes
- Build modular, swappable components
- Avoid over-engineering early on
- Make it easy to test different approaches

### Research-Oriented Code
- **Experimentation > Production**: Prioritize rapid iteration over robustness
- **Logging > Performance**: Track everything for analysis
- **Configurability > Hardcoding**: Use config files for easy experiment variations
- **Reproducibility**: Ensure experiments can be re-run with same results

---

## Code Structure Principles

### Modularity
- Separate concerns: agents, communication protocols, datasets, evaluation
- Make components pluggable (easy to swap protocols, agent types, etc.)
- Keep interfaces simple and consistent

### Simplicity
- Start with minimal viable implementations
- Add complexity only when needed
- Prefer clarity over cleverness
- Document non-obvious design choices

### Extensibility
- Design for adding new collaboration protocols
- Support multiple datasets easily
- Allow new agent roles to be defined
- Enable custom evaluation metrics

---

## When Helping with This Project

### Do:
- ✅ Ask clarifying questions about research goals
- ✅ Suggest simple starting points
- ✅ Provide modular, reusable code
- ✅ Explain trade-offs between approaches
- ✅ Keep code readable and well-commented
- ✅ Focus on what enables experimentation

### Don't:
- ❌ Over-engineer solutions prematurely
- ❌ Lock into specific designs too early
- ❌ Add unnecessary complexity
- ❌ Assume production-level requirements
- ❌ Implement features not yet needed

---

## Key Components to Support

1. **Agent System**: Flexible agent classes with different roles
2. **Communication Protocols**: Pluggable collaboration strategies
3. **Dataset Integration**: Easy loading of medical QA datasets
4. **Logging/Tracking**: Comprehensive interaction logging
5. **Evaluation**: Metrics and analysis tools
6. **Experiment Management**: Config-driven experiment running

---

## Current Status
- **Phase**: Early exploration
- **Design**: Evolving
- **Priority**: Enable rapid experimentation

---

*This guidance helps the coding assistant understand the research context and provide appropriate support.*
