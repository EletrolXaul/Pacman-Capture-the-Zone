# Pacman Capture the Zone

Un progetto di Reinforcement Learning dove un agente Pacman impara a navigare in un labirinto per raggiungere gli obiettivi.

## Descrizione

Questo progetto implementa un sistema di apprendimento per rinforzo (RL) dove un agente Pacman deve imparare a muoversi in un labirinto per raggiungere uno dei quattro obiettivi disponibili. L'agente utilizza diversi algoritmi di RL come Q-Learning, SARSA e Deep Q-Network per imparare la strategia ottimale.

### Caratteristiche principali:
- Implementazione di diversi algoritmi RL:
  - Q-Learning
  - Q-Learning con Eligibility Trace
  - SARSA
  - SARSA con Eligibility Trace 
  - Deep Q-Network con Experience Replay
- Sistema di reward/penalty:
  - Reward positivo (+10.0) per raggiungimento obiettivo
  - Penalità per movimento (-0.05)
  - Penalità per celle già visitate (-0.25)
  - Penalità per movimenti impossibili (-0.75)
- Visualizzazione grafica dell'apprendimento
- Analisi delle performance con grafici di win rate e reward cumulativo

## Requisiti

```python
pip install numpy matplotlib tensorflow keras