# 🎮 Pacman Capture the Zone

## 📝 Descrizione
Un progetto di Reinforcement Learning dove un agente Pacman impara a navigare in un labirinto per raggiungere gli obiettivi in modo ottimale. Il progetto utilizza diversi algoritmi di apprendimento per rinforzo (RL) per insegnare all'agente come muoversi efficacemente nel labirinto.

## 🎯 Funzionalità
- L'agente deve raggiungere uno dei quattro obiettivi disponibili
- Sistema di apprendimento basato su reward/penalty
- Visualizzazione real-time del processo di apprendimento
- Training configurabile con diversi algoritmi RL

## 🧠 Algoritmi Implementati
- **Q-Learning**
  - Implementazione base 
  - Versione con Eligibility Trace
- **SARSA**
  - Implementazione base
  - Versione con Eligibility Trace
- **Deep Q-Network**
  - Con Experience Replay
  - Salvataggio e caricamento modelli

## 💎 Sistema di Reward/Penalty
| Azione | Reward/Penalty |
|--------|---------------|
| Raggiungimento obiettivo | +10.0 |
| Movimento base | -0.05 |
| Cella già visitata | -0.25 |
| Movimento impossibile | -0.75 |

## 🚀 Installazione

### Prerequisiti
- Python 3.7-3.11
- pip (gestore pacchetti Python)

### Setup Ambiente Virtuale (Consigliato)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac 
python -m venv venv
source venv/bin/activate