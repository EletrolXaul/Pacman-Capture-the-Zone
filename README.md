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

### Installazione Dipendenze
```bash
# Aggiorna pip
python -m pip install --upgrade pip

# Installa librerie richieste
pip install numpy matplotlib tensorflow keras

###💻 Utilizzo
Avvia il Training
```bash
python main.py

### Modifica Algoritmo
In main.py, modifica:
```bash
test = Test.Q_LEARNING     # Q-Learning base
#test = Test.SARSA        # SARSA
#test = Test.DEEP_Q       # Deep Q-Network

### 📊 Visualizzazione
- **Mappa del labirinto in tempo reale**
-**Heatmap delle azioni ottimali**
-**Grafici di performance:**
  - Win rate per episodio
  - Reward cumulativo
  - Comparazione algoritmi
###👥 Autori
Celani
Pizzoli
###📄 Licenza
MIT License

###🤝 Contributing
Le pull request sono benvenute. Per modifiche importanti, apri prima un issue per discutere cosa vorresti cambiare.

