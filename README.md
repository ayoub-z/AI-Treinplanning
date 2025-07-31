# AI Treinplanning (ProRail Afstudeerproject)

Dit project onderzoekt hoe Reinforcement Learning (Q-learning) kan worden toegepast om automatisch conflictvrije treinroutes te plannen binnen het Nederlandse spoorwegnetwerk. De focus ligt op goederenplanning in de Specifieke Dagen (SD)-fase van de dienstregeling.

## Functionaliteiten

- Automatisch plannen van een conflictvrij treinpad van A naar B
- Detectie en vermijden van conflicten met bestaande reserveringen
- Headway (veiligheidsmarge tussen treinen) wordt gerespecteerd
- Off-policy training op basis van historische data
- Export van geplande routes en performance statistieken

## Gebruikte AI-techniek

- Q-learning (tabelgebaseerd reinforcement learning)
- Epsilon-greedy actie selectie voor exploratie en exploitatie
- Bellman-updates tijdens training
- Off-policy pretraining op historische ritdata

## Bestandsstructuur

```
├── main.py                    # Experimenteer en visualiseer resultaten
├── TrainPlanningEnv.py        # Simulatieomgeving met conflictchecks en constraints
├── QLearningAgent.py          # Q-learning implementatie
├── planned_route.csv          # Laatste geplande route (AI output)
├── modelroute_per_trein.csv   # Trainingsdata (historisch)
├── qtable.csv                 # Q-waarden per state-action combinatie (AI output)
```

## Installatie & Gebruik

### 1. Clone de repository

```bash
git clone <repository-url>
cd <projectfolder>
```

### 2. Installeer de benodigde packages

```bash
pip install -r requirements.txt
```
Of handmatig:
```bash
pip install pandas numpy matplotlib
```

### 3. Start het experiment

```bash
python main.py
```

## Output & Visualisatie

- Geplande routes worden opgeslagen in `planned_route.csv`
- Q-table (AI-leerproces) wordt geëxporteerd naar `qtable.csv`
- Resultaten en learning curves als PNG-grafiek in de projectmap

## Aanpasbaar

Alle belangrijke instellingen (zoals headway, episodes, departure window) zijn eenvoudig aan te passen via `main.py`.

## Projectinfo

Dit project is onderdeel van een afstudeeronderzoek bij ProRail, uitgevoerd in 2025 door Ayoub Zouin.