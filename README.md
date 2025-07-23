# AI Treinplanning (ProRail Afstudeerproject)

Dit project onderzoekt hoe Reinforcement Learning (Q-learning) kan worden toegepast om automatisch een conflictvrije treinroute te plannen binnen een bestaand spoorwegnetwerk. De focus ligt op goederenplanning in de Specifieke Dagen (SD) fase van de dienstregeling.

## Functionaliteiten

- Automatisch plannen van een treinroute van A naar B
- Vermijden van conflicten met bestaande reserveringen
- Headway wordt gerespecteerd (veiligheidsmarge tussen treinen)
- Off-policy training op basis van historische data

## Gebruikte AI-techniek

- **Q-learning** (tabelgebaseerd)
- Epsilon-greedy actie selectie
- Bellman-updates tijdens training
- Ondersteuning voor off-policy pretraining

## Structuur

```plaintext
├── main.py                    # Experimenteer en visualiseer resultaten
├── TrainPlanningEnv.py        # Simulatieomgeving met conflictchecks en constraints
├── QLearningAgent.py          # Q-learning implementatie
├── planned_route.csv          # Laatste geplande route (automatisch geëxporteerd)
├── modelroute_per_trein.csv   # Trainingsdata (historisch)
├── qtable.csv                 # Q-waarden per state-action combinatie (automatisch geëxporteerd)
