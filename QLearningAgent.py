import random
import pandas as pd
import ast
from datetime import datetime, timedelta
import os

class QLearningAgent:
    """
    Q-learning agent met occupancy-based state-representatie.
    De state komt altijd van env.get_state(), en de Q-table is een dict:
        self.q[state][action] = q_value
    """

    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        min_epsilon=0.01,
        decay=0.995,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.q = {}

    def _init_state_actions(self, state):
        """
        Zorg dat er voor elke mogelijke actie in deze state
        een Q-entry bestaat (initialiseer op 0).
        """
        if state not in self.q:
            self.q[state] = { action: 0.0 for action in self.env.possible_actions(state) }

    def choose_action(self, state):
        """
        Epsilon-greedy: met kans epsilon explore, anders exploit.
        """
        self._init_state_actions(state)
        if random.random() < self.epsilon:
            return random.choice(list(self.q[state].keys()))
        # kies actie met hoogste Q
        max_q = max(self.q[state].values())
        best = [a for a,q in self.q[state].items() if q == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state):
        """
        Voer de Bellman-update uit:
          Q(s,a) := Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        self._init_state_actions(next_state)
        old_q = self.q[state][action]
        future_q = max(self.q[next_state].values()) if self.q[next_state] else 0.0
        self.q[state][action] = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

    def update_q_value(self, qtable, station, time, action, q_value):
        mask = (
            (qtable["station"] == station) &
            (qtable["time"] == time) &
            (qtable["action"] == str(action))  # actie wordt als string opgeslagen
        )

        if mask.any():
            qtable.loc[mask, "q_value"] = q_value
        else:
            qtable.loc[len(qtable)] = [station, time, str(action), q_value]

        return qtable

    def decay_epsilon(self):
        """Verlaag epsilon tot min_epsilon via decay-factor."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def train(self, episodes=1000, export_qtable=True, csv_path='qtable.csv', trainings_data=None):
        """
        Train de agent. Indien trainings_data is opgegeven, voer eerst off-policy training uit.
        Retourneert metrics: {'rewards': [...], 'steps': [...]}
        Exporteert optioneel de Q-table naar CSV.
        """

        if trainings_data:
            df = pd.read_csv(trainings_data, sep=';', dtype=str)
            df["plantijd_dt"] = pd.to_datetime(df["plantijd"], errors="coerce")
            df = df.sort_values(by=["treinnr", "plantijd_dt"])

            grouped = df.groupby(["treinnr", df["plantijd_dt"].dt.date])

            for (_, _), groep in grouped:
                groep = groep.reset_index(drop=True)
                for i in range(len(groep) - 1):
                    row_now = groep.loc[i]
                    row_next = groep.loc[i + 1]

                    station = row_now["drp"]
                    tijd = row_now["plantijd_dt"]
                    
                    time_in_min = tijd.hour * 60 + tijd.minute
                    next_station = row_next["drp"]
                    next_tijd = row_next["plantijd_dt"]
                    next_time_in_min = next_tijd.hour * 60 + next_tijd.minute

                    # bepaal actie
                    if row_now["drp"] == row_next["drp"]:
                        action = ("wait", int((row_next["plantijd_dt"] - row_now["plantijd_dt"]).total_seconds() // 60))
                    else:
                        reistijd = int((row_next["plantijd_dt"] - row_now["plantijd_dt"]).total_seconds() // 60)
                        action = (next_station, reistijd)

                    state = (station, time_in_min)
                    next_state = (next_station, next_time_in_min)
                    reward = 50 if action[0] != "wait" else 20

                    if self.env._conflict((station, next_station), time_in_min, next_time_in_min):
                        reward = 0                 
                            
                    # init Q entries
                    if state not in self.q:
                        self.q[state] = {}
                    if action not in self.q[state]:
                        self.q[state][action] = 0.0
                    if next_state not in self.q:
                        self.q[next_state] = {}
                        for a in self.env.possible_actions(next_state):
                            self.q[next_state][a] = 0.0

                    # Q-update
                    future_q = max(self.q[next_state].values()) if self.q[next_state] else 0.0
                    old_q = self.q[state][action]
                    self.q[state][action] = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

            print(f"Off-policy training gedaan op {len(df)} stappen.")
   
        metrics = {'rewards': [], 'steps': []}

        for ep in range(episodes):
            self.env.is_last_episode = (ep == episodes - 1)
            # reset en haal initiële state
            self.env.reset()
            state = self.env.get_state()
            done = False
            total_reward = 0

            # loop totdat de episode eindigt
            while not done:
                action = self.choose_action(state)
                _, reward, done = self.env.step(action)
                next_state = self.env.get_state()
                old_q = self.q[state][action]
                station = state[0]
                
                self.update(state, action, reward, next_state)                
                state = next_state
                total_reward += reward

            self.decay_epsilon()
            metrics['rewards'].append(total_reward)
            metrics['steps'].append(self.env.step_count)

        # 3) optioneel Q-table exporteren
        if export_qtable:

            # 1. Zet self.q om naar DataFrame
            rows = []
            base_minute = self.env.base_departure_minute
            for state, actions in self.q.items():
                station, rel_time = state
                abs_time = datetime.strptime("00:00", "%H:%M") + timedelta(minutes=base_minute + rel_time)
                time = abs_time.strftime("%H:%M")
                for action, q_val in actions.items():
                    rows.append({
                        "station": station,
                        "time": time,
                        "action": str(action),
                        "q_value": q_val
                    })

            new_df = pd.DataFrame(rows)
            new_df.set_index(["station", "time", "action"], inplace=True)

            # 2. Laad bestaande CSV als die er is
            if os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path)
                existing_df.set_index(["station", "time", "action"], inplace=True)
            else:
                existing_df = pd.DataFrame(columns=["station", "time", "action", "q_value"])
                existing_df.set_index(["station", "time", "action"], inplace=True)

            existing_df.sort_index(inplace=True)
            new_df.sort_index(inplace=True)

            # 3. Werk bestaande waardes bij en voeg nieuwe toe
            existing_df.update(new_df)
            combined_df = pd.concat([existing_df, new_df[~new_df.index.isin(existing_df.index)]])

            # 4. Wegschrijven
            combined_df.reset_index(inplace=True)
            combined_df.to_csv(csv_path, index=False)
            print(f"Q-table exported to {csv_path}")
            
        return metrics

    def load_qtable(self, csv_path):
        """
        Laad een Q-table uit CSV met kolommen: station, time, action, q_value.
        Zet tijd expliciet om naar HH:MM formaat.
        """
        df = pd.read_csv(csv_path)
        self.q = {}

        for _, row in df.iterrows():
            station = row['station']
            
            # Zorg dat tijd altijd HH:MM formaat heeft
            raw_time = str(row['time']).strip()
            time_obj = datetime.strptime(raw_time, "%H:%M" if len(raw_time) == 5 else "%H:%M:%S")
            time_in_min = time_obj.hour * 60 + time_obj.minute
            
            action = ast.literal_eval(row['action'])
            qv = float(row['q_value'])

            state = (station, time_in_min)
            if state not in self.q:
                self.q[state] = {}
            self.q[state][action] = qv

        print(f"Q-table loaded from {csv_path}")


    def plan_route(self):
        """
        Plan een route op basis van de Q-table (epsilon = 0).
        Geeft lijst van dicts met station, tijd (HH:MM), actie, en next_station.
        """
        self.epsilon = 0.0
        self.env.reset()
        done = False
        steps = []

        base_minute = self.env.base_departure_minute
        base_time = datetime.strptime("00:00", "%H:%M") + timedelta(minutes=base_minute)

        while not done:
            raw_state = self.env.get_state()
            station, t_min = raw_state
            # time_in_min = (base_time + timedelta(minutes=t_min)).strftime("%H:%M")
            # state = (station, time_in_min)
            state = (station, t_min)
            if state not in self.q:
                print(f"Geen Q-waarde beschikbaar voor state {state}, planning stopt.")
                break

            # Kies actie met hoogste Q
            best_action = max(self.q[state], key=self.q[state].get)

            prev_station = self.env.current
            prev_time = self.env.time

            _, _, done = self.env.step(best_action)
            curr_station = self.env.current
            curr_time = self.env.time

            abs_time = base_time + timedelta(minutes=curr_time)

            steps.append({
                'station': prev_station,
                'time': abs_time.strftime("%H:%M"),
                'action': 'wait' if prev_station == curr_station else 'move',
                'next_station': curr_station
            })

        return steps