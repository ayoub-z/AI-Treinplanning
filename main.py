from TrainPlanningEnv import TrainPlanningEnv
from QLearningAgent import QLearningAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def smooth_exp(data, alpha=0.1):
    """
    Pas een exponentiële moving average (EMA) toe voor vloeiende curve.

    Args:
        data (array-like): Originele data.
        alpha (float): Smoothing factor tussen 0 en 1 (kleiner = vloeiender).

    Returns:
        np.ndarray: Gesmoothde data met dezelfde lengte.
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def run_experiments(num_runs: int, episodes: int, env_config: dict, trainings_data: str = None) -> np.ndarray:
    """
    Voer meerdere trainingsruns uit. Alleen bij de eerste run wordt off-policy training toegepast.
    """
    all_rps = np.zeros(episodes)
    # shared_q = None  # om Q-table door te geven

    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs}, Episodes={episodes} ===")
        env_config['print_state'] = (run == num_runs - 1)
        env = TrainPlanningEnv(**env_config)
        agent = QLearningAgent(env)

        # Alleen bij eerste run: off-policy pretraining
        if run == 0 and trainings_data:
            results = agent.train(episodes=0, export_qtable=True, trainings_data=trainings_data)
        results = agent.train(episodes=episodes, export_qtable=True)
        rewards = np.array(results['rewards'])
        rps = rewards
        all_rps += rps

    return all_rps / num_runs

if __name__ == '__main__':
    # Configuratie netwerk
    env_config = {
        'nodes': ['Ah', 'Ahp', 'Va', 'IJbww', 'Wtv', 'Dvn', 'Zv', 'Zvbtwa', 'Brdvno', 'Zvo', 'Zvg', 'Ahpr', 'did'],
        'edges': {
            ('Ah', 'Ahp'): 2,
            ('Ahp', 'Va'): 1,
            ('Va', 'IJbww'): 2,
            ('Va', 'Ahpr'): 1,
            ('IJbww', 'Wtv'): 2,
            ('Wtv', 'Dvn'): 4,
            ('Dvn', 'Zv'): 4,
            ('Zv', 'Zvbtwa'): 1,
            ('Zv', 'did'): 4,
            ('Zvbtwa', 'Brdvno'): 4,
            ('Zvbtwa', 'Zvo'): 1,
            ('Zvo', 'Zvg'): 1,         
        },
        'start': 'Ah',
        'destination': 'Zvg',
        'reserved_segments': { # tijden zijn in minuten na departure. In dit voorbeeld 14u
            ('Ah','Ahp'): [(0, 2)],
            ('Ahp','Va'): [(2, 3)],
            ('Va','IJbww'): [(3, 5)],
            ('IJbww','Wtv'): [(5, 7)],
            ('Wtv','Dvn'): [(7, 11)],
            ('Dvn','Zv'): [(11, 15)],
            ('Zv','Zvbtwa'): [(15, 16)],
            ('Zvbtwa','Brdvno'): [(16, 20)],
        },
        'headway': 2, # veiligheids marge (in minuten) tussen treinen
        'max_wait': 4, # max min dat een trein op een station mag wachten
        'max_time': 50, # max totale reistijd 
        'max_steps': 50, # maximum totaal aantal steps
        'departure_window': ("14:00", "14:30"),  # gewenst tijdsvenster voor vertrek
}
    
    # Gebruikersparameters  
    num_runs = 100
    episodes = 2000
    alpha = 0.05     # EMA smoothing factor voor grafiek (kleiner = vloeiender)

    trainings_data = "modelroute_per_trein.csv"

    # Run experiments
    avg_rps = run_experiments(num_runs, episodes, env_config, trainings_data)

    route_df = pd.read_csv("planned_route.csv")  # door AI gegenereerd
    reserved_segments = env_config["reserved_segments"]    # optioneel, mag ook None

    # Exponentieel smoothen
    smoothed_rps = smooth_exp(avg_rps, alpha=alpha)

    # Plot raw en gesmoothde data
    plt.figure(figsize=(8,5))
    plt.plot(range(1, episodes+1), avg_rps, alpha=0.2, label='Avg. reward')
    plt.plot(range(1, episodes+1), smoothed_rps, linewidth=2, label=f'Smoothed')
    plt.xlabel('Episode')
    plt.ylabel('Avg. reward per episode')
    plt.title(f'Avg. reward per episode across {num_runs} runs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_reward_per_ep_ema.png')
    plt.show()