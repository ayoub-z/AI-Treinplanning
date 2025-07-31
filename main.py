from TrainPlanningEnv import TrainPlanningEnv
from QLearningAgent import QLearningAgent
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

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

def plot_planning_vs_reservations(route_df, reserved_segments=None, starttime_str="14:00"):
    # Zet tijd om
    starttime = pd.to_datetime(starttime_str, format="%H:%M")
    route_df["time"] = pd.to_datetime(route_df["time"], format="%H:%M")

    # Extract alle stations
    stations = set(route_df["station"]) | set(route_df["next_station"])
    if reserved_segments:
        for (a, b), segments in reserved_segments.items():
            stations.add(a)
            stations.add(b)

    station_order = ['Ah', 'Ahp', 'Ahpr', 'Va', 'IJbww', 'Wtv', 'Dvn', 'Zv', 'did', 'Zvbtwa', 'Zvo', 'Zvg', 'Brdvno']
    station_order = list(reversed(station_order))
    station_y = {station: i for i, station in enumerate(station_order)}

    fig, ax = plt.subplots(figsize=(14, 6))

    # 🟦 AI-planning (rood/blauw)
    for i, row in route_df.iterrows():
        x = row["time"]
        y = station_y[row["station"]]
        ax.plot(x, y, 'o', color='red', markersize=14, zorder=10)
        ax.text(x, y + 0.5, row["time"].strftime("%H:%M"), fontsize=11, ha='center', va='bottom', fontweight='bold')
        ax.text(x, y - 0.5, row["station"], fontsize=10, ha='center', va='top', color="black")

    for i, row in route_df[:-1].iterrows():
        x1 = row["time"]
        y1 = station_y[row["station"]]
        x2 = route_df.iloc[i+1]["time"]
        y2 = station_y[route_df.iloc[i+1]["station"]]
        if row["station"] == route_df.iloc[i+1]["station"]:
            y2 = y1
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=5, color="blue", mutation_scale=30), zorder=5
        )

    # 🟧 Reserved segments (oranje/groen)
    if reserved_segments:
        plotted_points = set()
        for (from_station, to_station), times in reserved_segments.items():
            for t_start, t_end in times:
                x1 = starttime + timedelta(minutes=t_start)
                x2 = starttime + timedelta(minutes=t_end)
                y1 = station_y.get(from_station)
                y2 = station_y.get(to_station)

                if y1 is None or y2 is None:
                    continue  # skip onbekende stations

                if (from_station, x1) not in plotted_points:
                    ax.plot(x1, y1, 'o', color='orange', markersize=14, zorder=12)
                    ax.text(x1, y1 + 0.5, x1.strftime("%H:%M"), fontsize=11, ha='center', va='bottom')
                    plotted_points.add((from_station, x1))

                if (to_station, x2) not in plotted_points:
                    ax.plot(x2, y2, 'o', color='orange', markersize=14, zorder=12)
                    ax.text(x2, y2 + 0.5, x2.strftime("%H:%M"), fontsize=11, ha='center', va='bottom')
                    plotted_points.add((to_station, x2))

                ax.annotate("",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=5, color="green", mutation_scale=30), zorder=11
                )

    ax.set_yticks(list(station_y.values()))
    ax.set_yticklabels([s for s in station_order], fontsize=12)
    ax.set_xlabel("Tijd", fontsize=13)
    ax.set_ylabel("Dienstregelpunten (stations)", fontsize=13)
    ax.set_title("Tijd-locatie grafiek: geplande route vs. reserveringen", fontsize=15)
    ax.set_ylim(-1, len(station_y))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    fig.autofmt_xdate()
    eindtijd = max(route_df["time"].max(), starttime + timedelta(minutes=30))
    ax.set_xlim(starttime, eindtijd)
    ax.grid(axis="x", linestyle='--', color='gray', alpha=0.3)
    plt.tight_layout()
    plt.show()

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

    plot_planning_vs_reservations(route_df, reserved_segments, starttime_str="14:00")

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
