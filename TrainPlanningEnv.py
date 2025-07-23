import pandas as pd
from datetime import datetime, timedelta

class TrainPlanningEnv:
    """
    Omgeving voor het plannen van één treinrit binnen een vaste dienstregeling.

    Attributen:
        nodes (list): Stations in het netwerk.
        edges (dict): Verbindingen tussen stations, met reistijden per spoorsegment {(u, v): duur_in_minuten}.
        start (str): Startstation.
        destination (str): Doelstation.
        reserved_segments (dict): Bestaande reserveringen per segment door andere treinen {(u, v): [(start, end), ...]}.
        headway (int): Minimale buffertijd (in minuten) tussen treinen op hetzelfde segment.
        max_wait (int): Maximale wachttijdoptie in minuten.
        max_time (int): Planningshorizon in minuten (max simulatie tijd).
        max_steps (int): Maximale stappen per episode.
        departure_window (tuple): (earliest, latest) vertrekmoment-venster.
        arrival_deadline (int): Laatste toelaatbare aankomsttijd.
    """
    def __init__(
        self,
        nodes,
        edges,
        start,
        destination,
        reserved_segments=None,
        headway=1,
        max_wait=5,
        max_time=100,
        max_steps=50,
        departure_window=None,
        arrival_deadline=None,
        print_state=False
    ):
        """
        Initialiseer de planningomgeving met netwerk en planningparameters.
        """
        self.nodes = nodes
        self.edges = edges
        extra_edges = {}  # Maak edges bidirectioneel
        for (u, v), tijd in self.edges.items():
            if (v, u) not in self.edges:
                extra_edges[(v, u)] = tijd
        self.edges.update(extra_edges)
        self.start = start
        self.destination = destination
        self.previous_location = None
        self.reserved_segments = reserved_segments or {}
        self.headway = headway
        self.max_wait = max_wait
        self.max_time = max_time
        self.max_steps = max_steps
        self.departure_window = departure_window
        self.arrival_deadline = arrival_deadline
        self.print_state = print_state
        self.is_last_episode = False
        self.reset()

        # Converteer vertrekvenster naar minuten
        if departure_window:
            self.base_departure_minute = self._parse_time(departure_window[0])
        else:
            self.base_departure_minute = 0

    def reset(self):
        """
        Reset de omgeving naar de begintoestand:
          - Agent op startstation
          - Tijd op 0
          - Eigen boekingen wissen
          - Statistieken resetten

        Returns:
            tuple: (huidig_station, huidige_tijd)
        """
        self.current = self.start
        self.time = 0
        self.step_count = 0
        self.conflict_count = 0
        self.wait_time_total = 0
        self.travel_time_total = 0
        self.reward_total = 0
        self.my_bookings = {edge: [] for edge in self.edges}
        self.route = [(self.current, self.time)]
        return (self.current, self.time)
    
    def _parse_time(self, time_str):
        """
        Zet een string zoals '14:00' om naar totaal aantal minuten sinds middernacht.
        """
        h, m = map(int, time_str.split(":"))
        return h * 60 + m
    
    def _print_and_export_if_last(self, new_state):
        if self.print_state and self.is_last_episode:
            abs_time = self.base_departure_minute + self.time
            kloktijd = str(timedelta(minutes=abs_time))[:-3]
            print(f"{kloktijd} | Station: {self.current} | Volgende actie: {new_state[0]}")
            self.export_route_to_csv("planned_route.csv")

    def possible_actions(self, state=None):
        """
        Bepaal mogelijke acties: bewegen of wachten.
        Returns: lijst van (actie, duur)
        """
        curr_staiton, time = state if state is not None else (self.current, self.time)
        actions = []
        # beweeg-acties
        for (from_station, to_station), travel_time in self.edges.items():
            if from_station == curr_staiton and to_station != self.previous_location and to_station != curr_staiton:
                actions.append((to_station, travel_time))

        for wait in range(1, self.max_wait + 1):
            actions.append(('wait', wait))
        return actions

    def _conflict(self, edge, start, end):
        """
        Controleer of er een tijdsconflict is op een spoorsegment, 
        met eigen reserveringen én met externe reserveringen.
        Houd bij externe reserveringen rekening met de headway.
        """
         # Check tegen eigen reserveringen
        for own_start, own_end in self.my_bookings.get(edge, []):
            if not (end <= own_start or start >= own_end):
                return True

        # Check tegen externe reserveringen (inclusief headway)
        for reserved_start, reserved_end in self.reserved_segments.get(edge, []):
            if not (start >= reserved_start + self.headway or end <= reserved_end - self.headway):
                return True

        return False

    def get_state(self):
        """
        Geef de huidige state terug als tuple:
          (huidig station, huidige tijd in minuten).
        """
        return (self.current, self.time)


    def step(self, action):
        """
        Voer een stap uit en return (state, reward, done).
        Hanteer departure_window en arrival_deadline.
        """
        self.step_count += 1
        nxt, dt = action
        old_state = (self.current, self.time)
        self._print_and_export_if_last(action) 

        # TODO: vertrekvenster tijd valideren
        # # 1) vertrekvenster check bij eerste beweging
        # if self.current == self.start and nxt != 'wait' and self.departure_window:
        #     earliest, latest = self.departure_window
        #     if self.time < earliest or self.time > latest:
        #         reward = -25
        #         self.reward_total += reward
        #         return (old_state, reward, True)

        # wacht-actie
        if nxt == 'wait':
            # print("test2")
            self.time += dt
            self.wait_time_total += dt
            reward = -150
            new_state = (self.current, self.time)
            self.route.append(new_state)
            self.reward_total += reward
            done = False
        else:
            # beweeg-actie
            edge = (self.current, nxt)
            start_t, end_t = self.time, self.time + dt
            if edge not in self.edges:
                print(f"Ongeldige edge: {edge}, wordt overgeslagen.")
                reward = 0
                self.time += 1
                new_state = (self.current, self.time)
                self.route.append(new_state)
                self.reward_total += reward
                done = True
                return (new_state, reward, done)

            # conflict?
            if self._conflict(edge, start_t, end_t):
                self.conflict_count += 1
                reward = -1000
                self.time += 1
                self.reward_total += reward
                return (old_state, reward, False)

            # boek segment
            self.my_bookings[edge].append((start_t, end_t))
            self.previous_location = self.current
            self.current = nxt
            self.time = end_t
            self.travel_time_total += dt
            reward = -1
            new_state = (self.current, self.time)            
            self.route.append(new_state)
            self.reward_total += reward
            done = False
            # aankomst?
            if nxt == self.destination:
                # deadline check
                done = True
                if self.arrival_deadline and self.time > self.arrival_deadline:
                    reward = -25
                    self.reward_total += reward
                    return (new_state, reward, done)
                reward = 200
                self.reward_total += reward
                return (new_state, reward, done)
             
        # planningshorizon of max stappen
        if self.time > self.max_time or self.step_count > self.max_steps:
            reward = -1
            self.reward_total += reward
            done = True
            return ((self.current, self.time), reward, done)

        return ((self.current, self.time), reward, done)
    
    def export_route_to_csv(self, filename="planned_route.csv"):
        """
        Exporteer de geplande route naar CSV met kolommen:
        station, time, action, next_station.

        TODO: kleine bug, waarbij laatste route regel er niet bij komt
        """
        base = datetime.strptime("00:00", "%H:%M") + timedelta(minutes=self.base_departure_minute)
        records = []

        for i in range(1, len(self.route)):
            prev_station, prev_time = self.route[i - 1]
            curr_station, curr_time = self.route[i]
            action = "wait" if curr_station == prev_station else "depart"
            abs_time = base + timedelta(minutes=prev_time)
            records.append({
                "station": prev_station,
                "time": abs_time.strftime("%H:%M"),
                "action": action,
                "next_station": curr_station
            })

        df = pd.DataFrame(records)
        df.to_csv(filename, index=False)