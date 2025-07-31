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
        # De basisvertrektijd (minuut vanaf middernacht) wordt gebruikt om alle interne
        # tijden relatief weer te geven. Bijvoorbeeld: bij een vertrekvenster ("14:00", "14:30")
        # staat self.time = 0 gelijk aan 14:00 en is de toegestane vertrektijd maximaal 30 minuten
        # later.  We slaan het venster daarom zowel in absolute minuten als in relatieve minuten op.
        if departure_window:
            # absolute begin (minuten sinds middernacht)
            start_abs = self._parse_time(departure_window[0])
            end_abs = self._parse_time(departure_window[1])
            self.base_departure_minute = start_abs
            # relatieve toegestane venstergrenzen (0 = vroegste vertrek, (end_abs - start_abs) = laatste vertrek)
            self.departure_window_minutes = (0, max(0, end_abs - start_abs))
        else:
            self.base_departure_minute = 0
            self.departure_window_minutes = None

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

    def _print_and_export_if_last(self, action):
        """
        Druk tijdens de laatste episode de huidige tijd, het station en de
        geselecteerde actie af. Deze functie exporteert **niet** langer de
        route; het exporteren van de route wordt nu na het updaten van
        self.route gedaan om te voorkomen dat de laatste regel ontbreekt.

        Args:
            action (tuple): De gekozen actie (station of 'wait', duur)
        """
        if self.print_state and self.is_last_episode:
            abs_time = self.base_departure_minute + self.time
            kloktijd = str(timedelta(minutes=abs_time))[:-3]
            next_action = action[0]
            print(f"{kloktijd} | Station: {self.current} | Volgende actie: {next_action}")

    def possible_actions(self, state=None):
        """
        Bepaal mogelijke acties: eerst proberen we een conflictvrije verplaatsing te vinden.
        Als dat niet kan, bepaal de minimale wachttijd die nodig is om een segment vrij te krijgen
        en geef alleen die wachtactie terug.

        Returns:
            list[tuple]: lijst van (next_station | 'wait', duur) acties
        """
        curr_station, current_time = state if state is not None else (self.current, self.time)

        # Potentiële verplaatsingen vanaf het huidige station bepalen
        move_candidates = []
        for (from_station, to_station), travel_time in self.edges.items():
            if from_station == curr_station and to_station != self.previous_location and to_station != curr_station:
                move_candidates.append((to_station, travel_time))

        # Controleer welke verplaatsingen conflictvrij zijn op basis van reserveringen en headway
        valid_moves = []
        for to_station, dt in move_candidates:
            start_t, end_t = current_time, current_time + dt
            if not self._conflict((curr_station, to_station), start_t, end_t):
                valid_moves.append((to_station, dt))

        # Indien er een conflictvrije verplaatsing bestaat, staan we geen wachtopties toe
        if valid_moves:
            return valid_moves

        # Als alle verplaatsingen conflicteren, bereken de minimale wachttijd die nodig is om een van
        # de segmenten vrij te laten worden. Dit voorkomt onnodig lange wachttijden.
        min_wait = None
        for to_station, dt in move_candidates:
            # Zoek het vroegste moment waarop deze edge vrij is na de huidige tijd
            earliest_available = None
            for reserved_start, reserved_end in self.reserved_segments.get((curr_station, to_station), []):
                # Een volgende trein mag pas headway minuten na het vertrek van de reservering vertrekken
                available_time = reserved_start + self.headway
                if available_time > current_time:
                    if earliest_available is None or available_time < earliest_available:
                        earliest_available = available_time
            # Bereken benodigde wachttijd voor deze edge
            if earliest_available is not None:
                wait_needed = earliest_available - current_time
                if wait_needed <= 0:
                    wait_needed = 1
                if min_wait is None or wait_needed < min_wait:
                    min_wait = wait_needed

        if min_wait is None or min_wait <= 0:
            min_wait = 1
        # Zorg dat de wachttijd binnen de toegestane grenzen valt
        min_wait = max(1, min(int(min_wait), self.max_wait))
        return [('wait', min_wait)]

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
            # We hanteren de headway rond het startmoment van de reservering. Een volgende trein
            # mag pas vertrekken wanneer 'start' ten minste headway minuten na de reservering is.
            # Dit voorkomt dat we moeten wachten tot de eerste trein helemaal door het segment is.
            # We gebruiken een symmetrisch venster rondom reserved_start (van reserved_start - headway
            # tot reserved_start + headway) om ook treinen uit tegengestelde richting af te vangen.
            if not (start >= reserved_start + self.headway or end <= reserved_start - self.headway):
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
        # Print tijd, station en gekozen actie tijdens de laatste episode voor visuele feedback
        self._print_and_export_if_last(action)
        nxt, dt = action
        old_state = (self.current, self.time)

        # ===== Validatie vertrekvenster =====
        # Controleer alleen bij de allereerste verplaatsing vanaf het startstation (dus niet bij wachten)
        if self.current == self.start and nxt != 'wait' and self.departure_window_minutes:
            earliest_rel, latest_rel = self.departure_window_minutes
            # self.time is in relatieve minuten sinds base_departure_minute
            if self.time < earliest_rel or self.time > latest_rel:
                # Te vroeg of te laat vertrokken: direct beëindigen met straf
                reward = -25
                self.reward_total += reward
                # voeg de huidige state nogmaals toe aan de route zodat export compleet is
                self.route.append((self.current, self.time))
                # Exporteer route als dit de laatste episode is
                if self.print_state and self.is_last_episode:
                    self.export_route_to_csv("planned_route.csv")
                return (old_state, reward, True)

        # ===== Afhandelen wachtactie =====
        if nxt == 'wait':
            # wachten verhoogt de tijd maar verplaatst niet
            self.time += dt
            self.wait_time_total += dt
            reward = -15
            new_state = (self.current, self.time)
            self.route.append(new_state)
            self.reward_total += reward
            done = False
        else:
            # ===== Afhandelen verplaatsing =====
            edge = (self.current, nxt)
            start_t, end_t = self.time, self.time + dt
            if edge not in self.edges:
                # Ongeldige edge: penaliseer en beëindig episode
                print(f"Ongeldige edge: {edge}, wordt overgeslagen.")
                reward = 0
                self.time += 1
                new_state = (self.current, self.time)
                self.route.append(new_state)
                self.reward_total += reward
                done = True
                # Exporteer route indien laatste episode
                if self.print_state and self.is_last_episode:
                    self.export_route_to_csv("planned_route.csv")
                return (new_state, reward, done)

            # Check op conflicts met eigen of externe reserveringen
            if self._conflict(edge, start_t, end_t):
                # Conflict: grote straf en tijd verhogen met 1 minuut (zoals in originele code)
                self.conflict_count += 1
                reward = -1000
                self.time += 1
                self.reward_total += reward
                # We blijven op dezelfde locatie, dus old_state blijft geldig
                new_state = (self.current, self.time)
                self.route.append(new_state)
                done = False
                # Exporteer indien laatste episode zodat route klopt
                if self.print_state and self.is_last_episode:
                    self.export_route_to_csv("planned_route.csv")
                return (old_state, reward, done)

            # Geen conflict: segment boeken en verplaatsen
            self.my_bookings[edge].append((start_t, end_t))
            self.previous_location = self.current
            self.current = nxt
            self.time = end_t
            self.travel_time_total += dt
            # Basisse beloning voor verplaatsing is licht negatief om wachttijd te ontmoedigen
            reward = -1
            new_state = (self.current, self.time)
            self.route.append(new_state)
            self.reward_total += reward
            done = False
            # Controleer of we op bestemming zijn
            if nxt == self.destination:
                done = True
                # Controleer deadlines: arrival_deadline wordt opgegeven in minuten sinds base vertrek
                if self.arrival_deadline is not None and self.time > self.arrival_deadline:
                    reward = -25
                    self.reward_total += reward
                    # Exporteer route indien laatste episode
                    if self.print_state and self.is_last_episode:
                        self.export_route_to_csv("planned_route.csv")
                    return (new_state, reward, done)
                # Succesvolle aankomst belonen positief
                reward = 200
                self.reward_total += reward
                # Exporteer route indien laatste episode
                if self.print_state and self.is_last_episode:
                    self.export_route_to_csv("planned_route.csv")
                return (new_state, reward, done)

        # ===== Planninghorizon of maximum stappen =====
        if self.time > self.max_time or self.step_count > self.max_steps:
            reward = -1
            self.reward_total += reward
            done = True
            # Exporteer route indien laatste episode
            if self.print_state and self.is_last_episode:
                self.export_route_to_csv("planned_route.csv")
            return ((self.current, self.time), reward, done)

        # ===== Export en print voor monitoring =====
        # Alleen in de laatste episode exporteren we na elke stap zodat geplande route volledig is
        if self.print_state and self.is_last_episode:
            self.export_route_to_csv("planned_route.csv")
        return ((self.current, self.time), reward, done)
    
    def export_route_to_csv(self, filename="planned_route.csv"):
        """
        Exporteer de geplande route naar CSV met kolommen:

            - **station**: het station waar de actie plaatsvindt (vertrek/wachten)
            - **time**: de absolute tijd (HH:MM) van het vertrek of wachtmoment
            - **action**: "depart" voor een verplaatsing naar het volgende station,
              of "wait" wanneer de trein op hetzelfde station blijft wachten
            - **next_station**: het doelstation van de actie (ook bij wachten gelijk aan station)

        Het CSV-bestand bevat een regel per stap in de route. Er is geen aparte
        regel voor de laatste eindtoestand omdat er geen actie meer wordt uitgevoerd.
        De route wordt in de laatste episode na elke stap geüpdatet zodat de
        export altijd de volledige route bevat.
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