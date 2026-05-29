from datetime import datetime

class TrafficTwin:
    def __init__(self):
        self.timestamp = datetime.now()
        self.vehicle_count = 0
        self.avg_speed = 0
        self.signal_state = "GREEN"
        self.congestion = "LOW"
        self.ambulance = False

    def update(
        self,
        vehicle_count,
        avg_speed,
        signal_state,
        congestion,
        ambulance
    ):
        self.timestamp = datetime.now()
        self.vehicle_count = vehicle_count
        self.avg_speed = avg_speed
        self.signal_state = signal_state
        self.congestion = congestion
        self.ambulance = ambulance

    def display(self):
        print(
            f"Vehicles={self.vehicle_count}, "
            f"Speed={self.avg_speed:.2f}, "
            f"Signal={self.signal_state}, "
            f"Congestion={self.congestion}, "
            f"Ambulance={self.ambulance}"
        )