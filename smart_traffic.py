from datetime import datetime

def smart_green_time(vehicle_count, avg_speed, ambulance_detected):

    current_hour = datetime.now().hour

    if ambulance_detected:
        return 120, "Emergency"

    base_time = 20
    vehicle_factor = vehicle_count * 2

    speed_factor = 0
    if avg_speed < 20:
        speed_factor = 20

    rush_factor = 0
    if 8 <= current_hour <= 10 or 17 <= current_hour <= 20:
        rush_factor = 15

    green_time = base_time + vehicle_factor + speed_factor + rush_factor
    green_time = min(green_time, 150)

    if vehicle_count < 10:
        congestion = "Low"
    elif vehicle_count < 25:
        congestion = "Medium"
    else:
        congestion = "High"

    return int(green_time), congestion
