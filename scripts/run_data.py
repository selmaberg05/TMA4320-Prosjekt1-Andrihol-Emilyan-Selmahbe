#Lag et script for å plotte data

import matplotlib.pyplot as plt
from project import (
    load_config,
    generate_training_data,
)

def main():
    cfg = load_config("config.yaml")
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    num_sensor = len(cfg.sensor_locations) # Antall sensorer (9)

    for i in range(9): # Plotter for hver sensor, men blir mest oversiktlig med 4, kan prøve seg frem
        start = i * 25 # Plotter for 24 timer per sensor (feks 0-24 og 25-)
        end = (i + 1) * 25

        time = sensor_data[start:end, 2] 
        temp = sensor_data[start:end, 3]

        plt.plot(time, temp, marker='o')

    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title("Sensor data")
    plt.show()
    
    

if __name__ == "__main__":
    main()


