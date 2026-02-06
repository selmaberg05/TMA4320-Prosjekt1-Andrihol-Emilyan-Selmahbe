# Lager et eget script til å plotte sensordata
# Bruker koden fra run_fdm.py som utgangspunkt og endrer parametre

import matplotlib.pyplot as plt
from project import (
    load_config,
    generate_training_data,
)

def main():
    cfg = load_config("config.yaml")
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg) # Henter inn parametre

    num_sensor = len(cfg.sensor_locations) # Henter ut antall sensorer fra configuration-filen, som i utgangspunktet er lik 9 sensorer

    # Plotter sensordata i en graf
    
    for i in range(num_sensor): # Kan plotte for alle 9 sensorer, men plottet blir mer oversiktlig med for eksempel 4, siden sensor 4-9 har ganske like målinger
        start = i * 25 # Plotter for 24 timer per sensor, altså fra 0-24 og 25-49 timer osv.
        end = (i + 1) * 25

        time = sensor_data[start:end, 2] # Henter ut tid fra vektoren med ulik sensordata
        temp = sensor_data[start:end, 3] # Henter ut temperatur fra vektoren med ulik sensordata

        plt.plot(time, temp, marker='o')

    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title("Sensor data")
    plt.show()
    
    

if __name__ == "__main__":
    main()


