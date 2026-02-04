import matplotlib.pyplot as plt
from project import (
    load_config, 
    generate_training_data,

)

def main():
    cfg=load_config("config.yaml")
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    num_sensor =len(cfg.sensor_locations) #antall sensorer (9)

    for i in range(4): #plotter for hver sensor, men blir mest oversiktelig med 4
        start=i*25 #plotter for 24 timer per sensor
        end=(i+1)*25

        time=sensor_data[start:end, 2]
        temp=sensor_data[start:end, 3]

        plt.plot(time, temp, marker="o")

    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title("Sensor data")
    plt.show()


if __name__ == "__main__":
    main()