from rocketpy import Environment, Rocket, SolidMotor, Flight
import datetime
import pandas as pd
import matplotlib.pyplot as plt

latitude = 55.17997
longtitude = 118.76177

env = Environment(latitude=latitude, longitude=longtitude, elevation=0)

tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)

env.set_date(
    (tomorrow.year, tomorrow.month, tomorrow.day, 12)
)

env.set_atmospheric_model(type="Forecast", file="GFS")


Pro75M1670 = SolidMotor(
    thrust_source="data/motors/cesaroni/Cesaroni_M1670.eng",
    dry_mass=1.815,
    dry_inertia=(0.125, 0.125, 0.002),
    nozzle_radius=33 / 1000,
    grain_number=5,
    grain_density=1815,
    grain_outer_radius=33 / 1000,
    grain_initial_inner_radius=15 / 1000,
    grain_initial_height=120 / 1000,
    grain_separation=5 / 1000,
    grains_center_of_mass_position=0.397,
    center_of_dry_mass_position=0.317,
    nozzle_position=0,
    burn_time=3.9,
    throat_radius=11 / 1000,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

# Pro75M1670.info()

calisto = Rocket(
    radius=127 / 2000,
    mass=14.426,
    inertia=(6.321, 6.321, 0.034),
    power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
    power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)

calisto.add_motor(Pro75M1670, position=-1.255)

rail_buttons = calisto.set_rail_buttons(
    upper_button_position=0.0818,
    lower_button_position=-0.6182,
    angular_position=45,
)
nose_cone = calisto.add_nose(
    length=0.55829, kind="von karman", position=1.278
)

fin_set = calisto.add_trapezoidal_fins(
    n=4,
    root_chord=0.120,
    tip_chord=0.060,
    span=0.110,
    position=-1.04956,
    cant_angle=0.5,
    airfoil=("data/airfoils/NACA0012-radians.txt","radians"),
)

tail = calisto.add_tail(
    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
)

main = calisto.add_parachute(
    name="main",
    cd_s=10.0,
    trigger=800,      # ejection altitude in meters
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
    radius=1.5,
    height=1.5,
    porosity=0.0432,
)

drogue = calisto.add_parachute(
    name="drogue",
    cd_s=1.0,
    trigger="apogee",  # ejection at apogee
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
    radius=1.5,
    height=1.5,
    porosity=0.0432,
)

from rocketpy import Accelerometer, Barometer, GnssReceiver, Gyroscope

accel_noisy_nosecone = Accelerometer(
    sampling_rate=100,
    consider_gravity=False,
    orientation=(60, 60, 60),
    measurement_range=70,
    resolution=0.4882,
    noise_density=0.05,
    random_walk_density=0.02,
    constant_bias=1,
    operating_temperature=25,
    temperature_bias=0.02,
    temperature_scale_factor=0.02,
    cross_axis_sensitivity=0.02,
    name="Accelerometer in Nosecone",
)
accel_clean_cdm = Accelerometer(
    sampling_rate=100,
    consider_gravity=False,
    orientation=[
        [0.25, -0.0581, 0.9665],
        [0.433, 0.8995, -0.0581],
        [-0.8661, 0.433, 0.25],
    ],
    name="Accelerometer in CDM",
)
calisto.add_sensor(accel_noisy_nosecone, 1.278)
calisto.add_sensor(accel_clean_cdm, -0.10482544178314143)  # , 127/2000)

gyro_clean = Gyroscope(sampling_rate=100)
gyro_noisy = Gyroscope(
    sampling_rate=100,
    resolution=0.001064225153655079,
    orientation=(-60, -60, -60),
    noise_density=[0, 0.03, 0.05],
    noise_variance=1.01,
    random_walk_density=[0, 0.01, 0.02],
    random_walk_variance=[1, 1, 1.05],
    constant_bias=[0, 0.3, 0.5],
    operating_temperature=25,
    temperature_bias=[0, 0.01, 0.02],
    temperature_scale_factor=[0, 0.01, 0.02],
    cross_axis_sensitivity=0.5,
    acceleration_sensitivity=[0, 0.0008, 0.0017],
    name="Gyroscope",
)
calisto.add_sensor(gyro_clean, -0.10482544178314143)  # +0.5, 127/2000)
calisto.add_sensor(gyro_noisy, (1.278 - 0.4, 127 / 2000 - 127 / 4000, 0))


test_flight = Flight(
    rocket=calisto,
    environment=env,
    rail_length=5.2,
    inclination=85,
    heading=0,
    time_overshoot=False,
    terminate_on_apogee=True,
)
# To export sensor data to a csv file:
# get first column of every row as time from [(time,(ax,ay,az)),...] = a.measured_data
time1, ax, ay, az = zip(*accel_noisy_nosecone.measured_data)
time2, bx, by, bz = zip(*accel_clean_cdm.measured_data)


plt.plot(time1, ax, label="Noisy Accelerometer")
plt.plot(time2, bx, label="Clean Accelerometer")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration ax (m/s^2)")
plt.legend()
plt.grid()
plt.title("Acceleration comparison - ax")
plt.show()

plt.plot(time1, ay, label="Noisy Accelerometer")
plt.plot(time2, by, label="Clean Accelerometer")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration ay (m/s^2)")
plt.legend()
plt.grid()
plt.title("Acceleration comparison - ay")
plt.show()

plt.plot(time1, az, label="Noisy Accelerometer")
plt.plot(time2, bz, label="Clean Accelerometer")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration az (m/s^2)")
plt.legend()
plt.grid()
plt.title("Acceleration comparison - az")
plt.show()
