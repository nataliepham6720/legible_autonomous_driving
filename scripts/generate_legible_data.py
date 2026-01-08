import os
import pickle
import random
from tqdm import tqdm

from utils.training_utils import generate_legible_scenarios
from utils.vector_utils import *

OUT_PATH = "data/vqa_test_legible.pkl"
BASE_PATH = "data/vqa_train_10k.pkl"
NUM_SAMPLES = 1000 # 20000 for train

os.makedirs("data", exist_ok=True)

# ---- load base dataset once ----
with open(BASE_PATH, "rb") as f:
    base_data = pickle.load(f)

dataset = []

def make_observation_prompt(obs: dict) -> str:
    """
    Convert a VectorObservation dict to a legible human-readable prompt.
    """
    prompt = []

    # Ego vehicle speed
    ego_speed = obs["ego_vehicle_descriptor"][EgoField.SPEED].item() * VELOCITY_MS_SCALE * MS_TO_MPH
    prompt.append(f"The ego vehicle is moving at {ego_speed:.1f} mph.")

    # Traffic lights on route
    tl_state, tl_distance = get_tl_state(obs["route_descriptors"])
    if tl_state is not None:
        prompt.append(f"There is a {tl_state} traffic light {tl_distance:.1f} meters ahead.")

    # Roundabout info
    if determine_roundabout(obs["route_descriptors"]):
        prompt.append("The ego vehicle is approaching a roundabout.")

    # Vehicles
    vehicle_texts = []
    for v in obs["vehicle_descriptors"]:
        if v[VehicleField.ACTIVE]:
            x, y = v[VehicleField.X], v[VehicleField.Y]
            speed = v[VehicleField.SPEED] * VELOCITY_MS_SCALE * MS_TO_MPH
            angle = traveling_angle_deg_from_vehicle_desc(v.reshape(1, -1)).item()
            dynamic = "moving" if v[VehicleField.DYNAMIC] else "static"
            direction = object_direction(angle)
            vehicle_texts.append(
                f"{dynamic} vehicle at ({x:.1f}, {y:.1f}) moving {direction} at {speed:.1f} mph"
            )
    if vehicle_texts:
        prompt.append("Nearby vehicles: " + "; ".join(vehicle_texts))

    # Pedestrians
    pedestrian_texts = []
    for p in obs["pedestrian_descriptors"]:
        if p[PedestrianField.ACTIVE]:
            x, y = p[PedestrianField.X], p[PedestrianField.Y]
            crossing = "crossing" if p[PedestrianField.CROSSING] else "waiting"
            angle = traveling_angle_deg_from_pedestrian_desc(p.reshape(1, -1)).item()
            direction = object_direction(angle)
            pedestrian_texts.append(f"{crossing} pedestrian at ({x:.1f}, {y:.1f}) moving {direction}")
    if pedestrian_texts:
        prompt.append("Nearby pedestrians: " + "; ".join(pedestrian_texts))

    # Final instruction
    prompt.append("Describe the situation and decide the best action for the ego vehicle.")

    return " ".join(prompt)


for i in tqdm(range(NUM_SAMPLES)):
    d = random.choice(base_data)
    base_obs = d["observation"]   # ✅ correct type

    obs = generate_legible_scenarios(
        base_obs=base_obs,
        scenario_type=random.choice([
            "pedestrian_crossing",
            "car_crossing",
            "opposite_direction",
            "same_direction_front",
        ]),
    )

    dataset.append({
        "frame_num": i,
        "input": "",
        "output": make_observation_prompt(obs),
        "route_descriptors": obs["route_descriptors"],
        "vehicle_descriptors": obs["vehicle_descriptors"],
        "pedestrian_descriptors": obs["pedestrian_descriptors"],
        "ego_vehicle_descriptor": obs["ego_vehicle_descriptor"],
    })

with open(OUT_PATH, "wb") as f:
    pickle.dump(dataset, f)

print(f"Saved {len(dataset)} samples to {OUT_PATH}")
