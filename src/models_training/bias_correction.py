from sklearn.linear_model import LinearRegression
import pickle
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import config

all_samples = config.CAM_BIAS

def train_bias_regression(samples, cam_id, save):
    # Convert to training format
    X = []
    y_dx = []
    y_dy = []

    for sample in samples:
        x, y = sample["det_center"]
        true_x, true_y = sample["true_footprint"]
        dx = true_x - x
        dy = true_y - y
        
        X.append([x, y])  # features
        y_dx.append(dx)   # label for x correction
        y_dy.append(dy)   # label for y correction

    model_dx = LinearRegression().fit(X, y_dx)
    model_dy = LinearRegression().fit(X, y_dy)

    if save:
        save_model(cam_id, model_dx, model_dy)

def correct_center(x, y, model_dx, model_dy):
    dx = model_dx.predict([[x, y]])[0]
    dy = model_dy.predict([[x, y]])[0]
    corrected_x = x + dx
    corrected_y = y + dy
    return corrected_x, corrected_y

def save_model(cam_id, model_dx, model_dy):
    path = f"{config.MODEL_DIR}/cam_{cam_id}_bias_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"dx": model_dx, "dy": model_dy}, f)

def load_model(cam_id):
    path = f"{config.MODEL_DIR}/cam_{cam_id}_bias_model.pkl"
    with open(path, "rb") as f:
        models = pickle.load(f)
        model_dx = models["dx"]
        model_dy = models["dy"]
    return model_dx, model_dy


if __name__ == "__main__":
    # for cam_id, samples in all_samples.items():
    #     train_bias_regression(samples, cam_id, True)
    model_dx, model_dy = load_model(8)
    print(correct_center(451, 195, model_dx, model_dy))