import numpy as np
import os

def train():

    os.makedirs("trained_weights", exist_ok=True)

    dummy_weights = np.array([1,2,3])

    np.save("trained_weights/model_weights.npy", dummy_weights)

    print("Training completed. Weights saved.")

if __name__ == "__main__":
    train()
