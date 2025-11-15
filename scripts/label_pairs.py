import os
import json
import numpy as np

traj_folder = "../traj_data"
labels_file = "../traj_data/labels.jsonl"

def load_segments(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])
    segments = []
    for f in files:
        path = os.path.join(folder, f)
        data = np.load(path, allow_pickle=True)
        segments.append(data)
    return segments, files

def compare_segments(segA, segB):
    print("\n--- CLIP A ---")
    print("observations:", segA['obs'])
    print("--- CLIP B ---")
    print("observations:", segB['obs'])
    choice = input("Choose: 1=A better, 2=B better, 0=equal, -1=skip: ")
    return choice

def main():
    segments, files = load_segments(traj_folder)
    labels = []

    for i in range(0, len(segments)-1, 2):
        segA, segB = segments[i], segments[i+1]
        fileA, fileB = files[i], files[i+1]
        choice = compare_segments(segA, segB)
        labels.append({
            "clipA": fileA,
            "clipB": fileB,
            "choice": choice
        })

    with open(labels_file, "w") as f:
        for entry in labels:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved labels to {labels_file}")

if __name__ == "__main__":
    main()

