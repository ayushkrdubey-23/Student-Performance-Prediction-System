import os

def run_command(command):
    print(f"\n🔹 Running: {command}")
    os.system(command)

if __name__ == "__main__":
    print(" Student Performance Prediction System Started")

    print("\n Step 1: Training Model")
    run_command("python src/train_xgboost.py")

    print("\n Step 2: Running Batch Prediction + Visualization")
    run_command("python src/predict.py")

    print("\n All steps completed successfully!")
    print(" Check outputs/ and images/ folders")
    