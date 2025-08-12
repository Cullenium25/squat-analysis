import yaml
import torch
from ultralytics import YOLO

def train_custom_pose():
# Load a pretrained YOLO11 pose model
    model = YOLO('yolo11n-pose.pt')

    # Load your custom dataset configuration
    with open('./configs/data.yaml', 'r') as file:
        data_config = yaml.safe_load(file)

    print("Starting optimized training with 19 keypoints...")

    # Determine the device(s) to use for training
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            training_device = list(range(num_gpus)) # Use all available GPUs
            print(f"CUDA GPUs are available ({num_gpus} detected). Training will use devices: {training_device}.")
        else:
            training_device = 'cuda'
            print("CUDA GPU is available (1 detected). Training will use the GPU.")
    else:
        training_device = 'cpu'
        print("CUDA GPU is not available. Training will use the CPU.")

    # Train the model
    results = model.train(
        data= './configs/data.yaml',  # path to dataset YAML
        epochs=100,                  # number of training epochs
        imgsz=640,                   # training image size
        batch=16,                    # batch size (adjust based on your GPU memory)
        device=training_device,      # use 'cuda' if available, otherwise 'cpu'
        save=True,                   # save training checkpoints
        project='runs/pose',         # project name
        name='pose19_experiment',    # experiment name
        verbose=True                 # verbose output
    )

    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_custom_pose()