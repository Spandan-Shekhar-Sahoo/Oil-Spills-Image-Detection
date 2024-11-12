import torch
from pathlib import Path
from PIL import Image

def test():
    # Load the model
    model = torch.hub.load('', 'custom', path='runs/train/exp/weights/best.pt', source='local')
    model.eval()  # Set the model to evaluation mode
    
    # Specify the test images folder and output folder
    test_images_folder = Path(r'S:/IOE_Project/dataset/images/test')
    output_folder = Path(r'S:/IOE_Project/results')
    output_folder.mkdir(parents=True, exist_ok=True)  # Create the output folder if it doesn't exist

    # Loop over each image in the test directory
    for filename in test_images_folder.iterdir():
        if filename.suffix in ['.jpg', '.png', '.jpeg']:  # Only process image files
            # Perform inference on each image
            results = model(filename)  # This runs inference on the image
            
            # Render and save the annotated image manually
            results.render()  # Renders results on the image
            annotated_img = Image.fromarray(results.ims[0])  # Convert the first result image array to PIL format
            result_image_path = output_folder / f"{filename.stem}_prediction{filename.suffix}"
            annotated_img.save(result_image_path)  # Save the image directly in results folder

            print(f'Processed {filename.name}, annotated results saved to {result_image_path}')

test()


