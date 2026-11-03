import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.baseline_cnn import BaselineCNN

##########################################################################
# WORKERS: on Windows each worker reloads this file, so all the work must
# sit inside this if, or the workers would try to start the test again
if __name__ == "__main__":
##########################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Same clean images as validation: no random changes
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder("classification_data_square_288/test", transform=test_transform)
    ##########################################################################
    # WORKERS: 8 CPU processes load the images, so the GPU waits less
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    ##########################################################################

    model = BaselineCNN(num_classes=200).to(device)
    model.load_state_dict(torch.load("outputs/square_288_400_wd005/best.pt", map_location=device, weights_only=True))
    model.eval()

    correct = torch.zeros(200)
    total = torch.zeros(200)
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for label, prediction in zip(labels, predictions):
                total[label] += 1
                correct[label] += (label == prediction).item()

    print(f"Test images: {int(total.sum())}")
    print(f"Test accuracy: {100 * correct.sum() / total.sum():.2f}%")
    print(f"Average accuracy per species: {100 * (correct / total).mean():.2f}%")
