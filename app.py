import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mappings
label_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
cancerous_classes = {'akiec', 'bcc', 'mel'}

# Define validation/test transforms (must match your training notebook)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the trained model
def load_model():
    model = models.resnet50(weights=None)  # same architecture as trained
    model.fc = nn.Linear(model.fc.in_features, len(label_names))
    model.load_state_dict(torch.load("skin_cancer_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Prediction function
def predict_skin_lesion(image):
    img_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_label = label_names[pred_idx]
        is_cancerous = pred_label in cancerous_classes

        prob_scores = {label_names[i]: f"{probs[i].item() * 100:.2f}%" for i in range(len(label_names))}

    result = (
        f"Predicted Type: {pred_label}\n"
        f"Cancerous: {'Yes' if is_cancerous else 'No'}\n"
        f"Probability Scores:\n" +
        "\n".join([f"{k}: {v}" for k, v in prob_scores.items()])
    )
    return result

# Gradio interface
demo = gr.Interface(
    fn=predict_skin_lesion,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Skin Cancer Detection",
    description="Upload a skin lesion image to predict the type and whether it's cancerous, with probability scores."
)

if __name__ == "__main__":
    demo.launch()
