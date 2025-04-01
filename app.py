import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os
import zipfile
import shutil
import numpy as np

# **黑色背景**
plt.style.use("dark_background")
sns.set(style="darkgrid")

# **数据集处理**
def process_dataset(dataset_zip):
    extract_path = "dataset"
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(dataset_zip.name, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    categories = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
    if len(categories) < 2:
        return "Error: Dataset must contain at least 2 category folders.", None
    return f"Dataset extracted successfully! Found classes: {categories}", categories

# **训练模型**
def train_model(epochs, batch_size, learning_rate, dataset_zip):
    # 数据集处理
    status, categories = process_dataset(dataset_zip)
    if categories is None:
        return status, None, None, None, None, None

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset_path = "dataset"
    train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_classes = len(train_dataset.classes)
    
    # 确保至少有2个类别
    if num_classes < 2:
        return "Training failed: At least two categories are required.", None, None, None, None, None

    # 模型定义（使用ResNet50）
    model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 修改最后一层输出维度

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    acc_history = []
    all_preds = []
    all_labels = []
    training_log = ""

    # 训练过程
    for epoch in range(int(epochs)):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total * 100
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        training_log += f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n"

        # 每个epoch结束后更新进度
        yield gr.update(value=f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%"), None, None, None, None

    # 保存模型
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)

    # **交互式 Loss 曲线**
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=list(range(1, int(epochs) + 1)), y=loss_history, mode='lines', name='Loss'))
    loss_fig.update_layout(title='Training Loss Curve', xaxis_title='Epoch', yaxis_title='Loss')

    # **交互式 Accuracy 曲线**
    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(x=list(range(1, int(epochs) + 1)), y=acc_history, mode='lines', name='Accuracy'))
    acc_fig.update_layout(title='Training Accuracy Curve', xaxis_title='Epoch', yaxis_title='Accuracy (%)')

    # **混淆矩阵**
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        confusion_matrix[t, p] += 1
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)

    return training_log, loss_fig, acc_fig, confusion_matrix_path, model_path


# **界面**
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🔥 Interactive Model Training Platform")

    dataset_upload = gr.File(label="Upload Dataset (ZIP Format)")
    dataset_output = gr.Textbox(label="Dataset Processing Status")
    category_list = gr.Textbox(label="Detected Categories")
    dataset_btn = gr.Button("Extract Dataset")
    dataset_btn.click(process_dataset, inputs=dataset_upload, outputs=[dataset_output, category_list])

    epochs = gr.Slider(minimum=1, maximum=100, step=1, value=10, label="Epochs")
    batch_size = gr.Slider(minimum=1, maximum=128, step=1, value=32, label="Batch Size")
    learning_rate = gr.Slider(minimum=0.0001, maximum=0.1, step=0.0001, value=0.001, label="Learning Rate")

    train_btn = gr.Button("Start Training")
    output_text = gr.Textbox(label="Training Status")
    loss_plot = gr.Plot(label="Loss Curve")
    acc_plot = gr.Plot(label="Accuracy Curve")
    confusion_matrix_img = gr.Image(label="Confusion Matrix")
    model_download = gr.File(label="Download Trained Model")

    # 训练按钮点击后开始训练并显示进度
    train_btn.click(
        train_model,
        inputs=[epochs, batch_size, learning_rate, dataset_upload],
        outputs=[output_text, loss_plot, acc_plot, confusion_matrix_img, model_download]
    )

demo.launch()
