import these both before you run the training 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay# --- Data Preparation for CNN --- (Jonas)
from torch.utils.data import DataLoader, TensorDataset

# --- Data Preparation for CNN --- (Jonas)
def prepare_dataset(dataset, img_size=(200, 200)):
    point_clouds = [pc[:, :-1] for pc in dataset]  # Features
    y = np.array([int(pc[0, -1]) for pc in dataset])  # Single label per cloud (changed code here as error occured later)

    pc_images = []
    for idx, pc in enumerate(point_clouds):
        pc_images.append(convert_pc_to_image(pc, img_size))
        print(f"Converted {idx + 1}/{len(point_clouds)} point clouds to images.")

    X = np.array(pc_images, dtype=np.float32)
    return X, y



def convert_pc_to_image(pc, img_size=(200, 200)):
    """
    Converts a point cloud to a 2D image representation.
    
    Args:
        pc (np.ndarray): Point cloud of shape (N, 3) or (N, >=3).
        img_size (tuple): Size of the output image (height, width).
        
    Returns:
        np.ndarray: 2D image representation of the point cloud.
    """
    if pc.shape[1] < 3:
        raise ValueError("Point cloud must have at least 3 columns (x, y, z).")
    
    # Normalize the point cloud to fit within the image size
    pc_normalized = (pc[:, :2] - np.min(pc[:, :2], axis=0)) / (np.max(pc[:, :2], axis=0) - np.min(pc[:, :2], axis=0))
    pc_normalized *= img_size[0]  # Scale to image size
    
    # Create an empty image
    img = np.zeros(img_size)
    height, width = img_size

    x, y, z, color = pc_normalized[:, 0], pc_normalized[:, 1], pc[:, 2], pc[:, 3:6] if pc.shape[1] >= 6 else np.zeros((pc.shape[0], 3)) 

    # --- Interpolation of Point Cloud Data for Visualization --- (Jonas)

    # sort x,y,z by z in ascending order so the highest z is plotted over the lowest z
    zSort = z.argsort()
    x, y, z, color = x[zSort], y[zSort], z[zSort], color[zSort]

    # interpolation
    # generate a grid where the interpolation will be calculated
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    R = griddata(np.vstack((x, y)).T, color[:, 0], (X, Y), method='cubic')
    Rlinear= griddata(np.vstack((x, y)).T, color[:, 0], (X, Y), method='nearest')
    G = griddata(np.vstack((x, y)).T, color[:, 1], (X, Y), method='cubic')
    Glinear= griddata(np.vstack((x, y)).T, color[:, 1], (X, Y), method='nearest')
    B = griddata(np.vstack((x, y)).T, color[:, 2], (X, Y), method='cubic')
    Blinear= griddata(np.vstack((x, y)).T, color[:, 2], (X, Y), method='nearest')

    #Fill empty values with nearest neighbor
    R[np.isnan(R)] = Rlinear[np.isnan(R)]
    G[np.isnan(G)] = Glinear[np.isnan(G)]
    B[np.isnan(B)] = Blinear[np.isnan(B)]

    # Normalize the color channels to [0, 1]
    R = R - np.min(R)
    G = G - np.min(G)
    B = B - np.min(B)
    # Ensure no negative values
    R[R < 0] = 0
    G[G < 0] = 0
    B[B < 0] = 0

    R = R/np.max(R)
    G = G/np.max(G)
    B = B/np.max(B)

    interpolated = cv2.merge((R, G, B))

    return interpolated

def show_image(img, title="Point Cloud Image"):
    """
    Displays a 2D image representation of a point cloud.
    
    Args:
        img (np.ndarray): 2D image representation of the point cloud.
        title (str): Title of the image.
    """
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

labels = ["2lanes", "3lanes", "crossing", "split4lanes", "split6lanes", "transition"]
folder_path = r"C:/Users/Shino Daniel/Desktop/Microsoft VS Code/dataset" # Adjust the folder path as needed
dataset = load_point_clouds(labels, folder_path=folder_path)  # Load point clouds from the specified folder

dataset = downsample_point_clouds(dataset) # Downsample  before feature extraction
img_size = (200, 200)  # Define the size of the output images

# Prepare the dataset for deep learning
X, y = prepare_dataset(dataset, img_size=img_size)

# Perform a train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Optionally visualize the first point cloud from each category as an image
for i, label in enumerate(labels):
    for pc in dataset:
        # Extract the label from the last column of the point cloud
        pc_label = int(pc[0, -1])
        if pc_label == i:
            visualize_point_cloud(pc[:, :3], title=f"Point Cloud - {label} - {pc.shape[0]} points")
            break



# Build the CNN model
input_shape = (3, img_size[0], img_size[1])  # (C, H, W)


#CNN Model Definition  -- (Jonas)
class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
    
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.6)  # Increased dropout

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            dummy = self.pool(F.relu(self.bn2(self.conv2(dummy))))
            flattened_size = dummy.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(x.size(0), -1)             
        x = F.relu(self.fc1(x))               
        x = self.dropout(x)                   
        x = self.fc2(x)                       
        return x
    
#-- Training and Evaluation of the CNN model -- (Shino Daniel)
    
# Learning curve tracking
def plot_learning_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()            # Clear previous gradients
            outputs = model(inputs)          # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()                  # Backpropagation
            optimizer.step()                 # Update weights

            running_loss += loss.item() * inputs.size(0)

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        test_loss, test_acc = evaluate_model(model, test_loader, device, silent=True)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    print("Training complete.")
    plot_learning_curves(train_losses, test_losses, train_accuracies, test_accuracies)

# Evaluation with confusion matrix
def evaluate_model(model, test_loader, device, class_names=None, silent=False):
    model.eval()
    total, correct = 0, 0
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / total

    if not silent and class_names:
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.show()

    return avg_loss, accuracy if silent else accuracy


# Convert images and labels to torch tensors and reshape for CNN input
# Assume X, y are numpy arrays from prepare_dataset(), X.shape = (N, H, W, 3)
X = np.transpose(X, (0, 3, 1, 2))  # (N, 3, H, W)
X_tensor = torch.tensor(X, dtype=torch.float32)  #changed it to float32 to avoid memory loss issues
y_tensor = torch.tensor(y, dtype=torch.long)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor)

# Wrap in DataLoader
batch_size = 16
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define input shape and number of classes for the CNN
input_shape = (3, X.shape[2], X.shape[3])  # (C, H, W)
num_classes = len(np.unique(y))
model = CNN(input_shape, num_classes)

# Train the model
train_model(model, train_loader, test_loader, num_epochs=15, learning_rate=0.001)
evaluate_model(model, test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"), class_names=labels)
