import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model  
# Predict on validation set
# Assuming val_gen is a validation data generator, initialize it properly
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Example initialization (adjust parameters as needed)
datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_directory(
	'C:\\Users\\shiva\\OneDrive\\Desktop\\DeepFake Detection(2)\\dataset\\val',  # Replace with the actual path to your validation data
	target_size=(299, 299),     # Replace with the target size of your images
	batch_size=32,
	class_mode='binary'
)

val_gen.reset()
# Load the pre-trained model (replace 'model_path.h5' with the actual path to your model file)
model = load_model("C:\\Users\\shiva\\OneDrive\\Desktop\\DeepFake Detection(2)\\models\\xception_deepfake_model.h5")



# Predict on validation set
y_probs = model.predict(val_gen)
y_pred = (y_probs > 0.5).astype(int)
y_true = val_gen.classes

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

plt.savefig("roc_curve.png")
plt.close()

# Save classification report as a CSV file
report = classification_report(y_true, y_pred, target_names=["Real", "Fake"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv", index=True)

# Save confusion matrix as a CSV file
cm_df = pd.DataFrame(cm, index=["Real", "Fake"], columns=["Real", "Fake"])  
cm_df.to_csv("confusion_matrix.csv", index=True)
