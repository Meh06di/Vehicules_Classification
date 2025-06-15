import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
dataset_path = 'dataset/Vehicles'
resized_dataset_path = 'Resize_dataSet'
 
target_size = (224, 224) 
os.makedirs(resized_dataset_path, exist_ok=True)
for cls in os.listdir(dataset_path):
     os.makedirs(os.path.join(resized_dataset_path, cls), exist_ok=True)
 
# Preparation des images
for cls in os.listdir(dataset_path):
     class_path = os.path.join(dataset_path, cls)
     save_class_path = os.path.join(resized_dataset_path, cls)
 
     for img_name in os.listdir(class_path):
         try:
             img_path = os.path.join(class_path, img_name)
             img = Image.open(img_path).convert('RGB')
             img = img.resize(target_size)
             img.save(os.path.join(save_class_path, img_name))
         except Exception as e:
             print(f"Error resizing {img_name} in {cls}: {e}")
 
# # Image parameteres
img_height, img_width = 224, 224
batch_size = 32
num_classes = 8

# %%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    validation_split=0.2
)

# %%
train_generator = train_datagen.flow_from_directory(
    resized_dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# %%
validation_generator = train_datagen.flow_from_directory(
    resized_dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# %%
from tensorflow.keras import models, layers, regularizers

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])


# %%
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_schedule = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    verbose=1,
    min_lr=1e-6
)


# %%
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
model.summary()

# %%
epochs = 35
# Entrainement du modele
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[early_stop, lr_schedule]
)


# %%
model.save('model1.keras')

# %%
# Graphe de training et validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.savefig('training_plots.png')


# %%
# Evalualtion du modele
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# %%
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class_names = ['Bus', 'Cars', 'Truck', 'Motorcycles','Bikes','Planes','Ships','Trains']

img_path = 'test/track2.jpeg'

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

model_load = load_model("model1.keras")

predictions = model_load.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class}")
plt.imshow(img)
plt.title(f"Predicted class: {predicted_class}")
plt.axis('off')
plt.show()


