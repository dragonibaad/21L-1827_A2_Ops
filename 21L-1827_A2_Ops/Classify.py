import sys                                               
import tensorflow as tf
import mlflow.keras
import dagshub

dagshub.init(repo_owner="dragonibaad", repo_name="VisionGuard", mlflow=True)
IMG_SIZE = 224

model = mlflow.keras.load_model("models:/VisionGuard-classifier@champion")

image_path = sys.argv[1]                                 
img = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])         
batch = tf.expand_dims(img, 0)                           

prob = float(model.predict(batch, verbose=0)[0][0])      
label = "AI-Generated" if prob >= 0.5 else "Real"
print(f"{image_path}: {label} (probability AI-Generated = {prob:.3f})")