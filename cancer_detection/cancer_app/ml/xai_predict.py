# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model # type: ignore
# import matplotlib


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_visual_prediction(model_path, image_path, output_path='xai_result.png'):
    model = load_model(model_path, compile=False)
    img_size = model.input_shape[1:3]

    # Load and preprocess image
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_img, img_size)
    input_tensor = np.expand_dims(resized / 255.0, axis=0)

    # Predict
    pred = model.predict(input_tensor)[0][0]
    class_idx = int(pred <= 0.5)
    confidence = pred if class_idx == 0 else 1 - pred
    label = "NON-CANCER" if class_idx == 0 else "CANCER"

    # Grad-CAM setup
    conv_layer = None
    for l in reversed(model.layers):
        if isinstance(l, tf.keras.layers.Conv2D):
            conv_layer = l.name
            break
    if not conv_layer:
        raise ValueError("Conv2D layer not found.")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_tensor)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)[0]
    conv_output = conv_output[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_output.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_output[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]))
    cam = cam / (np.max(cam) + 1e-8)

    # Heatmap
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb_img, 0.6, heatmap_color, 0.4, 0)

    # Add label and confidence - placed in bottom-right corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    label_text = f"{label} ({confidence*100:.2f}%)"
    text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
    x = overlay.shape[1] - text_size[0] - 20
    y = overlay.shape[0] - 20
    cv2.putText(overlay, label_text, (x, y), font, font_scale, (255, 255, 255), thickness)

    # Combine original and heatmap overlay
    combined = np.hstack((rgb_img, overlay))
    result = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result)
    print(f"[✅] Output image saved to: {output_path}")


# --------------------------------------------------------------------------------------------

# matplotlib.use('Agg')  # prevent GUI backend issues
# import os

# def generate_visual_prediction(model_path, image_path, output_path='xai_output.png'):
#     # Load model
#     print(f"🔍 Loading model from1: {model_path}")
#     model = load_model(model_path, compile=False)
#     image_size = model.input_shape[1:3]

#     # Load image
#     img = cv2.imread(image_path)
#     original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     resized_img = cv2.resize(original_img, image_size)
#     input_array = np.expand_dims(resized_img / 255.0, axis=0)

#     # Predict
#     pred = model.predict(input_array)[0][0]
#     label = "NON_CANCER" if pred > 0.5 else "CANCER"
#     confidence = float(pred) if label == "CANCER" else 1 - float(pred)
#     confidence_percent = confidence * 100

#     # 🧠 Find last Conv2D layer
#     last_conv_layer = None
#     for layer in reversed(model.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             last_conv_layer = layer.name
#             break
#     if last_conv_layer is None:
#         raise ValueError("❌ No Conv2D layer found in model.")

#     # Grad-CAM
#     grad_model = tf.keras.models.Model(
#         [model.inputs],
#         [model.get_layer(last_conv_layer).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(input_array)
#         loss = predictions[:, 0]

#     grads = tape.gradient(loss, conv_outputs)
#     if grads is None:
#         raise ValueError("❌ Gradient is None — unable to compute Grad-CAM.")

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
#     heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     heatmap = heatmap.numpy()

#     # Prepare heatmap
#     heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     # Overlay heatmap
#     overlayed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

#     # 📝 Add prediction text with confidence
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     text = f"PREDICTION: {label} ({confidence_percent:.2f}%)"
#     cv2.putText(overlayed_img, text, (10, 30), font, 1, (255, 255, 255), 2)

#     # 🖼 Combine original and heatmap
#     combined = np.hstack((original_img, overlayed_img))
#     combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(output_path, combined_bgr)

#     print(f"✅ Visual output saved: {output_path}")
    
