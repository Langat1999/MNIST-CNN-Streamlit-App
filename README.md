# MNIST CNN + Streamlit App

![cnn](https://github.com/user-attachments/assets/4e4e54c4-d78a-4920-b449-bdbf739a9267)

This project trains a Convolutional Neural Network (CNN) on the MNIST dataset and serves a simple Streamlit app to classify handwritten digits.

## Features
- Reproducible training with seeds
- Efficient `tf.data` input pipeline with train/val split
- Training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Dual model export: SavedModel directory and H5 file
- Accuracy and loss plots saved to disk
- Streamlit app with robust preprocessing: aspect-ratio padding, inversion to MNIST-style, normalized input, ordered probability display

## Environment Setup (Windows PowerShell)
```powershell
# From the project root
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script execution, allow local scripts:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Training
```powershell
# Default run
python train_mnist_model.py

# With custom hyperparameters
python train_mnist_model.py --epochs 15 --batch_size 256 --learning_rate 0.0007 --val_split 0.1667 --model_dir artifacts/mnist_saved_model --model_h5 mnist_cnn_model.h5
```
Artifacts produced:
- SavedModel: `artifacts/mnist_saved_model/`
- Best H5 (by val_loss during training): `mnist_cnn_model.h5`
- Plots: `training_accuracy.png`, `training_loss.png`

## Running the App
Ensure `mnist_cnn_model.h5` exists in the project root (produce it via training above).
```powershell
streamlit run app.py
```
Open the local URL shown by Streamlit. Upload a 28×28-like grayscale digit image; the app will pad, invert to MNIST-style (white digit on black), normalize, and predict.

### Notes on Inputs
- If your image is black digit on white background, the app will auto-invert
- Non-square images are padded to square before resizing to 28×28
- Prefer centered digits with minimal noise for best results

## Troubleshooting
- Import warnings in editors about TensorFlow/NumPy usually disappear after installing the requirements in the active venv
- If you have a compatible GPU, TensorFlow may use mixed precision automatically; CPU-only installs also work
- If plots don’t display, check the saved PNGs in the project root

## Project Structure
```
.
├─ app.py                       # Streamlit app
├─ train_mnist_model.py         # Training script
├─ requirements.txt             # Pinned dependencies
├─ README.md                    # This file
└─ artifacts/                   # SavedModel directory (created after training)
```

## License
MIT License

Copyright (c) 2025 LangatJM

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

