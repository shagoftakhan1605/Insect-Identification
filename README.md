# Insect Image Classification with a CNN and Tkinter GUI

A convolutional neural network (CNN) that classifies insect images into five classes (Butterfly, Dragonfly, Grasshopper, Ladybird, Mosquito), with a Tkinter GUI for webcam or uploaded-image prediction.

* `Insect Recognition Model Building.ipynb`: trains the CNN and saves `insect_classifier.h5`
* `GUI Detection.ipynb`: loads the model and runs the GUI
* `Dataset_and_Pre_Trained_Model.md`: Google Drive links to the image folders and the trained model

---

## Results

Example predictions from the GUI:

![Screenshot (639)](https://github.com/user-attachments/assets/4bfb99d5-6db2-4c72-8607-049e0929ccaf)
![Screenshot (638)](https://github.com/user-attachments/assets/6fb0cdf9-48ae-45e8-b18b-1d6e46876c94)
![Screenshot (637)](https://github.com/user-attachments/assets/eaab3d35-124c-49de-8ffa-64a18146e916)
![Screenshot (640)](https://github.com/user-attachments/assets/7f2fa48b-6475-4084-8836-81ea9d931976)
![Screenshot (641)](https://github.com/user-attachments/assets/53eb96fc-a78c-4271-89eb-276064b8bed2)

From the saved training run (10 epochs):

| | Accuracy |
| --- | --- |
| Final training accuracy | 0.78 |
| Final validation accuracy | 0.67 |

Validation accuracy varied between 0.33 and 0.71 across epochs. Part of this was caused by `steps_per_epoch` settings that made some epochs use only a few batches; this has now been fixed (see below).

---

## Dataset

* 4,449 images in five class folders: 3,561 for training and 888 for validation (80/20 split with `validation_split`)
* Images resized to 150 x 150 RGB and scaled to [0, 1]
* Download link: see `Dataset_and_Pre_Trained_Model.md`
* Original source: *to be added* (please credit the original image collection and check its licence)

Expected folder layout:
```
data/
  ├── Butterfly/
  ├── Dragonfly/
  ├── Grasshopper/
  ├── Ladybird/
  └── Mosquito/
```

---

## Model

* Four convolution blocks (32, 64, 128, 128 filters, 3 x 3, ReLU), each with 2 x 2 max pooling
* Flatten, Dense (512, ReLU), Dense (5, softmax)
* Adam optimiser, categorical cross-entropy, batch size 32, 10 epochs
* No data augmentation or dropout

---

## How to Run

```bash
pip install -r requirements.txt
```

1. Download the `data/` folder and `insect_classifier.h5` from the links in `Dataset_and_Pre_Trained_Model.md` and place them in the project folder.
2. To retrain, run `Insect Recognition Model Building.ipynb` in Jupyter.
3. To use the GUI, run `GUI Detection.ipynb`. Choose **Real-Time Detection** (press `q` to stop) or **Upload Image** (you can keep uploading images; cancel the file dialog to stop).

---

## Fixes in This Version

* **Colour order:** OpenCV reads images as BGR but the model was trained on RGB. The GUI now converts to RGB before predicting.
* **Training steps:** removed `steps_per_epoch` and `validation_steps`, which caused alternate epochs to run on only part of the data.
* README corrected: no data augmentation is used, and notebooks are run in Jupyter rather than with `python`.

---

## Limitations

* Small CNN trained from scratch for 10 epochs; transfer learning from a pre-trained network would likely do better
* No separate test set and no per-class report
* Whole-image classification, so webcam results depend on how much of the frame the insect fills

---

## Tools

Python, TensorFlow/Keras, OpenCV, Tkinter, NumPy, Matplotlib
