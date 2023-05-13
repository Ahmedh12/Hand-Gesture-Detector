# Pattern Recognition

This is the course project for CMPN450 , pattern Recognition and Neural Networks.
In this project, We implement a Hand Gesture Recognition System.Given an image containing a single hand, your system is supposed to classify the hand gesture into one of six digits (from 0 to 5). we implement a complete machine
learning pipeline.

## Directory Structure
```bash
│   .gitignore
│   Project-Document.pdf
│   project.ipynb
│   README.md
│   Tesis.pdf  
│
├───modules
│       data.py
│       display_image.py
│       feature_extraction.py
│       models.py
│       preprocessing.py
│       test_model.py
│
│
├───out
│       results.txt
│       time.txt
│
└───screenshots
        HOG_features.png
        HOG_features_2.png
        preprocessing.png
```
## ScreenShots 

### Original VS Preprocessed Image
<table>
    <tr>
        <td>
            <img src = "screenshots\preprocessing.png" >
        </td>
    </tr>
</table>

### HOG Features in Preprocessed Image
<table>
    <tr>
        <td>
            <img src = "screenshots\HOG_features.png" >
        </td>
        <td>
            <img src = "screenshots\HOG_features_2.png" >
        </td>
    </tr>
</table>


## Results

    we have used two models :
    1- SVM
    2- Fully Connected Neural network

- ### SVM
 SVM with RBF kernel<br>
 Accuracy: 0.6906

| class | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0     | 0.92      | 0.96   | 0.94     | 25      |
| 1     | 0.62      | 0.88   | 0.72     | 24      |
| 2     | 0.66      | 0.54   | 0.59     | 35      |
| 3     | 0.51      | 0.57   | 0.54     | 35      |
| 4     | 0.67      | 0.50   | 0.57     | 28      |
| 5     | 0.84      | 0.79   | 0.82     | 34      |
        

- ### Neural Network
 Two layers deep fully connected Neural network <br>
 Accuracy: 0.6575
        
## Collaborators
- [Ahmed Hussien](https://www.github.com/ahmedh12)
- [Millania Sameh](https://www.github.com/)
- [Maged Magdi](https://www.github.com/)
- [Kareem Mostafa](https://www.github.com/)




