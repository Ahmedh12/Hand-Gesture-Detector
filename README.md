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

    we have used two models , SVM and Fully Connected Neural network

- ### SVM
        - Accuracy: 0.6906077348066298
        - |_|_|_|
          | | | |

- ### Neural Network
        - Accuracy: 0.6575
## Collaborators
- [Ahmed Hussien](https://www.github.com/ahmedh12)
- [Millania Sameh](https://www.github.com/)
- [Maged Magdi](https://www.github.com/)
- [Kareem Mostafa](https://www.github.com/)




