from matplotlib import pyplot as plt

def display_image(image, title = None):
    if image.ndim == 2:
        plt.gray()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()
