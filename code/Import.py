import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from a file, resize it to 100x100 pixels, and convert it to RGB format"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((100, 100)) 
    return image

def rgb_to_ycbcr(image):
    image = np.asarray(image, dtype=np.float32)
    # Transformation RGB vers YCbCr
    y = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    cb = 128 - 0.168736 * image[:, :, 0] - 0.331264 * image[:, :, 1] + 0.5 * image[:, :, 2]
    cr = 128 + 0.5 * image[:, :, 0] - 0.418688 * image[:, :, 1] - 0.081312 * image[:, :, 2]

    return y, cb, cr

def save_matrices_to_file(y, cb, cr, file_path):
    with open(file_path, 'w') as f:
        f.write("Matrice de luminance (Y):\n")
        np.savetxt(f, y, fmt='%.2f')
        f.write("\nMatrice de chrominance (Cb):\n")
        np.savetxt(f, cb, fmt='%.2f')
        f.write("\nMatrice de chrominance (Cr):\n")
        np.savetxt(f, cr, fmt='%.2f')

def process_images_in_folder(folder_images):
    """Process all JPG images in the specified folder"""
    for filename in os.listdir(folder_images):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder_images, filename)
            image = load_image(image_path)
            
            # Conversion en matrices de luminance et chrominance
            y, cb, cr = rgb_to_ycbcr(image)
            # Sauvegarde des matrices dans un fichier texte
            output_folder_path = './DataBase/matrices/'
            output_file_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_matrices.txt")
            save_matrices_to_file(y, cb, cr, output_file_path)
            
            
            """
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(y, cmap='gray')
            plt.title('Luminance (Y)')
            plt.subplot(1, 3, 2)
            plt.imshow(cb, cmap='viridis')
            plt.title('Chrominance (Cb)')
            plt.subplot(1, 3, 3)
            plt.imshow(cr, cmap='plasma')
            plt.title('Chrominance (Cr)')
            plt.show()
            """

if __name__ == "__main__":
    folder_images = './DataBase/image/'
    process_images_in_folder(folder_images)   
    print("Les matrices ont été sauvegardées dans le dossier matrices avec succès") #affichage d'un message de confirmation


#fais moi une fonction qui additionne deux matrice 
def add_matrices(y1, y2, cb1, cb2, cr1, cr2):
    y = y1 + y2
    cb = cb1 + cb2
    cr = cr1 + cr2
    return y, cb, cr



"""
import os
import numpy as np
from PIL import Image


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((100, 100)) 
    return image

def rgb_to_ycbcr(image):
    image = np.asarray(image, dtype=np.float32)
    # Transformation RGB vers YCbCr
    y = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    cb = 128 - 0.168736 * image[:, :, 0] - 0.331264 * image[:, :, 1] + 0.5 * image[:, :, 2]
    cr = 128 + 0.5 * image[:, :, 0] - 0.418688 * image[:, :, 1] - 0.081312 * image[:, :, 2]

    return y, cb, cr



def save_matrices_to_file(y, cb, cr, file_path):
    with open(file_path, 'w') as f:
        f.write("Matrice de luminance (Y):\n")
        np.savetxt(f, y, fmt='%.2f')
        f.write("\nMatrice de chrominance (Cb):\n")
        np.savetxt(f, cb, fmt='%.2f')
        f.write("\nMatrice de chrominance (Cr):\n")
        np.savetxt(f, cr, fmt='%.2f')

def process_images_in_folder(folder_images):
    for filename in os.listdir(folder_images):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder_images, filename)
            image = load_image(image_path)
            
            # Conversion en matrices de luminance et chrominance
            y, cb, cr = rgb_to_ycbcr(image)
            # Sauvegarde des matrices dans un fichier texte
            output_folder_path = '/Users/mohamed/Documents/cours/4A/TIP/DataBase/matrices/'
            output_file_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_matrices.txt")
            save_matrices_to_file(y, cb, cr, output_file_path)
           


if __name__ == "__main__":
    folder_images = '/Users/mohamed/Documents/cours/4A/TIP/DataBase/images/'
    process_images_in_folder(folder_images)   
    print("les matrices ont été sauvegardées dans le dossier matrices avec succès")

"""