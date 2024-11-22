import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import random

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

def egaliseur(matrice):
    matrice_flattened = matrice.flatten()  # vecteur 1d

    histogram, bins = np.histogram(matrice_flattened, bins=256, range=[0, 256])  # Calcul de l'histogramme
    cdf = histogram.cumsum() 
    cdf_normalized = cdf * (255 / cdf[-1])  
    matrix_equalized = np.interp(matrice_flattened, bins[:-1], cdf_normalized)
    return matrix_equalized.reshape(matrice.shape)

def apply_mean_filter(matrix):
    # Créer un filtre moyenneur 5x5
    kernel = np.ones((5, 5)) / 25
    # Appliquer le filtre à la matrice
    return convolve(matrix, kernel)

def cutout_data(image, mask_size):
    """Cache une zone aléatoire de l'image avec du noir."""
    h, w = image.shape[:2]
    top = np.random.randint(0, h - mask_size)
    left = np.random.randint(0, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    image[top:bottom, left:right] = 0
    return image

def save_matrices_to_file(y, cb, cr, file_path):
    with open(file_path, 'w') as f:
        f.write("Matrice de luminance (Y):\n")
        np.savetxt(f, y, fmt='%.2f')
        f.write("\nMatrice de chrominance (Cb):\n")
        np.savetxt(f, cb, fmt='%.2f')
        f.write("\nMatrice de chrominance (Cr):\n")
        np.savetxt(f, cr, fmt='%.2f')

def process_images_in_folder(folder_images):
    processed_image_ids = []
    transformed_images = []  # Liste pour stocker les images transformées
    filenames = [f for f in os.listdir(folder_images) if f.lower().endswith('.jpg')]

    # Créer des sous-dossiers pour chaque type de transformation
    mean_filter_folder = os.path.join(folder_images, 'mean_filter')
    equalizer_folder = os.path.join(folder_images, 'equalizer')
    cutout_folder = os.path.join(folder_images, 'cutout')
    os.makedirs(mean_filter_folder, exist_ok=True)
    os.makedirs(equalizer_folder, exist_ok=True)
    os.makedirs(cutout_folder, exist_ok=True)

    for filename in filenames:
        image_id = os.path.splitext(filename)[0]
        image_path = os.path.join(folder_images, filename)
        
        image = load_image(image_path)
        
        # Conversion en matrices de luminance et chrominance
        y, cb, cr = rgb_to_ycbcr(image)
        
        # Appliquer le filtre moyenneur
        y_mean = apply_mean_filter(y)
        cb_mean = apply_mean_filter(cb)
        cr_mean = apply_mean_filter(cr)
        
        # Sauvegarde des matrices dans un fichier texte
        output_folder_path = '/Users/mohamed/Documents/cours/4A/TIP/dataBase/matrices/'
        output_file_path = os.path.join(output_folder_path, f"{image_id}_mean_matrices.txt")
        save_matrices_to_file(y_mean, cb_mean, cr_mean, output_file_path)
        
        # Sauvegarde de l'image transformée
        augmented_image_path = os.path.join(mean_filter_folder, f"{image_id}.jpg")
        augmented_image = Image.fromarray(np.uint8(np.dstack((y_mean, cb_mean, cr_mean))))
        augmented_image.save(augmented_image_path)
        
        # Appliquer l'égaliseur
        y_equal = egaliseur(y)
        cb_equal = egaliseur(cb)
        cr_equal = egaliseur(cr)
        
        # Sauvegarde des matrices dans un fichier texte
        output_file_path = os.path.join(output_folder_path, f"{image_id}_equal_matrices.txt")
        save_matrices_to_file(y_equal, cb_equal, cr_equal, output_file_path)
        
        # Sauvegarde de l'image transformée
        augmented_image_path = os.path.join(equalizer_folder, f"{image_id}.jpg")
        augmented_image = Image.fromarray(np.uint8(np.dstack((y_equal, cb_equal, cr_equal))))
        augmented_image.save(augmented_image_path)
        
        # Appliquer le cutout
        y_cutout = cutout_data(y, mask_size=20)
        cb_cutout = cutout_data(cb, mask_size=20)
        cr_cutout = cutout_data(cr, mask_size=20)
        
        # Sauvegarde des matrices dans un fichier texte
        output_file_path = os.path.join(output_folder_path, f"{image_id}_cutout_matrices.txt")
        #save_matrices_to_file(y_cutout, cb_cutout, cr_cutout, output_file_path)
        
        # Sauvegarde de l'image transformée
        augmented_image_path = os.path.join(cutout_folder, f"{image_id}.jpg")
        augmented_image = Image.fromarray(np.uint8(np.dstack((y_cutout, cb_cutout, cr_cutout))))
        augmented_image.save(augmented_image_path)
        
        transformed_images.append(image_id)  # Ajouter l'image à la liste des images transformées
        
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
        
        # lecture des métadonnées de l'image
        processed_image_ids.append(image_id)

    return processed_image_ids, transformed_images

def read_metadata_from_file(file_path, image_ids):
    # Lire le fichier Excel avec pandas
    df = pd.read_excel(file_path)
    metadata_dict = {}
    for index, row in df.iterrows():
        image_id = row['isic_id']
        if image_id in image_ids:
            metadata_vector = [
                row['isic_id'],
                row['attribution'],
                row['copyright_license'],
                row['age_approx'],
                row['anatom_site_general'],
                row['benign_malignant'],
                row['benign_malignant'],
                row['diagnosis'],
                row['clin_size_long_diam_mm'],
                row['diagnosis'],
                row['diagnosis_1'],
                row['diagnosis_2'],
                row['diagnosis_3'],
                row['diagnosis_4'],
                row['diagnosis_confirm_type'],
                row['lesion_id'],
                row['patient_id'],
            ]
            metadata_dict[image_id] = metadata_vector 
    print('\n')
    print("DICTIONNAIRE DES METADONNEES :")
    print(metadata_dict)
    print('\n')
    # print(metadata_dict['ISIC_9922955']) accès à une métadonnée en particulier

if __name__ == "__main__":
    folder_images = '/Users/mohamed/Documents/cours/4A/TIP/dataBase/image/'
    
    processed_image_ids, transformed_images = process_images_in_folder(folder_images)
    metadata_file = os.path.join('/Users/mohamed/Documents/cours/4A/TIP/DataBase/meta.csv.xlsx')
    read_metadata_from_file(metadata_file, processed_image_ids)
    print("Les matrices ont été sauvegardées dans le dossier matrices avec succès")
    
    # Afficher la liste des images transformées
    print("Images transformées :")
    for image in transformed_images:
        print(image)