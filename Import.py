import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

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
    processed_image_ids = []
    """Process all JPG images in the specified folder"""
    for filename in os.listdir(folder_images):
        if filename.lower().endswith('.jpg'):
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(folder_images, filename)
            image = load_image(image_path)
            
            # Conversion en matrices de luminance et chrominance
            y, cb, cr = rgb_to_ycbcr(image)
            # Sauvegarde des matrices dans un fichier texte
            output_folder_path = '/Users/mohamed/Documents/cours/4A/TIP/dataBase/matrices/'
            output_file_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_matrices.txt")
            save_matrices_to_file(y, cb, cr, output_file_path)
            
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

    return processed_image_ids

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
    processed_image_ids = process_images_in_folder(folder_images)
    metadata_file = os.path.join('/Users/mohamed/Documents/cours/4A/TIP/DataBase/meta.csv.xlsx')
    read_metadata_from_file(metadata_file, processed_image_ids)
    print("Les matrices ont été sauvegardées dans le dossier matrices avec succès")