import os
import numpy as np
from PIL import Image

def create_folder_if_not_exists(folder_path):
    """Créer le dossier s'il n'existe pas."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Dossier créé : {folder_path}")

def load_image(image_path):
    """Load an image from a file, resize it to 100x100 pixels, and convert it to RGB format"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((100, 100)) 
    return image

def rgb_to_ycbcr(image):
    """Convertir une image RGB en YCbCr et renvoyer les matrices Y, Cb, et Cr"""
    image = np.asarray(image, dtype=np.float32)
    y = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    cb = 128 - 0.168736 * image[:, :, 0] - 0.331264 * image[:, :, 1] + 0.5 * image[:, :, 2]
    cr = 128 + 0.5 * image[:, :, 0] - 0.418688 * image[:, :, 1] - 0.081312 * image[:, :, 2]
    return y, cb, cr

def save_matrices_to_file(y, cb, cr, file_path):
    """Sauvegarder les matrices Y, Cb et Cr dans un fichier texte"""
    with open(file_path, 'w') as f:
        f.write("Matrice de luminance (Y):\n")
        np.savetxt(f, y, fmt='%.2f')
        f.write("\nMatrice de chrominance (Cb):\n")
        np.savetxt(f, cb, fmt='%.2f')
        f.write("\nMatrice de chrominance (Cr):\n")
        np.savetxt(f, cr, fmt='%.2f')

def process_images_in_folder(folder_images, output_folder):
    """Process all JPG images in the specified folder"""
    # Vérifier si le dossier d'entrée existe
    if not os.path.exists(folder_images):
        print(f"Le dossier spécifié n'existe pas : {folder_images}")
        return
    
    # Créer le dossier de sortie s'il n'existe pas
    create_folder_if_not_exists(output_folder)

    for filename in os.listdir(folder_images):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder_images, filename)
            image = load_image(image_path)
            
            # Conversion en matrices de luminance et chrominance
            y, cb, cr = rgb_to_ycbcr(image)
            
            # Sauvegarde des matrices dans un fichier texte
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_matrices.txt")
            save_matrices_to_file(y, cb, cr, output_file_path)

            print(f"Fichier traité et sauvegardé : {output_file_path}")

if __name__ == "__main__":
    # Créer le dossier DataBase s'il n'existe pas
    base_folder = os.path.join(os.path.dirname(__file__), 'DataBase')
    create_folder_if_not_exists(base_folder)
    
    # Créer les sous-dossiers Image et Matrices s'ils n'existent pas
    folder_images = os.path.join(base_folder, 'Images')
    folder_matrices = os.path.join(base_folder, 'Matrices')
    create_folder_if_not_exists(folder_images)
    create_folder_if_not_exists(folder_matrices)

    # Traiter les images dans le dossier Image
    process_images_in_folder(folder_images, folder_matrices)
    
    print("Traitement terminé : Les matrices ont été sauvegardées avec succès.")
