from scipy.ndimage import gaussian_filter
import numpy as np
from PIL import Image
import os

def search_image(image_folder, image_name):
    folder_files = [i for i in os.listdir(image_folder) if i == image_name and os.path.isfile(os.path.join(image_folder, i))]
    if folder_files:
        image_path = os.path.join(image_folder, folder_files[0])
        return image_path

def image_dimension(image):
    with Image.open(image) as img:
        width, height = img.size
    return width, height

def gaussianBlur(neighborhood, sigma=5):
    blurred_neighborhood = np.zeros_like(neighborhood, dtype=np.float32)
    for channel in range(neighborhood.shape[2]):  # Loop over each color channel
        blurred_neighborhood[..., channel] = gaussian_filter(neighborhood[..., channel], sigma=sigma, mode="reflect")
    return blurred_neighborhood

def fourier_transform(neighborhood, sigma=None):
    return np.abs(np.fft.fft2(neighborhood))

def calculate_sigma(percentage_blur):
    return percentage_blur / 100 * 5

def iterate_pixels(image_path, operation, percentage_blur, output_folder):
    with Image.open(image_path) as img:
        img_array = np.array(img)
        output_img_array = np.zeros_like(img_array, dtype=np.float32)
        sigma = calculate_sigma(percentage_blur)
        pad_width = int(sigma) // 2  # Padding width for the neighborhood
        for y in range(pad_width, img_array.shape[0] - pad_width):
            for x in range(pad_width, img_array.shape[1] - pad_width):
                neighborhood = img_array[y - pad_width:y + pad_width + 1, x - pad_width:x + pad_width + 1]
                output_img_array[y, x] = operation(neighborhood, sigma)
        output_img = Image.fromarray(np.uint8(output_img_array))
        output_img.save(os.path.join(output_folder, os.path.basename(image_path)))  # Save the result image

def main():
    image_folder = "C:/Users/Mateo/Desktop/test/blur/images"
    output_folder = "C:/Users/Mateo/Desktop/test/blur/blur"  # Output folder path
    image_name = input("Nombre de la imagen a transformar: ")
    operation = input("----------------\nSelect an operation:\n1) FFT compression\n2) Gaussian blur\n----------------\nYour choice: ")

    if image_name.endswith(('.jpg')):
        image = search_image(image_folder, image_name)
        width, height = image_dimension(image)
        print(f"----------------\nImagen: {image_name}\nWidth: {width}\nHeight: {height}")
    else: 
        print(f"No se encontro la imagen o falta agregar .jpg")
        exit()

    # Operaciones definidas sobre la imágen
    if operation == "1":
        compress_percentage = float(input("Porcentaje de compresion (0-100%): "))
        iterate_pixels(image, fourier_transform, compress_percentage, output_folder)
        
    if operation == "2":
        blur_percentage = float(input("Porcentaje de desenfoque (0-100%): "))
        iterate_pixels(image, gaussianBlur, blur_percentage, output_folder)

if __name__ == "__main__":
    main()
