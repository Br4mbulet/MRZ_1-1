from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def calculate_ci(pixel_color):
    return (2 * pixel_color) / 255 - 1


def reverse_calculate_c(pixel_color):
    return int(255 * (pixel_color + 1) // 2)


def get_initial_weights(input_size, output_size):
    return np.random.uniform(-1, 1, size=(input_size, output_size))


def get_data(pixels):
    all_pixels = np.zeros([image_height, image_width, 3], float)
    for row_number in range(image_height):
        for column_number in range(image_width):
            cpixel = list(pixels[column_number, row_number])
            for i in range(len(cpixel)):
                all_pixels[row_number][column_number][i] = calculate_ci(cpixel[i])
    return all_pixels


def split_into_rectangles(all_pixels, rectangle_height, rectangle_width):
    rectangles = []
    for row_number_pixel in range(image_height // rectangle_height):
        for column_number_pixel in range(image_width // rectangle_width):
            rectangle = fill_rectangle_with_pixel(all_pixels, row_number_pixel, column_number_pixel,
                                                  rectangle_height, rectangle_width)
            rectangles.append(rectangle)
    return np.array(rectangles)


def fill_rectangle_with_pixel(all_pixels, row_number_pixel, column_number_pixel, rectangle_height, rectangle_width):
    rectangle = []
    for row_number_rectangle in range(rectangle_height):
        for column_number_rectangle in range(rectangle_width):
            for color in range(3):
                rectangle.append(all_pixels[row_number_pixel * rectangle_height + row_number_rectangle,
                                            column_number_pixel * rectangle_width + column_number_rectangle, color])
    return rectangle


def rectangle_to_matrix(rectangles, rectangle_height, rectangle_width):
    result_matrix = []
    number_rectangle_in_row = image_height // rectangle_height
    number_rectangle_in_column = image_width // rectangle_width
    for count_rectangle_in_row in range(number_rectangle_in_row):
        for number_rectangle_row in range(rectangle_height):
            line = fill_in_line_for_result_matrix(rectangles, number_rectangle_row, number_rectangle_in_column,
                                                  count_rectangle_in_row, rectangle_width)
            result_matrix.append(line)
    return np.array(result_matrix)


def fill_in_line_for_result_matrix(rectangles, number_rectangle_row, number_rectangle_in_column, count_rectangle_in_row,
                                   rectangle_width):
    line = []
    for count_rectangle_in_column in range(number_rectangle_in_column):
        for number_rectangle_column in range(rectangle_width):
            dot = []
            for color in range(3):
                pixel_color = rectangles[
                    count_rectangle_in_row * number_rectangle_in_column + count_rectangle_in_column,
                    0, (number_rectangle_row * rectangle_width * 3) + (number_rectangle_column * 3) + color]
                dot.append(pixel_color)
            line.append(dot)
    return line


def update_weights_for_first_layer(w1, w2, Xi, delta_Xi):
    trans = np.transpose(Xi)
    return w1 - np.matmul(np.matmul((ALPHA * trans), delta_Xi), np.transpose(w2))


def update_weights_for_second_layer(w2, Yi, delta_Xi):
    return w2 - np.matmul((ALPHA * np.transpose(Yi)), delta_Xi)


def normalize_weights(w):
    for column_number in range(len(w[0])):
        sum = 0
        for row_number in range(len(w)):
            sum += w[row_number][column_number] ** 2
        sum = sum ** (0.5)
        for row_number in range(len(w)):
            w[row_number][column_number] = w[row_number][column_number] / sum
    return w


def save_array_to_file(path_name, array):
    np.save(path_name, array)


def load_array_from_file(path_name):
    return np.load(path_name)


def save_image_and_show(array_image, save_name):
    read_image = 1 * (array_image + 1) / 2
    plt.axis('off')
    plt.imshow(read_image)
    plt.savefig(save_name, transparent=True)
    plt.show()


def training(X, L, N, p, rectangle_height, rectangle_width, should_normalize, number_epochs):
    w1 = get_initial_weights(N, p)
    w2 = w1.transpose()
    errors = np.zeros(L)

    iteration = 0
    for number_epoh in range(number_epochs):
        sum_error = 0
        iteration += 1
        for number_rectangle in range(L):
            Yi = X[number_rectangle] @ w1
            Xe = Yi @ w2
            delta_Xi = Xe - X[number_rectangle]

            w1 = update_weights_for_first_layer(w1, w2, X[number_rectangle], delta_Xi)
            w2 = update_weights_for_second_layer(w2, Yi, delta_Xi)

            if should_normalize:
                w1 = normalize_weights(w1)
                w2 = normalize_weights(w2)

            errors[number_rectangle] = (delta_Xi ** 2).sum()
        sum_error = np.sum(errors)
        print(sum_error, iteration)
    save_array_to_file("first-layer-weights_" + str(rectangle_height) + "_" + str(rectangle_width) + "_" + str(p), w1)
    save_array_to_file("second-layer-weights_" + str(rectangle_height) + "_" + str(rectangle_width) + "_" + str(p), w2)
    return w1, w2


def input_height_width_rectangle_and_number_neural_second_layer():
    print('Enter number of rows in rectangle (m)')
    rectangle_height = int(input())
    print('Enter number of columns in rectangle (n)')
    rectangle_width = int(input())
    print('Enter number of neurals on the first layer (p)')
    p = int(input())  # number of neurons on the second layer
    return rectangle_height, rectangle_width, p


def compress_image_and_save_to_file():
    all_pixels = get_data(image_pixels)
    rectangle_height, rectangle_width, p = input_height_width_rectangle_and_number_neural_second_layer()
    L = (image_height * image_width) // (rectangle_height * rectangle_width)
    N = rectangle_height * rectangle_width * 3  # number of colors in Xi
    X = split_into_rectangles(all_pixels, rectangle_height, rectangle_width).reshape(L, 1, N)
    w1 = load_array_from_file("first-layer-weights_" + str(rectangle_height) + "_" + str(rectangle_width) + "_"
                              + str(p) + ".npy")

    Y = []
    for number_rectangle in range(L):
        Y.append(X[number_rectangle] @ w1)
    print(Y)
    save_array_to_file("compressed_image_array_" + str(rectangle_height) + "_" + str(rectangle_width) + "_" + str(p), Y)


def decompress_image_and_save_image():
    rectangle_height, rectangle_width, p = input_height_width_rectangle_and_number_neural_second_layer()
    L = (image_height * image_width) // (rectangle_height * rectangle_width)
    Y = load_array_from_file(
        "compressed_image_array_" + str(rectangle_height) + "_" + str(rectangle_width) + "_" + str(p) + ".npy")
    w2 = load_array_from_file(
        "second-layer-weights_" + str(rectangle_height) + "_" + str(rectangle_width) + "_" + str(p) + ".npy")
    result = []
    for number_rectangle in range(L):
        result.append(Y[number_rectangle] @ w2)
    result = np.array(result)
    save_image_and_show(rectangle_to_matrix(result, rectangle_height, rectangle_width),
                        "image_output_decompressed_" + str(rectangle_height) + "_" + str(rectangle_width) + "_"
                        + str(p) + ".png")


def training_and_save_result_image():
    rectangle_height, rectangle_width, p = input_height_width_rectangle_and_number_neural_second_layer()
    L = (image_height * image_width) // (rectangle_height * rectangle_width)  # number of rectangles
    N = rectangle_height * rectangle_width * 3  # number of colors in Xi

    print('Enter number of epochs')
    number_epochs = int(input())

    print('Choose how to train: 1 - with normalization, 0 - without normalization')
    should_normalize = bool(int(input()))

    all_pixels = get_data(image_pixels)
    rectangles = split_into_rectangles(all_pixels, rectangle_height, rectangle_width).reshape(L, 1, N)
    save_image_and_show(all_pixels, "image_output.png")
    w1, w2 = training(rectangles, L, N, p, rectangle_height, rectangle_width, should_normalize, number_epochs)

    result = []
    for rect in rectangles:
        result.append(np.matmul(np.matmul(rect, w1), w2))
    result = np.array(result)

    save_image_and_show(rectangle_to_matrix(result, rectangle_height, rectangle_width), "image_output1.png")


if __name__ == '__main__':
    image = Image.open('doge.png', 'r')
    image_pixels = image.load()
    image_width, image_height = np.size(image, 0), np.size(image, 1)
    ALPHA = 0.0003  # learning ratio

    while True:
        action_case = input("Select an action: 1) training, 2) compression image (with save to file),\n"
                            " 3) decompression image (with save to image), 4) exit\n")
        match action_case:
            case '1':
                training_and_save_result_image()
            case '2':
                compress_image_and_save_to_file()
            case '3':
                decompress_image_and_save_image()
            case '4':
                exit()
            case _:
                print("The language doesn't matter, what matters is solving problems.")
