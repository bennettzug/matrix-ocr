import cv2
import numpy as np
import pytesseract
from collections import defaultdict
import sympy
from PIL import ImageGrab


pytesseract.pytesseract.tesseract_cmd = (
    r"/opt/homebrew/Cellar/tesseract/5.3.4_1/bin/tesseract"
)
custom_config = r"--oem 3 --psm 10 -c tessedit_char_whitelist='0123456789-'"


def img_to_matrix(img):
    copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounds = []
    copy3 = thresh.copy()
    copy3 = cv2.cvtColor(copy3, cv2.COLOR_GRAY2BGR)
    hsum = 0
    hcount = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(copy3, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if w * 5 < h:
            bounds.append((x, y, w, h))
        elif h > w:
            hsum += h
            hcount += 1

    left_x = min([x for x, _, _, _ in bounds])
    right_x = max([x + w for x, _, w, _ in bounds])
    top_y = min([y for _, y, _, _ in bounds])
    bottom_y = max([y + h for _, y, _, h in bounds])
    cropped = copy[top_y:bottom_y, left_x:right_x]
    avgh = hsum // hcount
    new_row_height = 20
    scaling_factor = new_row_height / avgh

    if scaling_factor < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    scaled = cv2.resize(
        cropped,
        (0, 0),
        fx=scaling_factor,
        fy=scaling_factor,
        interpolation=interpolation,
    )

    gray2 = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    _, thresh2 = cv2.threshold(blurred2, 200, 255, cv2.THRESH_BINARY_INV)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    copy2 = scaled.copy()
    naive_arr = []
    img_reprentation = []
    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)
        if w * 5 < h:
            # cv2.rectangle(copy2, (x, y), (x + w, y + h), (0, 0, 255), 2)
            break
        else:
            if h < w:
                text = "-"
                # cv2.rectangle(copy2, (x, y), (x + w, y + h), (255, 255, 0), 2)
            else:
                text = pytesseract.image_to_string(
                    copy2[
                        max(0, y - 3) : min(y + h + 3, copy2.shape[0]),
                        max(0, x - 3) : min(x + w + 3, copy2.shape[1]),
                    ],
                    config=custom_config,
                )

                # copy2[y - 5 : y + h + 5, x - 5 : x + w + 5], config=custom_config
                text = text.split("\n")[0]
                # cv2.rectangle(copy2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(
            #     copy2, text, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2
            # )
            # cv2.imshow("image", copy2)
            # cv2.waitKey(0)
            img_reprentation.append((x, y, w, h, text))
            naive_arr.append(text.split("\n")[0])

    buckets = []

    ys = []

    for elem in img_reprentation:
        _, y, _, _, _ = elem
        ys.append(y)
    for y in ys:
        # Flag to check if y is within tolerance of any existing bucket
        found = False
        for elem in buckets:
            if (
                elem - (new_row_height * 2) // 3
                <= y
                <= elem + (new_row_height * 2) // 3
            ):
                found = True
                break
        if not found:
            buckets.append(y)

    arranged_items = defaultdict(list)
    for elem in img_reprentation:
        x, y, w, h, text = elem
        for bucket in buckets:
            if (
                bucket - (new_row_height * 2) // 3
                <= y
                <= bucket + (new_row_height * 2) // 3
            ):
                arranged_items[bucket].append((x, y, w, h, text))

    new_buckets = []
    for y in sorted(buckets):
        elems = arranged_items[y]
        elems = sorted(elems, key=lambda x: x[0])
        new_buckets.append(elems)

    merged_buckets = []

    for bucket in new_buckets:
        # Merge elements based on x values
        merged_bucket = []
        i = 0
        while i < len(bucket):
            x, y, w, h, text = bucket[i]
            # Merge adjacent digits or symbols
            while i + 1 < len(bucket) and bucket[i + 1][0] - (x + w) < 10:
                w = bucket[i + 1][0] + bucket[i + 1][2] - x
                text += bucket[i + 1][4]
                i += 1
            merged_bucket.append((x, y, w, h, text))
            i += 1
        merged_buckets.append(merged_bucket)
    # if all rows are not the same length, rerun analyis with estimated row height
    # if len(set([len(bucket) for bucket in merged_buckets])) != 1:

    #     most_common = max(set([len(bucket) for bucket in merged_buckets]), key = [len(bucket) for bucket in merged_buckets].count)
    #     # number of rows: how many rows have most common length
    #     num_rows = [len(bucket) for bucket in merged_buckets].count(most_common)
    #     # estimated row height
    #     row_height = copy2.shape[0] // num_rows

    matrix = []
    for bucket in merged_buckets:
        row = []
        test = []
        for i, elem in enumerate(bucket):
            _, _, _, _, text = elem
            if text == "":
                continue
            if text == "-":
                e = ValueError("Image Parsing Error")
                raise e
            test.append(text)
            row.append(int(text))

        matrix.append(row)

    return matrix


def append_vector(matrix, vector):
    for i in range(len(matrix)):
        matrix[i].append(vector[i])
    return matrix


def main():
    image = ImageGrab.grabclipboard()
    if image:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        print("No image found in clipboard")
        return
    matrix = img_to_matrix(img)
    print("Current matrix is:")
    try:
        sympy.pprint(sympy.Matrix(matrix))
    except ValueError:
        print("Matrix parsed incorrectly, please try again.")
        print("(Incorrectly) Parsed matrix is:")
        print(matrix)
        return

    append = input(
        "What matrix should be appended? \n0 for 0 matrix, b for b matrix, s for screenshot, n for none: "
    )

    if append == "0":
        matrix = append_vector(matrix, [0] * len(matrix))
    elif append == "b":
        b = [0] * len(matrix)
        for i in range(len(matrix)):
            b[i] = int(input(f"Enter b_{i + 1}: "))
        matrix = append_vector(matrix, b)
    elif append == "s":
        input("Press enter when you have the matrix in your clipboard: ")
        image2 = ImageGrab.grabclipboard()

        if image:
            img2 = np.array(image2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            b = img_to_matrix(img2)
            if len(b) != len(matrix):
                print("Matrix sizes do not match")
                return

            flat_b = [item for sublist in b for item in sublist]

            matrix = append_vector(matrix, flat_b)
        else:
            print("No image found in clipboard")
            return

    elif append == "n":
        pass
    else:
        print("Invalid input")
        return
    A = sympy.Matrix(matrix)
    sympy.init_printing(use_unicode=True)
    print("Augmented matrix is:")
    sympy.pprint(A)
    print("Row Reduced matrix is:")
    sympy.pprint(A.rref()[0])


if __name__ == "__main__":
    main()
