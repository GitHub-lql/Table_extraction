import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import xlsxwriter
import cv2


def preprocess_image(image):
    """
    对输入图像进行预处理,包括灰度化、增强对比度、锐化、自适应阈值二值化、Otsu二值化、去噪和形态学操作
    Preprocess the input image, including grayscale, enhanced contrast, sharpening, adaptive threshold binarization, Otsu binarization, denoising, and morphological operations

    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 增强对比度 Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    contrast_enhanced = clahe.apply(gray)

    # 锐化 sharpening
    kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
    cv2.imshow('gray',sharpened)
    cv2.waitKey()
    # 自适应阈值二值化  Adaptive threshold binarization
    thresh_adaptive = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7,
                                            3)

    # Otsu二值化
    _, thresh_otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    opening = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow('gray',closing)
    cv2.waitKey()
    # print(opening)
    return closing



def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_table_contour(contours):
    areas = [cv2.contourArea(cnt) for cnt in contours]
    table_contour = contours[np.argmax(areas)]
    return table_contour


def get_cells(table_contour, image):
    x, y, w, h = cv2.boundingRect(table_contour)
    table = image[y:y + h, x:x + w]

    scale = 150
    vert_kernel_height = max(3, h // scale)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_height))
    v_lines = cv2.morphologyEx(table, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    hori_kernel_width = max(3, w // scale)
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hori_kernel_width, 1))
    h_lines = cv2.morphologyEx(table, cv2.MORPH_OPEN, hori_kernel, iterations=1)

    table_lines = cv2.add(v_lines, h_lines)
    cell_contours, _ = cv2.findContours(table_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = [cv2.boundingRect(cnt) for cnt in cell_contours]
    rows = [[] for _ in range(max(bbox[1] + bbox[3] for bbox in bboxes) + 1)]
    for bbox in bboxes:
        x, y, w, h = bbox
        row = rows[y]
        row.append((x, y, w, h))
        row.sort(key=lambda x: x[0])

    cells = []
    for row in rows:
        if len(row) == 0:
            continue
        cells_row = []
        for x, y, w, h in row:
            cell = table[y:y + h, x:x + w]
            cells_row.append(cell)
        cells.append(cells_row)

    return cells


def extract_text_from_cells(cells):
    data = []
    for row in cells:
        data_row = []
        for cell in row:
            text = pytesseract.image_to_string(cell)
            data_row.append(text.strip())
        data.append(data_row)
    return data


def main():
    image = cv2.imread('data_1.jpg')
    processed = preprocess_image(image)
    contours = find_contours(processed)
    table_contour = get_table_contour(contours)
    cells = get_cells(table_contour, processed)
    data = extract_text_from_cells(cells)
    print("提取的数据:")
    print(data)
    df = pd.DataFrame(data)

    with pd.ExcelWriter('output.xlsx', engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        for i, col in enumerate(df.columns):
            column_len = df[col].astype(str).str.len().max()
            column_len = max(column_len, len(str(col))) + 2
            worksheet.set_column(i, i, column_len)

    print("提取的表格已成功导出到output.xlsx文件")


if __name__ == "__main__":
    main()