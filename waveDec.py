# -*- coding: utf-8 -*-

import pywt
import csv
from PIL import Image

def waveletDecomp(series):
    # c6, k6, k5, k4 ,k3, k2, k1 = pywt.wavedec(series,"db6",level=6)
    (c6,k6),(c5,k5),(c4,k4),(c3,k3),(c2,k2),(c1,k1)=pywt.swt(series,"haar",level=6)
    wave_matrix = []
    for i in range(10):
        #count = int(32/len(k6))
        #map(lambda n:[wave_matrix.append(n) for i in range(count)], k6)
        map(lambda n:wave_matrix.append(n), k6)
    for i in range(10):
        #count = int(32/len(k5))
        #map(lambda n:[wave_matrix.append(n) for i in range(count), k5)
        map(lambda n:wave_matrix.append(n), k5)
    for i in range(11):
        #count = int(32/len(k4))
        #map(lambda n:[wave_matrix.append(n) for i in range(count)], k4)
        map(lambda n:wave_matrix.append(n), k4)
    for i in range(11):
        #count = int(32/len(k3))
        #map(lambda n:[wave_matrix.append(n) for i in range(count)], k3)
        map(lambda n:wave_matrix.append(n), k3)
    for i in range(11):
        #count = int(32/len(k2))
        #map(lambda n:[wave_matrix.append(n) for i in range(count)], k2)
        map(lambda n:wave_matrix.append(n), k2)
    for i in range(11):
        #count = int(32/len(k1))
        #map(lambda n:[wave_matrix.append(n) for i in range(count)], k1)
        map(lambda n:wave_matrix.append(n), k1)
    return wave_matrix


if __name__ == '__main__':
    dataReader = csv.reader(open("USDJPY.csv","rU"))
    prices = []
    bools = []
    for row in list(dataReader):
        a = float(row[0])
        b = int(row[1])
        c = int(row[2])
        prices.append(a)
        bools.append([b,c])
    imageFile = open("test_image_swt_2.csv","ab")
    labelFile = open("test_label_swt_2.csv","ab")
    writer = csv.writer(imageFile)
    labelwriter = csv.writer(labelFile)
    rows = []
    labelrows = []
    priceLength = len(prices)
    print priceLength
    maxCoefficient = 0.0
    minCoefficient = 0.0
    for i in range(priceLength-64):
        wave_matrix = waveletDecomp(prices[i:i+64])
        if maxCoefficient < max(wave_matrix):
            maxCoefficient = max(wave_matrix)
        if minCoefficient > min(wave_matrix):
            minCoefficient = min(wave_matrix)
        #bool_value = bools[i+63]
        #labelrows.append(bool_value)
        #rows.append(wave_matrix)
        if i > 600:
            break
    # for i in range(priceLength-64):
    #     wave_matrix = waveletDecomp(prices[i:i+64])
    #     bool_value = bools[i+63]
    #     labelrows.append(bool_value)
    #     rows.append(map(lambda n:(n-minCoefficient)/(maxCoefficient-minCoefficient), wave_matrix))
    #     if i > 600:
    #         break
    # writer.writerows(rows)
    # labelwriter.writerows(labelrows)
    # imageFile.close()
    # labelFile.close()
    wave_matrices = waveletDecomp(prices[1:65])
    maxValue = max(wave_matrices)
    minValue = min(wave_matrices)
    img = Image.new("RGB",(64,64),(0,0,0))
    for y in range(64):
        for x in range(64):
            value = wave_matrices[y*64+x]
            color = 255.0*(value-minValue)/(maxValue-minValue)
            img.putpixel((x,y),(int(color),int(color),int(color)))
    img.save("wavelet_swt_2.jpg")
