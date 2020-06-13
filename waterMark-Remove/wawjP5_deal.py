# coding: utf8
import ssl
import cv2
import urllib.request
import traceback
import requests
ssl._create_default_https_context = ssl._create_unverified_context

def download_img(y_img_url,x_img_url,Type,n):
    try:
        # urllib.request.urlretrieve(x_img_url, filename = 'dataset/%s/x_%s/'%(Type,Type) + str(n) + '.jpg')
        # x_img_yuan = cv2.imread('dataset/%s/x_%s/'%(Type,Type) + str(n) + '.jpg')
        urllib.request.urlretrieve(x_img_url, filename = '2.jpg')
        x_img_yuan = cv2.imread('2.jpg')
        shape = x_img_yuan.shape
        print(shape)
        urllib.request.urlretrieve(y_img_url, filename='1.jpg')
        y_img_yuan = cv2.resize(cv2.imread('1.jpg'), (shape[1],shape[0]) , interpolation=cv2.INTER_CUBIC)
        x_img = x_img_yuan[370:480,660:800]
        cv2.imwrite('dataset/%s/x_%s/' % (Type, Type) + str(n) + '.jpg', x_img)
        y_img = y_img_yuan[370:480,660:800]
        cv2.imwrite('dataset/%s/y_%s/'%(Type,Type) + str(n) + '.jpg',y_img)
    except:
        print(traceback.format_exc())
        print('error:%s'%n + '  ' + x_img_url)



if __name__ == '__main__':
    # 下载要的图片
    n = 0
    with open('wawj.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            if n<=10000:
                Type = 'train'
            else:
                Type = 'test'
            x_img_url = line.replace('\n','')
            if 'style/P5' in x_img_url:
                n += 1
                y_img_url = x_img_url.replace('P5','P3')
                download_img(y_img_url,x_img_url,Type, n)

