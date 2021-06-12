# 硬貨検出プログラム

## requirements 
- Python 3.9.5
- tensorflow-2.4.1
- tensorflow-gpu-2.4.1
- cudatoolkit-10.1.243
- cudnn-7.6.5
- opencv 4.4.0
- pillow-8.2.0

## ハフ変換による円検出
ハフ変換を用いて円(硬貨)を検出している．  
coin_detection.pyから実行可能

## VGG16転移学習モデルを用いた硬貨の分類
ImageNetで学習済みのVGG16を用いた転移学習による硬貨画像の分類を行う．   
224×224の画像を入力すると分類結果が出力される．
![image](https://user-images.githubusercontent.com/29078336/121790463-953dbf80-cc1a-11eb-8585-efb4a63ae970.png)
![image](https://user-images.githubusercontent.com/29078336/121790464-a25aae80-cc1a-11eb-8b36-3fb912fed939.png)
![image](https://user-images.githubusercontent.com/29078336/121790469-b0103400-cc1a-11eb-9cda-18897b6d6272.png)
![image](https://user-images.githubusercontent.com/29078336/121790476-bbfbf600-cc1a-11eb-9754-b481abe4e5f6.png)
![image](https://user-images.githubusercontent.com/29078336/121790483-c4ecc780-cc1a-11eb-9873-8111f7c9c8ea.png)
![image](https://user-images.githubusercontent.com/29078336/121790485-d0d88980-cc1a-11eb-9e89-31d9798eb59b.png)


## 参考にしたサイト
- http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
