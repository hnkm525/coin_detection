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
現在認識精度は微妙…  
epoch数が足りていないのかも
![image](https://user-images.githubusercontent.com/29078336/121788691-45ef9300-cc0a-11eb-90a3-53714207e2ec.png)
![image](https://user-images.githubusercontent.com/29078336/121788692-4ee06480-cc0a-11eb-97f1-26dcc77df919.png)
![image](https://user-images.githubusercontent.com/29078336/121788696-556edc00-cc0a-11eb-84d0-5d87195ac8a2.png)


## 参考にしたサイト
- http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
