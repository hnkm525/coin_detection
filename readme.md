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

## 参考にしたサイト
- http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
