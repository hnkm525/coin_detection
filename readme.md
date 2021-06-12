# 硬貨検出プログラム

## requirements 
- Python 3.9.5
- tensorflow 2.4.1
- opencv 4.4.0

## ハフ変換による円検出
ハフ変換を用いて円(硬貨)を検出している．
coin_detection.pyから実行可能

## VGG16を用いた硬貨の分類
VGG16と呼ばれるニューラルネットワークを用いる事で検出した硬貨の種類を分類している．
ImageNetを学習済み．