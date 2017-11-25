# 行動班

## Dependencies
* numpy
* OpenCV (3.1.0+)
* cPickle
* more\_itertools

## 使い方
* 元画像ファイル (id\_xxx.jpg)を置いたフォルダを"minipose"として直下に配置
* 抽出済み姿勢及び姿勢プロットを置いたフォルダを"minipose\_annotation"として同様に直下に配置

そのうえで
```
python parse_pose.py
python train.py
```
結果が"plots"フォルダに保存される。
