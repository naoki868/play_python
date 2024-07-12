import cv2
from matplotlib import pyplot as plt

# 画像を表示する関数
def show_image(img):
    # OpenCVはBGRで画像を扱うため、表示前にRGBに変換する
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

# 顔検出用のカスケード分類器をロード
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 検出を行いたい画像を読み込み
img = cv2.imread('sample2.jpg')

# グレースケール画像に変換（検出精度向上のため）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔の検出を実行
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# 検出された顔の周りに矩形を描画
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 画像を表示
show_image(img)
