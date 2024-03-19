import sys, os
# 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(img)
    pil_img.show()


# 데이터 불러오기
(X_train, y_train), (X_test, y_test) = load_mnist(flatten=True, normalize=False)

# 이미지 출력
img = X_train[0]
label = y_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
