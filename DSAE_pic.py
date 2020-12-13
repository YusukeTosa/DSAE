import sys , os
import random

import numpy as np
import cv2


# データセットから画像とラベルをランダムに取得
def random_sampling( images, img_gray , train_num):
    image_train_batch = []
    image_train_batch_gray = []

    #乱数を発生させ，リストを並び替える．
    random_seq = list(range(len(images)))
    random.shuffle(random_seq)

    # バッチサイズ分画像を選択
    image_train_batch = images[:train_num]
    image_train_batch_gray = img_gray[:train_num]

    return image_train_batch, image_train_batch_gray


# フォルダーの画像をランダム位置でクリップした後にリサイズして読み込む
def make( folder_name="pendulun_pic" , img_size = 240 ,train_num = 300):
    train_image = []
    test_image = []

    files = os.listdir(folder_name)

    tmp_image = []
    tmp_image_gray = []
    count_ = 0
    for f in files:
        # 1枚の画像に対する処理
        if not 'jpg' in f:# jpg以外のファイルは無視
            continue

        # 画像読み込み
        img = cv2.imread(folder_name + '/' + f)
        # リサイズをする処理
        img = cv2.resize(img , (img_size , img_size))
        im_gray = cv2.resize(img , (60 , 60))
        im_gray = cv2.cvtColor(im_gray, cv2.COLOR_BGR2GRAY)
        img = 1 - img.astype(np.float32)/255.0
        im_gray = 1 - im_gray.astype(np.float32)/255.0

        tmp_image.append(img)
        tmp_image_gray.append(im_gray)

        if count_ % 100 == 0:
            print(count_)
        count_ += 1

        if count_ > 300:
            break


    # データセットのサイズが指定されているときはその枚数分ランダムに抽出する
    if train_num == 0 :
        train_image.extend( tmp_image )
    #　テスト画像の指定がないとき
    sampled_image, sampled_image_gray = random_sampling(tmp_image, tmp_image_gray, train_num)
    train_image.extend(sampled_image)
    test_image.extend(sampled_image_gray)

    # numpy配列に変換
    train_image = np.asarray( train_image )
    test_image = np.asarray( test_image )

    return train_image, test_image


def main(folder_name="pendulun_pic" , img_size = 60):
    train_image = []
    test_image = []

    files = os.listdir(folder_name)

    tmp_image = []
    count_ = 0
    for f in files:
        # 1枚の画像に対する処理
        if not 'jpg' in f:# jpg以外のファイルは無視
            continue

        # 画像読み込み
        img = cv2.imread(folder_name + '/' + f)
        # リサイズをする処理
        img = cv2.resize( img , (img_size , img_size))
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = im_gray.astype(np.float32)/255.0

        if count_ % 100 == 0:
            print(count_)
        count_ += 1

        if count_ > 300:
            break

