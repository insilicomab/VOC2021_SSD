import numpy as np
import matplotlib.pyplot as plt 
import cv2  # OpenCVライブラリ
import torch

#from utils.ssd_model import DataTransform
from voc import DataTransform

class SSDPredictions():
    '''SSDモデルで物体検出を行うクラス
    
    Attributes:
      eval_categories(list): クラス名(str)
      net(object): SSDモデル
      transform(object): 前処理クラス
    '''
    def __init__(self, eval_categories, net):
        # クラス名のリストを取得
        self.eval_categories = eval_categories
        # SSDモデル
        self.net = net
        color_mean = (104, 117, 123)  # VOCデータの色の平均値(BGR)
        input_size = 300  # 画像の入力サイズは300×300
        # 前処理を行うDataTransformオブジェクトを生成
        self.transform = DataTransform(input_size, color_mean)

    def show(self, image_file_path, confidence_threshold):
        '''物体検出の予測結果を出力する

        Parameters:
          image_file_path(str): 画像のファイルパス
          confidence_threshold(float): 確信度の閾値
        '''
        # SSDモデルで物体検出を行い、確信度が閾値以上のBBoxの情報を取得
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path,      # 画像のファイルパス
            confidence_threshold) # 確信度の閾値
        
        # 検出結果を写真上に描画する
        self.draw(rgb_img,           # 画像のRGB値
                  bbox=predict_bbox, # 物体を検出したBBoxのリスト
                  label_index=pre_dict_label_index, # 物体のラベルへのインデックス
                  scores=scores,                    # 物体の確信度
                  label_names=self.eval_categories) # クラス名のリスト

    def ssd_predict(self, image_file_path, confidence_threshold=0.5):
        '''SSDで物体検出を行い、確信度が高いBBoxの情報を返す

        Parameters:
          image_file_path(str): 画像のファイルパス
          confidence_threshold(float): 確信度の閾値
        
        Returns: 1画像中で物体を検出したBBoxの情報
          rgb_img: 画像のRGB値
          predict_bbox: 物体を検出したBBoxの情報
          pre_dict_label_index: 物体を検出したBBoxが予測する正解ラベル
          scores: 各BBoxごとの確信度
        '''

        # 画像データを取得
        img = cv2.imread(image_file_path)
        # ［高さ］, ［幅］, ［RGB値］の要素数をカウントして画像のサイズとチャネル数を取得
        height, width, channels = img.shape
        # BGRからRGBへ変換
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 画像の前処理
        phase = 'val'
        img_transformed, boxes, labels = self.transform(
            img,  # OpneCV2で読み込んだイメージデータ
            phase,# 'val'
            '',   # アノテーションは存在しないので''
            '') 
        # img_transformed(ndarray)の形状は(高さのピクセル数,幅のピクセル数,3)
        # 3はBGRの並びなのでこれをRGBの順に変更
        # (3, 高さのピクセル数, 幅のピクセル数)の形状の3階テンソルにする
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # 学習済みSSDモデルで予測
        self.net.eval()  # ネットワークを推論モードにする
        x = img.unsqueeze(0)  # imgの形状をミニバッチの(1,3,300,300)にする
        # detections: 1枚の画像の各物体に対するBBoxの情報が格納される
        # (1, 21(クラス), 200(Top200のBBox), 5)
        # 最後の次元の5は[BBoxの確信度, xmin, ymin, width, height]
        detections = self.net(x)

        # confidence_threshold:
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        # 予測結果から物体を検出したとする確信度の閾値以上のBBoxのインデックスを抽出
        # find_index(tuple): (［0次元のインデックス］,
        #                     ［1次元のインデックス],
        #                     [2次元のインデックス],
        #                     [3次元のインデックス],)
        find_index = np.where(detections[:, 0:, :, 0] >= confidence_threshold)
        
        # detections: (閾値以上のBBox数, 5)
        detections = detections[find_index]
        
        # find_index[1]のクラスのインデックスの数(21)回ループする
        for i in range(len(find_index[1])):
            if (find_index[1][i]) > 0: # クラスのインデックス0以外に対して処理する
                sc = detections[i][0]  # detectionsから確信度を取得
                # BBoxの座標[xmin, ymin, width, height]のそれぞれと
                # 画像の[width, height, width, height]をかけ算する
                bbox = detections[i][1:] * [width, height, width, height]
                # find_indexのクラスの次元の値から-1する(背景0を引いて元の状態に戻す)
                lable_ind = find_index[1][i]-1

                # BBoxのリストに追加
                predict_bbox.append(bbox)
                # 物体のラベルを追加
                pre_dict_label_index.append(lable_ind)
                # 確信度のリストに追加
                scores.append(sc)
        
        # 1枚の画像のRGB値、BBox、物体のラベル、確信度を返す
        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def draw(self, rgb_img, bbox, label_index, scores, label_names):
        '''物体検出の予測結果を写真上に描画する関数。

        Parameters:
          rgb_img: 画像のRGB値
          bbox(list): 物体を検出したBBoxのリスト
          label_index(list): 物体のラベルへのインデックス
          scores(list): 物体の確信度
          label_names(list): ラベル名の配列
        '''
        # クラスの数を取得
        num_classes = len(label_names)
        # BBoxの枠の色をクラスごとに設定
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 画像を表示
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        # 物体を検出したBBoxの数だけループ
        for i, bb in enumerate(bbox):
            # 予測した正解ラベルを取得
            label_name = label_names[label_index[i]]
            # ラベルに応じてBBoxの枠の色を変える
            color = colors[label_index[i]]
            
            # 物体名と確信度をBBoxの枠上に表示する
            # 例：person：0.92　
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            # BBoxの座標を取得
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            # BBoxを描画
            currentAxis.add_patch(plt.Rectangle(
                xy,
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=2)
                )

            # BBoxの枠の左上にラベルを描画
            currentAxis.text(
                xy[0],
                xy[1],
                display_txt,
                bbox={'facecolor': color, 'alpha': 0.5}
                )
