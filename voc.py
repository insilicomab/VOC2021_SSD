'''
1. 訓練、検証のイメージとアノテーションのファイルパスのリストを作成する関数

'''
import os.path as osp

def make_filepath_list(rootpath):
    '''データのパスを格納したリストを作成する

    Parameters:
      rootpath(str): データフォルダーのルートパス

    Returns:
      train_img_list : 訓練用イメージのパスリスト
      train_anno_list: 訓練用アノテーションのパスリスト
      val_img_list   : 検証用イメージのパスリスト
      val_anno_list  : 検証用アノテーションのパスリスト
    '''
    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

    # 訓練データの画像ファイルとアノテーションファイルへのパスを保存するリスト
    train_img_list = []
    train_anno_list = []
    
    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        # %sをファイルIDに置き換えて画像のパスを作る
        img_path = (imgpath_template % file_id)
        # %sをファイルIDに置き換えて画アノテーションのパスを作る
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)  # train_img_listに追加
        train_anno_list.append(anno_path)# train_anno_listに追加

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)   # 画像のパス
        anno_path = (annopath_template % file_id) # アノテーションのパス
        val_img_list.append(img_path)   # リストに追加
        val_anno_list.append(anno_path) # リストに追加

    return train_img_list, train_anno_list, val_img_list, val_anno_list
  
'''
2. バウンディングボックスの座標と正解ラベルをリスト化するクラス

'''
import xml.etree.ElementTree as ElementTree # XMLを処理するライブラリ
import numpy as np # NumPy

class GetBBoxAndLabel(object):
    '''
    1枚の画像のアノテーション
    (BBoxの座標,ラベルのインデックス)をNumPy配列で返す

    Attributes:
      classes(list): VOCのクラス名(str)を格納したリスト
    '''
    def __init__(self, classes):
        '''インスタンス変数にクラスのリストを格納する
        
        Parameters:
          classes(list): VOCのクラス名(str)を格納したリスト 
        '''
        self.classes = classes

    def __call__(self, xml_path, width, height):
        '''インスタンスから実行されるメソッド
        
        1枚の画像のアノテーションデータをリスト化して多重リストにまとめる
        バウンディングボックスの各座標は画像サイズで割り算して正規化する

        Parameters:
          xml_path(str): xmlファイルのパス
          width(int): イメージの幅(正規化に必要)
          height(int): イメージの高さ(正規化に必要)

        Returns(ndarray):
          [[xmin, ymin, xmax, ymax, ラベルのインデックス], ... ]
          要素数は画像内に存在するobjectの数と同じ
        '''
        # 画像内の全ての物体のアノテーションを格納するリスト
        annotation = []

        # アノテーションのxmlファイルを読み込む
        xml = ElementTree.parse(xml_path).getroot()

        # イメージの中の物体（object）の数だけループする
        for obj in xml.iter('object'):
            # --アノテーションで検知がdifficultのものは除外--
            # difficultの値(0または1)をtextで取得してint型に変換
            # difficult==1の物体は処理せずにforの先頭に戻る
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 1つの物体に対するアノテーションを格納するリスト
            bndbox = []
            
            # <name>の要素(物体名)名を抽出
            # 小文字に変換後、両端の空白削除
            name = obj.find('name').text.lower().strip()
            # <bndbox>を取得
            bbox = obj.find('bndbox')
            
            # アノテーションの xmin, ymin, xmax, ymaxを取得し、0～1に規格化
            grid = ['xmin', 'ymin', 'xmax', 'ymax']

            for gr in (grid):
                # バウンディングボックスの座標<xmin><ymin><xmax><ymax>を取得
                # VOCは原点が(1,1)なので1を引き算して
                # 各オフセットの原点を（0, 0）の状態にする
                axis_value = int(bbox.find(gr).text) - 1
                # バウンディングボックスの座標を正規化
                if gr == 'xmin' or gr == 'xmax':
                    # xmin、xmaxの値をイメージの幅で割算
                    axis_value /= width
                else:
                    # ymin、ymaxの値はイメージの高さで割算
                    axis_value /= height
                # 'xmin' 'ymin' 'xmax' 'ymax'の値を順にbndboxに追加
                bndbox.append(axis_value)

            # 物体名のインデックスを取得
            label_idx = self.classes.index(name)
            # bndboxにインデックスを追加して物体のアノテーションリストを完成
            bndbox.append(label_idx)

            # すべてのアノテーションリストをannotationに格納
            annotation += [bndbox]
        # 多重リスト[xmin, ymin, xmax, ymax, 正解ラベルのインデックス], ...]
        # を2次元のNumPy配列(ndarray)に変換
        return np.array(annotation)

'''
3. イメージとアノテーションの前処理を行うDataTransformクラス

'''
# augumentation.pyからデータの前処理をするクラスをインポート
from augmentations import Compose, ConvertFromInts, ToAbsoluteCoords, \
                          PhotometricDistort, Expand, RandomSampleCrop, \
                          RandomMirror, ToPercentCoords, Resize, SubtractMeans

class DataTransform(object):
    '''データの前処理クラス
    
    イメージのサイズを300x300にリサイズ
    訓練時は拡張処理を行う
    
    Attributes:
      data_transform(dict): 前処理メソッドを格納した辞書
    '''
    def __init__(self, input_size, color_mean):
        '''データの前処理を設定
        
        訓練時(train)と検証時(val)で異なる処理を行う
        
        Parameters:
          input_size(int): イメージをリサイズするときの大きさ
          color_mean(B, G, R): 色チャネルB,G,Rそれぞれの平均値
        '''
        self.transform = {
            'train': Compose([
                ConvertFromInts(),    # ピクセルデータのintをfloat32に変換
                ToAbsoluteCoords(),   # アノテーションの正規化を元の状態に戻す
                PhotometricDistort(), # コントラスト、色相、輝度の変化、歪み
                Expand(color_mean),   # イメージを拡大
                RandomSampleCrop(),   # イメージからランダムに切り出す
                RandomMirror(),       # ランダムにイメージを反転させる
                ToPercentCoords(), # アノテーションデータを0～1.0の範囲に正規化
                Resize(input_size),# イメージのサイズをinput_sizeにリサイズ
                SubtractMeans(color_mean)# ピクセルデータ(RGB）から平均値を引き算
            ]),
            'val': Compose([
                ConvertFromInts(),  # ピクセルデータのintをfloat32に変換
                Resize(input_size), # イメージのサイズをinput_sizeにリサイズ
                SubtractMeans(color_mean) # ピクセルデータ(RGB）から平均値を引き算
            ])
        }
        
    def __call__(self, img, phase, boxes, labels):
        '''データの前処理を実施
           DataTransformのインスタンスから実行される
        
        Parameters:
          img(Image): イメージ
          phase(str): 'train'または'val'
          boxes(Tensor): BBoxの座標(xmin,ymin,xmax,ymax)
          labels (Tensor): 正解ラベルのインデックス
        '''
        return self.transform[phase](img, boxes, labels)
      
'''
4. データセットの数だけ繰り返し呼ばれる前処理オブジェクトを生成
   __getitem__()は1データにつき前処理後のイメージ、BBoxとラベルの配列を返す
   
'''
import torch
import torch.utils.data as data
import cv2

class PreprocessVOC2012(data.Dataset):
    ''' PyTorchのDatasetクラスを継承
        DataTransformでVOC2012データセットを前処理して以下のデータを返す
        
       ・前処理後のイメージ[R,G,B](Tensor)
       ・BBoxとラベル(ndarray)
       ・イメージの高さ、幅(int)
      
       Datasetは、__getitem__()と__len__()の実装が必要
    '''
    def __init__(self,
                 img_list, anno_list, phase, transform, get_bbox_label):
        '''
        Parameters:
          img_list(list): イメージのファイルパスを格納したリスト
          anno_list(list): アノテーションのファイルパスを格納したリスト
          phase(str): 'train'または'test'で訓練か検証かを指定
          transform(object): 前処理クラスDataTransform
          bbox_label(object): BBox座標と正解ラベルを取得するGetBBoxAndLabel
        '''
        self.img_list = img_list    # イメージのファイルパスリスト
        self.anno_list = anno_list  # アノテーションのファイルパスリスト
        self.phase = phase          # trainまたはval
        self.transform = transform  # DataTransformオブジェクト
        self.get_bbox_label = get_bbox_label # GetBBoxAndLabelオブジェクト

    def __len__(self):
        '''イメージの数を返す
        '''
        return len(self.img_list)

    def __getitem__(self, index):
        '''データの数だけイテレート(反復処理)される
           前処理後のイメージ、BBox座標とラベルの2次元配列を取得する
        
        Parameter:
          index(int): 訓練、または検証用イメージのインデックス
          
        Returns:
          im(Tensor):
            前処理後のイメージを格納した3階テンソル
            (3, 高さのピクセル数, 幅のピクセル数)) 3はRGB
          bl(ndarray):
            BBoxとラベルの2次元配列
        '''
        # pull_item()にイメージのインデックスを渡して前処理
        # 処理後のイメージデータとBBoxとラベルの2次元配列を返す
        im, bl, _, _ = self.pull_item(index)
        return im, bl

    def pull_item(self, index):
        '''前処理後のテンソル形式のイメージデータ、アノテーション、
           イメージの高さ、幅を取得する
        
        Parameter:
          index(int): 訓練、または検証用イメージのインデックス
        
        Returns:
          img(Tensor): 前処理後のイメージ(3, 高さのピクセル数, 幅のピクセル数)
          boxlbl(ndarray): BBoxとラベルの2次元配列(複数の物体があるので)
          height(int): イメージの高さ
          width(int): イメージの幅
        '''
        # イメージの高さ、幅、チャネル数を取得
        img_path = self.img_list[index] # インデックスを指定してイメージのパスを取得
        img = cv2.imread(img_path)   # OpenCV2でイメージの[高さ,幅,[B,G,R]]を取得
        height, width, _ = img.shape # 配列要素数を数えて高さ,幅のみを取得

        # アノテーションデータのリストを取得
        # インデックスを指定してアノテーションファイル(xml)のパスを取得
        anno_file_path = self.anno_list[index]
        # アノテーションファイルからBBoxの座標、正解ラベルのリストを取得
        bbox_label = self.get_bbox_label(anno_file_path, # XMLのパス
                                         width,          # イメージ幅
                                         height)         # イメージ高さ

        # DataTransformで前処理を実施
        img, boxes, labels = self.transform(
            img,               # OpneCV2で読み込んだイメージデータ
            self.phase,        # 'train'または'val'
            bbox_label[:, :4], # BBoxの座標
            bbox_label[:, 4])  # 正解ラベルのインデックス

        # img(ndarray)の形状は(高さのピクセル数, 幅のピクセル数, 3))
        # 3はBGRの並びなのでこれをRGBの順に変更
        # (3, 高さのピクセル数, 幅のピクセル数)の形状の3階テンソルにする
        img = torch.from_numpy(
            img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        
        # [label,label, ...]を[[label],[label], ...]のように次元拡張して
        # [[xmin,ymin,xmax,ymax], ...]と水平方向に連結し、
        # [[xmin,ymin,xmax,ymax,label], ...]の形状にする
        boxlbl = np.hstack(
            (boxes, np.expand_dims(labels, axis=1)))

        # 前処理後のイメージ[R,G,B]、BBoxとラベルのndarray、
        # イメージの高さ、幅を返す
        return img, boxlbl, height, width

'''
5. ミニバッチを作る関数

'''
def multiobject_collate_fn(batch):
    '''Pytorchのcollate_fn()をカスタマイズ
    
    イメージとイメージに対応するアノテーション(複数あり)を
    ミニバッチの数だけ生成する機能を実装
    
    Parameters:
      batch(tuple):
        PreprocessVOC2012の__getitem__()で返される要素数2のタプル
        (処理後のイメージ(Tensorのlist), BBox座標とラベルの2次元ndarray)
    
    Returns:
      imgs(Tensor):
        前処理後のイメージ(RGB)をミニバッチの数だけ格納した4階テンソル
        形状は(ミニバッチのサイズ, 3, 300, 300)
      targets(list):
        [[xmin,ymin,xmax,ymax,label], ...]の2階テンソル([物体数, 5])
        を格納したリスト、要素数はミニバッチの数
    '''
    imgs = []    # ミニバッチのイメージデータ(テンソル)を保持
    targets = [] # ミニバッチのBBox座標とラベルの2次元配列を保持
    
    # ミニバッチの前処理後のイメージをimgsに追加
    # ミニバッチのBBox座標とラベルの2次元配列をtargetsに追加
    for sample in batch:
        # sample[0] はイメージデータ[R,G,B](torch.Size([3, 300, 300]))
        imgs.append(sample[0])
        # sample[1] はBBox座標とラベルの2次元配列
        # これをTensorにしてリストtargetsに追加
        targets.append(torch.FloatTensor(sample[1]))

    # リストimgsの要素torch.Size([3, 300, 300])の0次元を拡張して
    # (ミニバッチのサイズ, 3, 300, 300)の4階テンソルにする
    imgs = torch.stack(imgs, dim=0)

    # targetsは[[xmin,ymin,xmax,ymax,label], ...]の2階テンソル
    # すなわち[物体数, 5]の2階テンソルをミニバッチの数だけ格納したリスト
    # リストの中身は[[物体数, 5],[物体数, 5],...]となる
    return imgs, targets
