'''
1. vggネットワークを生成する関数の定義
'''
import torch.nn as nn

def make_vgg():
    '''
    Returns：
      (nn.ModuleList): vggのモジュール(部品)のリスト
    '''
    layers = []      # モジュールを格納するリスト
    in_channels = 3  # チャネル数はRGBの3値

    # vggに配置する畳み込み層のフィルター数(チャネル数に相当)
    # 'M''MC'はプーリング層を示す
    cfg = [64, 64, 'M',         # vgg1
           128, 128, 'M',       # vgg2
           256, 256, 256, 'MC', # vgg3
           512, 512, 512, 'M',  # vgg4
           512, 512, 512        # vgg5
           ]
    # vgg1～vgg5の畳み込み層までを生成
    for v in cfg:
        # vgg1、vgg2、vgg4のプーリング層
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, # ウィンドウサイズ2×2
                                    stride=2)]     # ストライド2
        # vgg3のプーリング層
        elif v == 'MC':
            # vgg3のプーリングで(75, 75)の特徴量マップを半分のサイズにする際に、
            # ceil_modeをTrueにすることで75/2=37.5を切り上げて38にする
            # この結果、vgg3のプーリング層から出力される特徴量マップのサイズは
            # (38, 38)になる
            layers += [nn.MaxPool2d(kernel_size=2, # ウィンドウサイズ2×2
                                    stride=2,      # ストライド2
                                    ceil_mode=True)]
        # vgg1～vgg5の畳み込み層
        else:
            conv2d = nn.Conv2d(in_channels,  # 入力時のチャネル数
                               v,            # 出力時のチャネル数(フィルター数)
                               kernel_size=3,# フィルターサイズ3×3
                               padding=1)    # パディングのサイズは1
            
            # 畳み込み層に活性化関数ReLUをセットしてlayersに追加
            # inplace=TrueにするとReLUへの入力値は保持されない(メモリ節約)
            layers += [conv2d, nn.ReLU(inplace=True)]
            # チャネル数を出力時のチャネル数(フィルター数)に置き換える
            in_channels = v
            
    # vgg5のプーリング層
    pool5 = nn.MaxPool2d(kernel_size=3, # ウィンドウサイズ3×3
                         stride=1,      # ストライド1
                         padding=1)     # パディングのサイズは1
    # vgg6の畳み込み層1
    conv6 = nn.Conv2d(512,  # 入力時のチャネル数
                      1024, # 出力時のチャネル数(フィルター数)
                      kernel_size=3,# フィルターサイズ3×3
                      padding=6,    # パディングのサイズは6
                      dilation=6)   # 畳み込みのポイント間の間隔を6にする
    # vgg6の畳み込み層2
    conv7 = nn.Conv2d(1024, # 入力時のチャネル数
                      1024, # 出力時のチャネル数(フィルター数)
                      kernel_size=1) # フィルターサイズ1×1
    # vgg5のプーリング層、vgg6の畳み込み層1と畳み込み層2をlayersに追加
    layers += [pool5,
               conv6, nn.ReLU(inplace=True), # 畳み込みの活性化はReLU
               conv7, nn.ReLU(inplace=True)] # 畳み込みの活性化はReLU
    
    # リストlayersをnn.ModuleListに格納してReturnする
    return nn.ModuleList(layers)


'''
2. extrasネットワークを生成する関数の定義
'''
def make_extras():
    '''
    Returns：
      (nn.ModuleList): extrasのモジュール(部品)のリスト
    '''
    layers = []        # モジュールを格納するリスト
    in_channels = 1024 # vggから出力される画像データのチャネル数

    # vggに配置する畳み込み層のフィルター数(チャネル数に相当)
    cfg = [256, 512,  # extras1
           128, 256,  # extras2
           128, 256,  # extras3
           128, 256]  # extras4
    
    # extras1
    # 出力の形状:(バッチサイズ, 512, 10, 10)
    layers += [nn.Conv2d(in_channels,      # 入力時のチャネル数(1024)
                         cfg[0],           # 出力時のチャネル数(256) 
                         kernel_size=(1))] # フィルターサイズ1×1
    layers += [nn.Conv2d(cfg[0],           # 入力時のチャネル数(256)
                         cfg[1],           # 出力時のチャネル数(512) 
                         kernel_size=(3),  # フィルターサイズ3×3
                         stride=2,         # ストライドは2
                         padding=1)]       # パディングのサイズは1
    
    # extras2    
    # 出力の形状:(バッチサイズ, 256, 5, 5)
    layers += [nn.Conv2d(cfg[1],           # 入力時のチャネル数(512)
                         cfg[2],           # 出力時のチャネル数(128) 
                         kernel_size=(1))] # フィルターサイズ1×1
    layers += [nn.Conv2d(cfg[2],           # 入力時のチャネル数(128)
                         cfg[3],           # 出力時のチャネル数(256) 
                         kernel_size=(3),  # フィルターサイズ3×3
                         stride=2,         # ストライドは2
                         padding=1)]       # パディングのサイズは1
    
    # extras3
    # 出力の形状:(バッチサイズ, 256, 3, 3)
    layers += [nn.Conv2d(cfg[3],           # 入力時のチャネル数(256)
                         cfg[4],           # 出力時のチャネル数(128)
                         kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4],           # 入力時のチャネル数(128)
                         cfg[5],           # 出力時のチャネル数(256)
                         kernel_size=(3))] # フィルターサイズ3×3
    
    # extras4
    # 出力の形状:(バッチサイズ, 256, 1, 1)
    layers += [nn.Conv2d(cfg[5],           # 入力時のチャネル数(256)
                         cfg[6],           # 出力時のチャネル数(128)
                         kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6],           # 入力時のチャネル数(128)
                         cfg[7],           # 出力時のチャネル数(256)
                         kernel_size=(3))] # フィルターサイズ3×3

    # リストlayersをnn.ModuleListに格納してReturnする
    return nn.ModuleList(layers)


'''
3. locネットワークを生成する関数の定義
'''
def make_loc(dbox_num=[4, 6, 6, 6, 4, 4]):
    ''' デフォルトボックスのオフセットを出力するlocネットワークを生成
    
    Parameters:
      dbox_num(intのリスト):
          out1～out6それぞれに用意されるデフォルトボックスの数
    Returns：
      (nn.ModuleList): extrasのモジュール(部品)のリスト
    '''
    # ネットワークのモジュールを格納するリスト
    loc_layers = []
    # vgg4の畳み込み層3からの出力にL2Normでの正規化の処理を適用した
    # out1に対する畳み込み層1
    loc_layers += [nn.Conv2d(512,           # 入力時のチャネル数
                             dbox_num[0]*4, # 出力時のチャネル数16
                             kernel_size=3, # フィルターサイズ3×3
                             padding=1)]    # パディングのサイズは1

    # vgg6からの最終出力out2に対する畳み込み層2
    loc_layers += [nn.Conv2d(1024,          # 入力時のチャネル数
                             dbox_num[1]*4, # 出力時のチャネル数24
                             kernel_size=3, # フィルターサイズ3×3
                             padding=1)]    # パディングのサイズは1

    # extrasのext1からの出力out3に対する畳み込み層3
    loc_layers += [nn.Conv2d(512,           # 入力時のチャネル数
                             dbox_num[2]*4, # 出力時のチャネル数24
                             kernel_size=3, # フィルターサイズ3×3
                             padding=1)]    # パディングのサイズは1

    # extrasのext2からの出力out4に対する畳み込み層4
    loc_layers += [nn.Conv2d(256,           # 入力時のチャネル数
                             dbox_num[3]*4, # 出力時のチャネル数24
                             kernel_size=3, # フィルターサイズ3×3
                             padding=1)]    # パディングのサイズは1

    # extrasのext3からの出力out5に対する畳み込み層5
    loc_layers += [nn.Conv2d(256,           # 入力時のチャネル数
                             dbox_num[4]*4, # 出力時のチャネル数16
                             kernel_size=3, # フィルターサイズ3×3
                             padding=1)]    # パディングのサイズは1

    # extrasのext4からの出力out6に対する畳み込み層6
    loc_layers += [nn.Conv2d(256,           # 入力時のチャネル数
                             dbox_num[5]*4, # 出力時のチャネル数16
                             kernel_size=3, # フィルターサイズ3×3
                             padding=1)]    # パディングのサイズは1

    # リストloc_layersをnn.ModuleListに格納してReturnする
    return nn.ModuleList(loc_layers)

'''
4. confネットワークを生成する関数の定義
'''
def make_conf(classes_num=21, dbox_num=[4, 6, 6, 6, 4, 4]):
    ''' デフォルトボックスに対する各クラスの確率を出力するネットワークを生成
    
    Parameters:
      class_num(int): クラスの数
      dbox_num(intのリスト):
        out1～out6それぞれに用意されるデフォルトボックスの数
        
    Returns：
      (nn.ModuleList): extrasのモジュール(部品)のリスト
    '''
    # ネットワークのモジュールを格納するリスト
    conf_layers = []

    # vgg4の畳み込み層3からの出力にL2Normでの正規化の処理を適用した
    # out1に対する畳み込み層1
    conf_layers += [nn.Conv2d(512,          # 入力時のチャネル数
                              dbox_num[0]*classes_num, # 出力時は84
                              kernel_size=3,# フィルターサイズ3×3
                              padding=1)]   # パディングのサイズは1

    # vgg6からの最終出力out2に対する畳み込み層2
    conf_layers += [nn.Conv2d(1024,         # 入力時のチャネル数
                              dbox_num[1]*classes_num, # 出力時は126
                              kernel_size=3,# フィルターサイズ3×3
                              padding=1)]   # パディングのサイズは1

    # extrasのext1からの出力out3に対する畳み込み層3
    conf_layers += [nn.Conv2d(512,          # 入力時のチャネル数
                              dbox_num[2]*classes_num, # 出力時は126
                              kernel_size=3,# フィルターサイズ3×3
                              padding=1)]   # パディングのサイズは1

    # extrasのext2からの出力out4に対する畳み込み層4
    conf_layers += [nn.Conv2d(256,          # 入力時のチャネル数
                              dbox_num[3]*classes_num, # 出力時は126
                              kernel_size=3,# フィルターサイズ3×3
                              padding=1)]   # パディングのサイズは1

    # extrasのext3からの出力out5に対する畳み込み層5
    conf_layers += [nn.Conv2d(256,          # 入力時のチャネル数
                              dbox_num[4]*classes_num, # 出力時は84
                              kernel_size=3,# フィルターサイズ3×3
                              padding=1)]   # パディングのサイズは1

    # extrasのext4からの出力out6に対する畳み込み層6
    conf_layers += [nn.Conv2d(256,          # 入力時のチャネル数
                              dbox_num[5]*classes_num, # 出力時は84
                              kernel_size=3,# フィルターサイズ3×3
                              padding=1)]   # パディングのサイズは1

    # リストconf_layersをnn.ModuleListに格納してReturnする
    return nn.ModuleList(conf_layers)

'''
5. L2Normで正規化する層を生成するクラス
'''
import torch
import torch.nn.init as init

class L2Norm(nn.Module):
    '''vgg4の畳み込み層3の出力out1をL2Normで正規化する層
    
    Attributes:
      weight: L2Norm層のパラメーター(重み)
      scale: 重みの初期値
      eps: L2ノルムの値に加算する極小値
    '''
    def __init__(self, input_channels=512, scale=20):
        '''インスタンス変数の初期化を行う
        
        Parameters:
          input_channels(int):
            入力データ(vgg4の出力)のチャネル数(デフォルト値512)
          scale(int):
            重みweightの初期値として設定する値(デフォルト値20)
        '''
        super(L2Norm, self).__init__() # スーパークラスのコンストラクターを実行
        # レイヤーの重みとして(512,)の1階テンソルを生成
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale      # weightの初期値として設定する値
        self.reset_parameters() # weightの全要素をscaleの値で初期化
        self.eps = 1e-10        # L2ノルムの値に加算する極小値(0.0000000001)

    def reset_parameters(self):
        '''すべての重みをscaleの値で初期化する
        '''
        # torch.nn.init.constant_()で重みテンソルに初期値を設定
        # weightの値をすべてscale（=20）にする
        init.constant_(self.weight, self.scale)

    def forward(self, x):
        '''L2Normにおける順伝播を行う
        
        Parameters:
          x(Tensor):
            vgg4の畳み込み層3からの出力(バッチサイズ, 512, 38, 38)
        
        Returns:
          L2ノルムで正規化した後、scale（=20）の重みを適用した
          (バッチサイズ, 512, 38, 38)の4階テンソル
        '''
        # 各チャネルにおける38×38個の特徴量の個々のセルについて、
        # チャネル方向の2乗和の平方根(L2ノルム)を求め、極小値epsを加算する
        # normの形状は(バッチサイズ, 1, 38, 38)
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        # 各チャネルにおける38×38の個々のセルの値を
        # 同じセルのnormで割って正規化する
        x = torch.div(x, norm)

        # self.weightの1階テンソル(512,)を(バッチサイズ, 512, 38, 38)の
        # 4階テンソルに変形してxと同じ形状にする
        weights = self.weight.unsqueeze(
            0).unsqueeze(2).unsqueeze(3).expand_as(x)
        # 変形後のxに重みを適用する
        out = weights * x

        return out # outの形状は(バッチサイズ, 512, 38, 38)


'''
6. デフォルトボックスを出力するクラス
'''
from itertools import product as product
from math import sqrt as sqrt

class DBox(object):
    '''8732個のDBoxの(x座標, y座標, 幅, 高さ)を生成する
    
    Attributes:
      image_size(int): イメージのサイズ
      feature_maps(list): out1～out6における特徴量マップのサイズのリストを保持
      num_priors(int): feature_mapsの要素数、out1～out6の個数6を保持
      steps(list): DBoxのサイズのリストを保持
      min_sizes(list): 小さい正方形のDBoxのサイズを保持
      max_sizes(list): 大きい正方形のDBoxのサイズを保持
      aspect_ratios(list): 長方形のDBoxのアスペクト比を保持
    '''
    def __init__(self, cfg):
        '''インスタンス変数の初期化を行う
        '''
        super(DBox, self).__init__() # スーパークラスのコンストラクターを実行

        # 画像サイズ(300)を設定
        self.image_size = cfg['input_size']
        # out1～out6における特徴量マップのサイズ[38, 19, …]を設定
        self.feature_maps = cfg['feature_maps']
        # out1～out6の個数=6を設定
        self.num_priors = len(cfg["feature_maps"])
        # DBoxのサイズ[8, 16, 32, …]を設定 
        self.steps = cfg['steps']
        # 小さい正方形のDBoxのサイズ[30, 60, 111, …] を設定
        self.min_sizes = cfg['min_sizes']
        # 大きい正方形のDBoxのサイズ[60, 111, 162, …] を設定
        self.max_sizes = cfg['max_sizes']
        # 長方形のDBoxのアスペクト比[[2],[2,3],[2,3], ...]を設定
        self.aspect_ratios = cfg['aspect_ratios']

    def make_dbox_list(self):
        '''DBoxを作成する
        
        Returns:
          (Tensor)DBoxの[cx, cy, width, height]を格納した(8732, 4)の形状のテンソル
        '''
        mean = []
        # out1～out6における特徴量マップの数(6)だけ繰り返す
        # 特徴量マップのサイズのリストからインデックスをk、サイズをfに取り出す
        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        # k: 0,1,2,3,4,5
        # f: 38, 19, 10, 5, 3, 1
        for k, f in enumerate(self.feature_maps):
            # fまでの数をrepeat=2を指定して2つのリストにして組み合わせ(直積)を作る
            # f=38の場合
            # i: 0,0,0,0,... の38個の0に対して
            # j: 0,1,2,3, ..., 37を組み合わせる
            # (i,j)は(0,0)(0,1)...(0,37)~(37,0)...(37,37)
            for i, j in product(range(f), repeat=2):  
                # 特徴量の画像サイズをDBoxのサイズsteps[k]で割る(kはインデックス)
                # 300 / 'steps': [8, 16, 32, 64, 100, 300]
                f_k = self.image_size / self.steps[k]

                # 特徴量ごとのDBoxの中心のx座標、y座標を求める
                # (0～1の範囲に規格化)
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 小さい正方形のDBoxのサイズmin_sizes[k](kはインデックス)を
                # 画像のサイズで割る
                # 'min_sizes': [30, 60, 111, 162, 213, 264] / 300
                s_k = self.min_sizes[k]/self.image_size
                # 小さい正方形のDBoxの[cx,cy, width, height]をリストに追加
                mean += [cx, cy, s_k, s_k]

                # 大きい正方形のDBoxのサイズmin_sizes[k](kはインデックス)を
                # 画像のサイズで割る
                # max_sizes': [45, 99, 153, 207, 261, 315] / 300
                # さらにs_kを掛けて平方根を求める
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                # 大きい正方形のDBoxの[cx,cy, width, height]をリストに追加
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 長方形のDBoxの[cx,cy, width, height]をリストに追加
                for ar in self.aspect_ratios[k]:
                    # widthはs_kにアスペクト比の平方根を掛けたもの
                    # heightはs_kをアスペクト比と平方根で割ったもの
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    # widthはs_kをアスペクト比と平方根で割ったもの
                    # heightはs_kにアスペクト比の平方根を掛けたもの
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # DBoxの[cx,cy, width, height]のリストを(8732, 4)の2階テンソルに変換
        output = torch.Tensor(mean).view(-1, 4)
        # DBoxの大きさが1を超えている場合は1にする
        output.clamp_(max=1, min=0)
        
        # DBoxの[cx,cy, width, height]を格納した2階テンソルを返す
        return output


'''
7. デフォルトボックスをバウンディングボックスに変換する関数
'''
def decode(loc, dbox_list):
    '''
    locネットワークが出力するオフセット情報を使用してDBoxをBBoxに変換する

    Parameters:
      loc(Tensor):
        locが出力する(8732,4)の形状のテンソル
        8,732個のDBoxのオフセット情報(Δcx, Δcy, Δwidth, Δheight)
      dbox_list(Tensor):
        DBoxの情報(cx, cy, width, height)を格納した(8732,4)のテンソル
        
    Returns(Tensor):
      BBoxの情報(xmin, ymin, xmax, ymax)を格納したテンソル(8732, 4)
    '''
    # DBoxにlocのオフセットを適用してBBoxの(cx, cy, width, height)を求める
    # 変数boxesの形状は(8732, 4)
    boxes = torch.cat((
        # cx = cx_d + 0.1Δcx ･ w_d
        # cy = cy_d + 0.1Δcy ･ h_d
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        # w = w_d ･ exp(0.2Δw)
        # h = h_d ･ exp(0.2Δh)
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)
        ),dim=1)

    # BBoxの情報(cx, cy, width, height)を(xmin, ymin, xmax, ymax)に変換
    boxes[:, :2] -= boxes[:, 2:] / 2  # (cx, cy)を(xmin,ymin)にする
    boxes[:, 2:] += boxes[:, :2]      # (width, height)を(xmax,ymax)にする

    return boxes


'''
8. Non-Maximum Suppressionを行う関数
'''
def nonmaximum_suppress(
        boxes, scores, overlap=0.5, top_k=200): ###### nm_suppressionを変更、オリジナルの0.5に戻す######
    '''1つの物体に対して1つのBBoxだけを残す
    
    画像分類のクラスごとにNon-Maximum Suppressionを実施
    クラス単位で抽出された確信度0.01以上のboxesから同一の物体に対する被り度
    （IoU値）が大きいBBoxを集めて、その中で最大の確信度を持つBBoxだけを取り出す

    Parameters:
      boxes(Tensor):
        1クラスあたり8,732個のBBoxのうち、確信度0.01を超えたDBoxの座標情報
        テンソルの形状は(1クラスにつき確信度0.01を超えたDBoxの数, 4)
      scores(Tensor):
          confネットワークの出力(DBoxの各クラスの確信度)からクラスごとに
          確信度の閾値0.01を超えるBBoxの確信度だけを抜き出したもの
          テンソルの形状は(1クラスにつき確信度0.01を超えたBBoxの数, )
      overlap(float):
        被り度合い（IoU値）の基準にする値
        overlapが0.5以上である場合に、それらのバウンディングボックスは
        同じ物体に対するバウンディングボックスと判断する
      top_k(int)
　      scoresから確信度が高い順にサンプルを取り出す際の、取り出すサンプルの数

    Returns:
      keep(Tensor): 画像中に存在するBBoxのインデックスが格納される
      count(int):  画像中に存在するBBoxの数が格納される
    '''
    # NMSを通過したBBoxの数を保持する変数の初期化
    count = 0
    # scoresと同じ形状の0で初期化したテンソルを生成
    # keepの形状は(1クラスにつき確信度0.01を超えたBBoxの数,)
    keep = scores.new(scores.size(0)).zero_().long()

    # 各BBoxの面積areaを計算
    # areaの形状は(確信度0.01を超えるBBoxの数,)
    x1 = boxes[:, 0] # x軸最小値
    y1 = boxes[:, 1] # y軸最小値
    x2 = boxes[:, 2] # x軸最大値
    y2 = boxes[:, 3] # y軸最大値
    area = torch.mul(x2 - x1, y2 - y1) # torch.mulで底辺×高さを求める

    # boxesのコピーをBBox情報の要素の数だけ作成
    # BBoxの被り度(IoU)の計算の際に使用する
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # socreを昇順(確信度が低い方から)に並び変える
    v, idx = scores.sort(0) # idxに元の要素のインデックスのリストを格納

    # idxの上位top_k個（200個）のBBoxのインデックスを取り出す
    # 200個存在しない場合もある
    idx = idx[-top_k:]

    # idx(初期の要素数top_k個（200個）)の要素数が0でない限りループ
    while idx.numel() > 0:
        i = idx[-1]  # 最大の確信度(conf値)のインデックスを取得
        
        # keepの形状は(1クラスにつき確信度0.01を超えたBBoxの数,)
        # keepのインデックスcountの位置に最大確信度(conf値)のインデックス値を格納
        # このインデックスのBBoxと被りが大きいBBoxを以下の処理で取り除く
        keep[count] = i
        # keepのインデックスを1増やす
        count += 1

        # idxの要素数を取得し、1(最後のBBox)であればループを抜ける
        if idx.size(0) == 1:
            break

        ### Non-Maximum Suppressionの処理を開始 ###
        # 昇順に並んでいるscoresのインデックスの末尾を除外する
        idx = idx[:-1]

        # idxの昇順スコアのインデックス値を使ってBBoxの座標情報xmin, ymin, xmax, ymaxの
        # 情報を抽出してtmp_x1、tmp_y1、tmp_x2、tmp_y2に格納
        # index_select(入力Tensor,
        #              対象の次元,
        #              抽出する要素のインデックス,
        #              out=出力Tensor名)
        torch.index_select(x1, 0, idx, out=tmp_x1) # 昇順スコアに対応するxminの並び
        torch.index_select(y1, 0, idx, out=tmp_y1) # 昇順スコアに対応するyminの並び
        torch.index_select(x2, 0, idx, out=tmp_x2) # 昇順スコアに対応するxmaxの並び
        torch.index_select(y2, 0, idx, out=tmp_y2) # 昇順スコアに対応するymaxの並び

        # idxに残っているBBoxのxmin, ymin, xmax, ymaxの下限値を
        # それぞれインデックスi(確信度最上位のBBox）の値までに切り詰める
        # torch.clamp(入力Tensor,
        #             min=切り詰める下限値,
        #             max=切り詰める上限値,
        #             out=出力Tensor名)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i]) # xminの下限値を切り詰める
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i]) # yminの下限値を切り詰める
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i]) # xmaxの下限値を切り詰める
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i]) # xmaxの下限値を切り詰める

        # tmp_wとtmp_hのテンソルの形状をそれぞれtmp_x2、tmp_y2と同じ形状にする
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # tmp_x1, tmp_y1, tmp_x2, tmp_y2を使って重なる部分の幅と高さを求め
        # tmp_wとtmp_hに代入する
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # 幅や高さが負の値になっていたら0にする
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # intersect(交差)部分の面積(A ∩ B)を求める
        inter = tmp_w*tmp_h

        # IoU = intersect部分 / (area(a) + area(b) - intersect部分)の計算
        # areaからidxに残っているすべてのBBoxの面積を取得
        rem_areas = torch.index_select(
            area, # 確信度0.01以上のすべてのBBoxの面積
            0,    # 処理対象の次元
            idx)  # 確信度上位200から現存するBBoxのインデックス値の並び
        # (BBoxの元の面積 - 交差部分の面積)+基準となるBBox(確信度最上位）の面積
        union = (rem_areas - inter) + area[i]  # A∪Bの面積
        IoU = inter/union # idxに残っているすべてのBBoxのIoUを求める

        # idxに残っているBBoxのうちIoUがoverlapより小さいものだけを残す
        # 同じ物体を囲むその他のBBoxがすべて取り除かれる
        idx = idx[IoU.le(overlap)]  # le()はoverlap以下の要素だけを残す

    # idxのBBoxが１個になりwhileループを抜けたら
    # 検出されたBBoxの数とBBoxを参照するためのインデックス値を返して終了
    return keep, count


'''
9. SSDの推論時にconfとlocの出力から真のBBoxを抽出するDetectクラス
'''
from torch.autograd import Function

class Detect(Function):
    '''推論時の順伝播処理のみを実装
    
    Attributes:
      softmax: torch.nn.Softmax
      conf_thresh: BBoxを抽出する際の閾値
      top_k: Non-Maximum Suppressionを実施するBBoxの数
      nms_thresh: 被り度合い（IoU値）の閾値
    '''
    @staticmethod
    def forward(ctx, loc_data, conf_data, dbox_list):
        '''loc、confの出力を順伝播しBBoxの情報と正解ラベルを出力する

        Parameters:
          loc_data(Tensor):
            locネットワークが出力するDBoxのオフセット情報
            (バッチサイズ, 8732, 4[Δcx,Δcy,Δw,Δh])
          conf_data(Tensor):
            confネットワークが出力するDBoxのクラスラベル(21個)ごとのconf値
            (バッチサイズ, 8732, 21)
          dbox_list(Tensor):
            DBoxの情報(8732, 4[cx, cy, width, height])

        Returns:
           output(Tensor): (バッチサイズ, 21, 200, 5)
              内訳: (バッチデータのインデックス,
                     クラスのインデックス,
                     Top200のBBoxのインデックス,
                     (BBoxの確信度, xmin, ymin, width, height))
        '''
        # confネットワークの出力を正規化するためのソフトマックス関数
        ctx.softmax = nn.Softmax(dim=-1)
        # BBoxを抽出する際の閾値(確信度が0.01より高いものを抽出)
        ctx.conf_thresh = 0.01
        # Non-Maximum Suppressionを実施するBBoxの数(確信度上位200個)
        ctx.top_k = 200
        # 被り度合い（IoU値）の閾値
        # 0.45より高ければ同一の物体へのBBoxと判定する
        ctx.nms_thresh = 0.45

        # ミニバッチのサイズを取得
        batch_num = loc_data.size(0)
        # クラス数(ラベル数)の21を取得
        classes_num = conf_data.size(2)

        # confが出力するDBoxのクラスラベルごとのconf値
        # (バッチサイズ, 8732, 21)にソフトマックス関数を適用
        # DBoxごとに全ラベルそれぞれのconf値が確率に置き換えられる
        conf_data = ctx.softmax(conf_data)

        # ソフトマックス関数を適用のconf_dataの形状(バッチサイズ, 8732, 21)を
        # (バッチサイズ, 21, 8732)に変更
        conf_preds = conf_data.transpose(2, 1)

        # 出力のoutput用のテンソルを用意
        # テンソルの形状は(バッチサイズ, 21, 200, 5)
        output = torch.zeros(batch_num, classes_num, ctx.top_k, 5)

        # バッチデータごとにループ
        for i in range(batch_num):
            # locネットワークが出力するDBoxオフセット情報
            # (バッチサイズ, 8732, 4) からbatch_numのi番目を取り出し、
            # オフセット値をBBox座標(xmin, ymin, xmax, ymax)に変換
            # decoded_boxesの形状は(8732, 4)
            decoded_boxes = decode(loc_data[i], dbox_list)

            # confネットワークが出力する確信度(batch_num, 21, 8732)の
            # batch_numのi番目のコピーを作成
            # conf_scoresの形状はクラスごとのDBox確信度(21, 8732)
            conf_scores = conf_preds[i].clone()

            # クラスのラベル単位でループ（背景クラスは除外して20回繰り返す）
            for cl in range(1, classes_num):
                # conf_scoresのインデックスcl(ラベルを示す)における8732個の
                # 確信度から閾値(0.01)を超えるものを取り出すためのビットマスク
                # (0と1の並び)を生成
                #
                # torch.gt（input, other)はotherを超えるinput要素を
                # True(1)、それ以外をFalse(0)にして返す
                #
                # c_maskの形状は(TrueまたはFalseが8732個)
                c_mask = conf_scores[cl].gt(ctx.conf_thresh)

                # conf_scoresのインデックスclを抽出し、
                # c_maskのTrueに対応する0.01超えの確信度を取得
                # scoresの形状は(閾値を超えた確信度の数,)
                scores = conf_scores[cl][c_mask]

                # scoresの要素が0(閾値を超える確信度が存在しない)場合は
                # 処理を中断してループの先頭に戻る
                if scores.nelement() == 0: # nelementで要素数の合計を取得
                    continue

                # c_maskの形状(8732,)をdecoded_boxesの形状(8732,4)に変形する
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)

                # l_maskをdecoded_boxesに適応してBBox座標(8732, 4)から
                # 閾値0.01越えのBBox座標を抽出
                # decoded_boxes[l_mask]で1階テンソルになるので、
                # view(-1,4)で2階テンソル(確信度0.01を超えるBBox数, 4)に
                boxes = decoded_boxes[l_mask].view(-1, 4)

                # Non-Maximum Suppressionを実施して1つの物体につき、
                # 確信度最上位のBBoxを1個取得する
                # ids: 検出されたBBoxを参照するためのインデックス値
                # count: 検出されたBBoxの数
                ids, count = nonmaximum_suppress(
                    boxes,  # ラベルiにおける確信度0.01を超えるBBoxの座標情報
                    scores, # ラベルiにおける0.01を超える確信度の値
                    ctx.nms_thresh, # 被り度合い（IoU値）の閾値(0.45)
                    ctx.top_k)      # NMSを実施するBBoxの数(確信度上位200個)

                # outputにNon-Maximum Suppressionの結果を格納
                # outputの1次元: バッチデータのインデックス(要素数はバッチの数)
                # outputの2次元: クラスのラベルのインデックス(要素数21)
                # outputの3次元: NMSを適用するBBoxのインデックス(要素数200)
                # outputの4次元: BBoxの確信度, xmin, ymin, width, height(5)
                output[i, cl, :count] = torch.cat(
                    # scoresからidxのcountまでの確信度を2階テンソルで取得
                    # boxesからidxのcountまでのBBox座標(2階テンソル)を取得
                    # 取得した確信度とBBox座標を2階テンソルの形状で連結
                    (scores[ids[:count]].unsqueeze(1),
                     boxes[ids[:count]]), 1)

        return output  # outputの形状は(バッチサイズ, 21, 200, 5)

'''
10. SSDクラスを作成する
'''
import torch.nn.functional as F

class SSD(nn.Module):
    '''SSDモデルを生成するクラス
    
    Attributes:
      phase(str): 'train'または'test'
      classes_num(int): クラスの数
      vgg(object): vggネットワーク
      extras(object): extrasネットワーク
      L2Norm(object): L2norm層
      loc(object): locネットワーク
      conf(object): confネットワーク
      dbox_list(Tensor):
        DBoxの[cx, cy, width, height]を格納した(8732, 4)の形状のテンソル
      detect(object):
        Detectクラスのforward()を実行する関数オブジェクト
    '''
    def __init__(self, phase, cfg):
        '''インスタンス変数の初期化を行う
        '''
        super(SSD, self).__init__()

        self.phase = phase  # 動作モードの'train'または'test'を取得
        self.classes_num = cfg["classes_num"]  # クラスの数(21)を取得

        # SSDのネットワークを生成
        self.vgg = make_vgg()       # vggネットワーク
        self.extras = make_extras() # extrasネットワーク
        self.L2Norm = L2Norm()      # L2Norm層
        # locネットワーク
        self.loc = make_loc(
            cfg["dbox_num"] # out1～out6にそれぞれに用意するDBoxの数
            )
        # confネットワーク
        self.conf = make_conf(
            cfg["classes_num"], # クラスの数
            cfg["dbox_num"]     # out1～out6にそれぞれに用意するDBoxの数
            )

        # DBoxの[cx, cy, width, height]を格納したテンソル(8732, 4)を取得
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 推論モード(test)で使用するDetectクラス
        if phase == 'test':
            # Detectのforward()を実行する関数オブジェクトを取得
            self.detect = Detect.apply

    def forward(self, x):
        '''SSDモデルの順伝播を行う
        
        Parameters:
          x: 300×300の画像を格納した4階テンソル
             (バッチサイズ, 3, 300, 300)
        
        Returns:
          推論モードの場合：
            1枚の画像の各物体に対するBBoxの情報が格納される
            (バッチサイズ, 21(クラス), 200(Top200のBBox), 5)
            最後の次元の5は[BBoxの確信度, xmin, ymin, width, height]
                         
          学習モードの場合:
            以下のテンソルを格納したタプル(loc, conf, dbox_list)
            ・locの出力(バッチサイズ, 8732, 4[Δcx, Δcy, Δw, Δh])
            ・confの出力(バッチサイズ, 8732, 21)
            ・DBoxの情報(8732, 4[cx, cy, width, height])
        '''
        out_list = list() # locとconfに入力するout1～6を格納するリスト
        loc = list()      # locネットワークの出力を格納するリスト
        conf = list()     # confネットワークの出力を格納するするリスト

        # out1を取得
        # vgg1からvgg4の畳み込み層3まで順伝播する
        # (0層～22層: 活性化関数も層としてカウント)
        for k in range(23):
            x = self.vgg[k](x)
        # vgg4の畳み込み層3の出力をL2Normで正規化する
        out1 = self.L2Norm(x)
        # out1をout_listに追加
        out_list.append(out1)
        
        # out2を取得
        # vgg4のプーリング層からvgg6まで順伝播する
        # (23層～35層: 活性化関数も層としてカウント)
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        # out2をout_listに追加
        out_list.append(x)
        
        # out3～6を取得
        # extras1～extras4まで8層の畳み込みすべてにReLUを適用しながら順伝播する
        for k, v in enumerate(self.extras):
            # 畳み込み層の出力にReLUを適用する
            x = F.relu(v(x), inplace=True)
            # extras1～extras4の各出力は層のインデックスが奇数になる
            if k % 2 == 1:
                # extras1～extras4の各出力、out3～out6を順次out_listに追加
                out_list.append(x)

        # out1～6に、それぞれ対応する畳み込みを1回ずつ適用する
        # zip()でout、loc、conf（すべて要素数6）を取り出して
        # loc1～6,conf1～6までの入出力を行う
        for (x, l, c) in zip(out_list,   # out1～out6(要素数6)
                             self.loc,   # locの畳み込みは6層
                             self.conf): # confの畳み込みは6層
        
            # locの畳み込み層1～6にそれぞれout1～6を入力して出力の形状を
            # (バッチサイズ, オフセット値4*DBoxの種類, 特徴量(h),特徴量(w))
            # ↓
            # (バッチサイズ, 特徴量(h), 特徴量(w), オフセット値4*DBoxの種類)
            # のように変換し、view()関数を適用できるように
            # torch.contiguous()でメモリ上に要素を連続的に配置し直す
            #
            # loc1:(bs, 38, 38, 16) 最後の次元は4個のオフセット値×DBoxの種類
            # loc2:(bs, 19, 19, 24)
            # loc3:(bs, 10, 10, 24)
            # loc4:(bs, 5, 5, 24)
            # loc5:(bs, 3, 3, 16)
            # loc6:(bs, 1, 1, 16)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            
            # confの畳み込み層1～6にそれぞれout1～6を入力して出力の形状を
            # (バッチサイズ, クラス数21*DBoxの種類, 特徴量(h),特徴量(w))
            # ↓
            # (バッチサイズ, 特徴量(h), 特徴量(w), クラス数21*DBoxの種類)
            # のように変換し、view()関数を適用できるように
            # torch.contiguous()でメモリ上に要素を連続的に配置し直す
            #
            # conf1:(bs, 38, 38, 84) 最後の次元はクラスのラベル21×DBoxの種類
            # conf2:(bs, 19, 19, 126)
            # conf3:(bs, 10, 10, 126)
            # conf4:(bs, 5, 5, 126)
            # conf5:(bs, 3, 3, 84)
            # conf6:(bs, 1, 1, 84)
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # locとconfのバッチ以下の形状をフラットにする
        # locの形状は(バッチサイズ, 34928)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # confの形状は(バッチサイズ, 183372)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # DBoxごとに4個のオフセット値が格納されるように
        # (バッチサイズ, 8732, 4)の形状にする
        loc = loc.view(loc.size(0), -1, 4)
        # DBoxごとに21クラスの確信度が格納されるように
        # (バッチサイズ, 8732, 21)の形状にする
        conf = conf.view(conf.size(0), -1, self.classes_num)

        # locの出力(バッチサイズ, 8732, 4)
        # confの出力(バッチサイズ, 8732, 21)
        # DBoxのオフセット情報(DBoxの数,4[Δcx,Δcy,Δw,Δh])
        # これらをSSDモデルの出力としてタプルにまとめる
        output = (loc, conf, self.dbox_list)
        
        # 推論モードの場合
        if self.phase == 'test':
            # Detectクラスのforwardに
            # ・locの出力(バッチサイズ, 8732, 4[Δcx,Δcy,Δw,Δh])
            # ・confの出力(バッチサイズ, 8732, 21)
            # ・DBoxの情報(DBoxの数,4[cx, cy, width, height])
            # を渡してNMSによるBBoxの絞り込みを実行
            #
            # 戻り値として1枚の画像の各物体に対する1個のBBoxの情報が
            # (バッチサイズ, 21(クラス), 200(Top200のBBox), 5)として返される
            # 最後の次元の5は[BBoxの確信度, xmin, ymin, width, height]
            return self.detect(output[0], output[1], output[2])

        else:
            # 学習モードの場合はoutput(loc, conf, dbox_list)を返す
            return output


'''
11. SSDの損失関数のクラを作成する
'''
# 関数matchを記述したmatch.pyをインポート
from match import match

class MultiBoxLoss(nn.Module):
    '''SSDの損失関数のクラス
    
    Attributes:
      jaccard_thresh(float): 背景のDBoxに分類するときのjaccard係数の閾値(0.5)
      negpos_ratio(int): 背景のDBoxを絞り込むときの割合(Positive DBoxの「3」倍)
      device(torch.device): 使用するデバイス(CPUまたはGPU)
    '''
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        # 関数match()に渡すのjaccard係数の閾値(0.5)をセット
        self.jaccard_thresh = jaccard_thresh
        # Negative DBoxを絞り込むときの割合(Positive DBoxの「3」倍)をセット
        self.negpos_ratio = neg_pos
        # 使用するデバイスの情報(CPUまたはGPU)をセット
        self.device = device

    def forward(self, predictions, targets):
        '''損失関数を適用してlocとconfの出力の誤差(損失)を求める

        Parameters:
          predictions(tuple):
            SSDの訓練時の出力(loc, conf, DBox)
            ・locの出力(バッチサイズ, 8732, 4[Δcx, Δcy, Δw, Δh])
            ・confの出力(バッチサイズ, 8732, 21)
            ・DBoxの情報(8732, 4[cx, cy, width, height])
          targets(Tensor):
            正解BBoxのアノテーション情報
            (バッチサイズ, 物体数, 5[xmin, ymin, xmax, ymax, label_index])

        Returns:
          loss_l(Tensor):
            ミニバッチにおける「Positive DBoxのオフセット情報の損失平均」
          loss_c(Tensor):
            ミニバッチにおける「num_pos+num_negの確信度の損失平均」
        '''
        # ※loc_data:
        #    オフセットの予測値(バッチサイズ, 8732, 4[Δcx, Δcy, Δw, Δh])
        # ※conf_data:
        #    21クラスの予測確信度(バッチサイズ, 8732, 21)
        # ※dbox_list:
        #    DBoxの情報(8732, 4[cx, cy, width, height])
        loc_data, conf_data, dbox_list = predictions

        # ※num_batch: ミニバッチのサイズ
        num_batch = loc_data.size(0)
        # ※num_dbox: DBoxの数(8732)
        num_dbox = loc_data.size(1)
        # ※num_classes: クラス数(21)
        num_classes = conf_data.size(2)

        # ※conf_t_label:
        #     「正解ラベルの教師データ」(8732個のDBox)を格納するためのテンソル
        #     ミニバッチのすべてのデータを格納
        #    (バッチサイズ, 8732[正解ラベル])
        #
        conf_t_label = torch.LongTensor(num_batch, # 1次元はバッチサイズ
                                        num_dbox   # 2次元は8732
                                        ).to(self.device)
        
        # ※loc_t:
        #   「オフセット値の教師データ」(8732個のDBox)を格納するためのテンソル
        #   ミニバッチのすべてのデータを格納
        #   (バッチサイズ, 8732, 4[Δcx, Δcy, Δw, Δh])
        #
        loc_t = torch.Tensor(num_batch, # 1次元はバッチサイズ
                             num_dbox,  # 2次元は8732
                             4          # 3次元は4(オフセット情報)
                             ).to(self.device)

        # ----------------------------------------
        # ※loc_t: 1画像あたり8732個のDBoxの教師データ(オフセット値)を登録
        # ※conf_t_label: 1画像あたり8732個の正解ラベルを登録
        # ----------------------------------------
        # ミニバッチの画像1枚ずつ処理
        for idx in range(num_batch):
            
            # truths: アノテーション情報から取得した正解BBoxの座標(BBoxの数,4)
            truths = targets[idx][:, :-1].to(self.device) 
            # labels: アノテーション情報から取得した正解ラベル(BBoxの数,)
            labels = targets[idx][:, -1].to(self.device)
            # dbox: DBoxの情報のコピー(8732, 4[cx, cy, width, height])
            dbox = dbox_list.to(self.device)
            # variance: DBoxのオフセット値を求めるときの係数のリスト
            variance = [0.1, 0.2]
            
            # 関数matchを実行して教師データloc_t、conf_t_labelの内容を更新
            #
            # loc_t: 「教師データ」オフセット値
            #   (バッチサイズ, 8732,4[Δcx,Δcy,Δw,Δh])のidxの位置に
            #   現在の画像の教師データ(8732,4[Δcx,Δcy,Δw,Δh]))が追加される
            #
            # conf_t_label: 「教師データ」正解ラベル
            #   (バッチサイズ, 8732[正解ラベル])のidxの位置に
            #   現在の画像の教師データ(8732[正解ラベル],)が追加される
            #   このときIoU値が0.5より小さいDBoxのラベルは背景(0)に振り分けられる
            match(
                self.jaccard_thresh, # 背景に振り分けるときのIoU値の閾値(0.5)
                truths, # 現在の画像中の正解BBoxの座標情報(BBoxの数,4)
                dbox,   # DBoxの情報(8732, 4[cx,cy,width,height])
                variance, # オフセット値計算時の係数リスト[0.1, 0.2]
                labels, # 正解ラベル[物体1のラベル, 物体2のラベル, …]
                loc_t,  # ミニバッチのオフセット値の教師データを格納するテンソル
                conf_t_label, # ミニバッチの正解ラベルを格納するテンソル
                idx) # バッチの何番目の画像かを示すインデックス

        # -------------------------------------------------------
        # 物体を検出したPositive DBoxのオフセット情報の損失「loss_l」を計算
        # -------------------------------------------------------
        
        # ※pos_mask:
        #     Positive DBoxを取り出すためのTrue(1)、False(0)のマスク
        #     (バッチサイズ, 8732)
        #
        # 正解ラベルの並びconf_t_label(バッチサイズ,8732[正解ラベル])を利用して
        # 背景の0をFalse(0)、それ以外をTrue(1)にしたテンソルを作成
        pos_mask = conf_t_label > 0

        # ※pos_idx:
        #     pos_mask(バッチサイズ,8732)をオフセット抽出用として
        #     (バッチサイズ,8732,4)に拡張
        #
        # locが出力する「オフセットの予測値」(バッチサイズ, 8732, 4[Δcx, Δcy, Δw, Δh])
        # と同じ形状にする
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # ※loc_p:
        #   画像1枚におけるPositive DBoxの「オフセットの予測値」を抽出
        #   (バッチサイズ,
        #    Positiveの数(8732-Negative),
        #    4[Δcx,Δcy,Δw,Δh]) ←予測値
        #
        # locの出力(バッチサイズ, 8732, 4[Δcx, Δcy, Δw, Δh])から
        # pos_idx(バッチサイズ,8732,4)でマスクしてPositiveのオフセット値を抽出
        # view(-1, 4)でDBoxの数(8732)がPositiveの数に調整される
        loc_p = loc_data[pos_idx].view(-1, 4)
        
        # ※loc_t:
        #     画像1枚におけるPositive DBoxの「教師データ(オフセット値)」
        #     (バッチサイズ,
        #      Positiveの数(8732-Negative),
        #      4[Δcx,Δcy,Δw,Δh]) ←正解のオフセット値
        # 
        # 「教師データ(オフセット値)loc_t(バッチサイズ, 8732,4[Δcx,Δcy,Δw,Δh])から
        # pos_idx(バッチサイズ,8732,4)でマスクしてPositiveのオフセット値を抽出
        # view(-1, 4)でDBoxの数(8732)がPositiveの数に調整される
        loc_t = loc_t[pos_idx].view(-1, 4)

        # ※loss_l(要素数1のテンソル):
        #     オフセット予測値の損失合計
        #
        # ミニバッチのすべての画像についてPositive DBoxの
        # 予測オフセット値の損失を求め、合計する
        loss_l = F.smooth_l1_loss(
            loc_p, # locが出力したPositive DBoxのオフセットの予測値
            loc_t, # Positive DBoxのオフセット値の教師データ
            reduction='sum') # 出力される損失値を合算する

        # ----------------------------------------
        # ※loss_c:
        #     21クラスの予測値(確信度)の損失を求める
        # ----------------------------------------
        
        # ※batch_conf:
        #     8732個のDBoxの21クラスに対する予測値(確信度): バッチデータすべて
        #     (バッチサイズ×8732, 21[確信度])
        # 
        # conf_data(バッチサイズ,8732,21[確信度])から確信度を抽出
        batch_conf = conf_data.view(
            -1,          # 0の次元はバッチサイズ×8732に調整
            num_classes) # 1の次元は21

        # ※loss_c
        #     正解ラベルに対する予測値(確信度)の損失(クロスエントロピー誤差)
        #     (バッチサイズ×8732[損失],)
        #
        # One-hot表現のbatch_confに対し、正解ラベルconf_t_labelは
        # 内部でOne-hot化されて処理 → 正解ラベルごとに損失を出力
        loss_c = F.cross_entropy(
            # 21クラスの確信度(バッチサイズ×8732, 21)
            batch_conf,
            # (バッチサイズ, 8732[正解ラベル])を(バッチサイズ×8732,)にする
            conf_t_label.view(-1),
            # 正解ラベルに対する損失(バッチサイズ×8732,)をそのまま出力
            reduction='none')

        # -----------------
        # ※neg_mask:
        #     Hard Negative MiningのためのNegative DBox抽出用のマスクを作成
        #
        #     21クラスの予測値(確信度)の損失が上位のDBoxを抽出する際に
        #     (Positiveの数×3)のNegative DBoxを除くためのTrue/Falseのマスク
        # -----------------

        # ※num_pos: 画像1枚中のPositive DBoxの数(バッチサイズ, Positiveの数)
        #            
        # pos_mask(バッチサイズ,8732)のTrueを数値の1にして
        # sum(1, keepdim=True)で合計する
        num_pos = pos_mask.long().sum(1, keepdim=True)
        
        # ※loss_c: (バッチサイズ, 8732[確信度の損失])
        #
        # loss_c(バッチサイズ×8732[正解ラベルに対する損失],)に
        # view(num_batch, -1)を適用して(バッチサイズ, 8732)にする
        loss_c = loss_c.view(num_batch, -1)
        
        # ※loss_c: (バッチサイズ, 8732[確信度の損失(Positiveのみ0])
        #
        # Positiveを抽出するマスクpos_mask(バッチサイズ,8732)を利用して
        # Positive Boxの正解ラベルに対する損失をすべて0にする
        loss_c[pos_mask] = 0

        # ※loss_idx:
        #     8732個の損失を降順で並べたときのDBoxのインデックス
        #     (バッチサイズ, 8732[DBoxのインデックス])
        #
        # loss_c(バッチサイズ, 8732[Positiveの損失のみ0に変更])から
        # 損失の値(1の次元)を降順で並べ、元の位置(loss_c)のインデックスを取得
        _, loss_idx = loss_c.sort(1, descending=True)
        
        # ※idx_rank:
        #     loss_cのDBoxごとの損失の大きさの順位
        #     (バッチサイズ, 8732[順位(0～)])
        #
        # loss_idxをインデックス値の昇順で並べ替えることで元のloss_cの並びにして
        # loss_idxのインデックス値(損失の順位を示す)を取得
        # 
        # idx_rank[0]に格納された値はloss_c[0]のDBoxの損失の大きさの順位
        _, idx_rank = loss_idx.sort(1)

        # ※num_neg: (バッチサイズ[Negative DBoxの上限数])
        #
        # num_pos: 画像1枚中のPositive DBoxの数(バッチサイズ, Positiveの数)
        # 1画像あたり抽出するNegative DBox(背景のDBox)の数＝Positiveの数×3
        num_neg = torch.clamp(
            num_pos*self.negpos_ratio,
            max=num_dbox) # 上限値はnum_dbox: DBoxの数(8732)


        # ※neg_mask:
        #     Negative DBoxの損失上位のDBoxをPositive×3の数だけ抽出するためのマスク
        #     (バッチサイズ, 8732[True or False])
        #
        # 1.num_neg(バッチサイズ[Negative DBoxの上限数])の形状を
        # バッチデータごとに並べて
        # (バッチサイズ,8732[Negative DBoxの上限数])にする
        #
        # 2.idx_rank(バッチサイズ, 8732[損失値の順位(0～)])と
        # num_neg(バッチサイズ,8732[Negative DBoxの上限数])を比較
        #
        # 3.Negativeの上限数までの損失上位のNegative DBoxをTrue、
        # 上限数を超える損失下位のNegative DBoxをFalseにする
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # ---------------------------------------------------------
        # ※pos_idx_mask: Positive DBoxの予測確信度を取り出すマスクを作成
        #
        # ※neg_idx_mask: Negative DBoxの損失上位の予測確信度を
        #                 取り出すマスクを作成(Positive×3の数)
        # --------------------------------------------------------

        # ※pos_idx_mask:
        #     Positive DBoxの予測確信度を取り出すマスク
        #     pos_maskを21クラス対応に拡張(バッチサイズ, 8732, 21)
        #
        # pos_mask: Positive DBoxを取り出すためのマスク
        # (バッチサイズ, 8732)
        # ↓
        # (バッチサイズ, 8732, 1) 2の次元(3)にサイズ1の次元を挿入
        # ↓
        # (バッチサイズ, 8732, 21) conf_data: 21クラスの予測確信度
        #                      (バッチサイズ, 8732, 21)と同じ形状に拡張
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        
        # ※neg_idx_mask:
        #   Negative DBoxの損失上位のDBoxを取り出すマスクneg_maskを
        #   予測確信度を抽出できるように21クラス対応に拡張(バッチサイズ, 8732, 21)
        #
        # neg_mask: Negative DBoxの損失上位のDBoxを抽出するためのマスク
        # (バッチサイズ, 8732)
        # ↓
        # (バッチサイズ, 8732, 1) 2の次元(3)にサイズ1の次元を挿入
        # ↓
        # (バッチサイズ, 8732, 21) conf_data: 21クラスの予測確信度
        #                      (バッチサイズ, 8732, 21)と同じ形状に拡張
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # -----------------
        # ※conf_hnm:
        #   Positive DBoxとHard Negative MiningしたNegative DBoxの予測確信度
        # ※conf_t_label_hnm
        #   conf_hnmに対する教師データ(正解ラベル)
        # ※loss_c
        #   Positive DBoxとHard Negative MiningしたNegative DBoxにおける
        #   予測確信度の損失合計(要素数1)
        # -----------------

        # ※conf_hnm:
        #     (バッチサイズ,
        #      画像1枚中のPositiveの数 + HNMしたNegativeの数,
        #      21)
        #     ・Positive DBoxの予測確信度(21クラス)
        #     ・損失上位のNegative DBoxの予測確信度(21クラス)
        #
        # 1.pos_idx_mask + neg_idx_maskでDBoxのインデックスごとの
        # True/Falseをまとめる
        # 2.pos_idx_mask + neg_idx_maskからgt(0)で0より大きいTrue(1)の、
        # インデックスを取得
        # 3.conf_data(21クラスの予測確信度(バッチサイズ, 8732, 21))
        # から2で取得したインデックスの予測確信度を取得
        # 4.view(-1, num_classes)で21クラスごとに整列
        conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)
                             ].view(-1, num_classes)
        
        # ※conf_t_label_hnm:
        #     (バッチサイズ, 画像1枚中のPositiveの数 + HNMしたNegativeの数)
        #     ・Positive DBoxの正解ラベル
        #     ・損失上位のNegative DBoxの正解ラベル
        #
        # 正解ラベルの教師データconf_t_label(バッチサイズ,8732[正解ラベル])から
        # 予測確信度と同じようにpos_maskとneg_maskの正解ラベルを取り出す
        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

        # ※loss_c
        #     Positive DBoxとHard Negative MiningしたNegative DBoxの
        #     予測確信度の損失
        #     (バッチサイズ, 画像1枚中のPositiveとHNMしたNegativeの損失)
        #     ↓
        #     [確信度の損失の合計,]
        #
        # One-hot表現のconf_hnmに対し、正解ラベルconf_t_label_hnmは
        # 内部でOne-hot化されて処理
        loss_c = F.cross_entropy(
            conf_hnm, # Posiと損失上位のNegaの予測確信度(21クラス)
            conf_t_label_hnm, # Posiと損失上位のNegaの正解ラベル
            reduction='sum')  # すべての損失値を合計

        # -----------------
        # ※loss_l:
        #    ミニバッチにおける「Positive DBoxのオフセット情報の損失平均」を求める
        # ※loss_c:
        #    ミニバッチにおける「num_pos+num_negの確信度の損失平均」を求める
        # -----------------

        # ※N(int):
        # ミニバッチにおける全てのPositive DBoxの数
        #
        # num_pos(バッチサイズ, 画像1枚中のPositiveの数)の合計を求める
        N = num_pos.sum()
        
        # ※loss_l:
        #    ミニバッチにおけるPositive DBoxのオフセット情報の損失平均
        # 
        # ミニバッチにおけるPositive DBoxのオフセット情報の損失合計を
        # ミニバッチのPositive DBoxの総数で割る
        loss_l /= N
        
        # ※loss_c:
        #    ミニバッチにおけるPositive DBoxの確信度の損失平均
        # 
        # ミニバッチにおけるPosiと損失上位のNegaの確信度の損失合計を
        # ミニバッチのPositive DBoxの総数で割る
        loss_c /= N

        return loss_l, loss_c

