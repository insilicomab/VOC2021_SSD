'''
1. DBoxの情報をBBox形式に変換する関数
'''
import torch

def point_form(boxes):
    '''DBoxの情報[cx, cy, width, height]を[xmin, ymin, xmax, ymax]に変換
    
    Parameters:
      boxes(Tensor): DBoxの情報(DBoxの数, 4[cx, cy, width, height])
    Returns:
        boxes(Tensor): BBoxの情報(BBoxの数, 4[xmin, ymin, xmax, ymax])
    '''
    return torch.cat(
        (boxes[:, :2] - boxes[:, 2:]/2, # (幅(高さ)-センター値)/2=xmin, ymin
         boxes[:, :2] + boxes[:, 2:]/2),# 幅(高さ)+(センター値/2)=xmax, ymax
        1)

'''
2. 2個のボックスが重なる部分の面積を求める関数
'''
def intersect(box_a, box_b):
    ''' 2個のボックスが重なる部分の面積を求める

    Parameters:
      box_a(Tensor): BBoxの4辺の座標(BBoxの数、4[xmin,ymin,xmax,ymax])
      box_b(Tensor): BBoxの4辺の座標(BBoxの数、4[xmin,ymin,xmax,ymax])
    Return(Tensor):
      box_aとbox_bの重なり部分の面積
    '''
    A = box_a.size(0) # box_aの0番目次元の要素数(ボックスの数)を取得
    B = box_b.size(0) # box_bの0番目次元の要素数(ボックスの数)を取得
    
    # 以下のtorch.minとtorch.maxでbox_aとbox_bの
    # すべての組み合わせに対して、双方のボックスが重なる部分の
    # 4辺の座標[xmin,ymin,xmax,ymax]を求める
    #
    # box_aとbox_bの(xmax,ymax)のすべての組み合わせに対して
    # min()で小さい方のxmax、ymaxを抽出(x軸、y軸の最大値を切詰め)
    #
    # A(0)からA(n)までB(1)～B(n)を繰り返し組み合わせることで
    # すべての組み合わせを作る
    # A(0)に対してB(1)～B(n)を組み合わせる
    # ↓
    # A(n)に対してB(1)～B(n)を組み合わせる
    max_xy = torch.min(
        # box_a(0)の[xmax,ymax]をbox_bの数だけ並べる(Aの数,2)
        # これをbox_a(n)まで繰り返す
        #
        # (Aの数,2)をunsqueeze(1)で(Aの数,1,2[x,y])に拡張して
        # expand(A, B, 2)で(Aの数,Bの数,2[x,y])にする
        # [[[A(0)のx, y],  ↑
        #    ...         Bの数
        #   [A(0)のx, y]], ↓
        #   ... 
        #  [[A(n)のx, y]   ↑
        #    ...         Bの数
        #   [A(n)のx, y]]] ↓
        #
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        # box_b(0)～(n)の[xmax,ymax]を並べる
        # これをbox_a(n)まで繰り返す
        #
        # (Bの数,2)をunsqueeze(0)で(1,Bの数,2[x,y])に拡張して
        # expand(A, B, 2)で(Aの数,Bの数,2)にする
        # [[[B(0)のx, y],  ↑
        #    ...         Bの数
        #   [B(n)のx, y]], ↓
        #   ... 
        #  [[B(0)のx, y]   ↑
        #    ...         Bの数
        #   [B(n)のx, y]]] ↓
        #
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    
    # box_aとbox_bの[xmin,ymin]の全ての組み合わせに対して
    # maxのxmin、yminを抽出(x軸、y軸の最小値を切り上げ)
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    
    # xmax-xmin、ymax-yminを計算して重なる部分の幅と高さを求める
    # 負の値は0にする
    inter = torch.clamp((max_xy - min_xy), min=0)
    
    # 幅×高さを計算してbox_a、box_bのすべてのボックスの組み合わせにおける
    # 重なり部分の面積(A ∩ B)を求める
    #
    # テンソルの形状は(box_aの要素数, box_bの要素数)
    # [[A(0)∩B(0), A(0)∩B(1), ...],
    #  [A(1)とB(0), A(1)∩B(1), ...],
    #  ... ]
    return inter[:, :, 0] * inter[:, :, 1]

'''
3. ジャッカード係数（IoU）を求める関数
'''
def jaccard(box_a, box_b):
    '''2つのボックス間のジャッカード係数(IoU)を計算
       A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    Parameters:
      box_a(Tensor):
        ミニバッチの1枚の画像におけるBBoxのアノテーションデータ
        (BBoxの数, 5[xmin, ymin, xmax, ymax, label_index])
      box_b(Tensor):
        DBoxの4辺の座標(DBoxの数、4[xmin,ymin,xmax,ymax])
    Return(Tensor):
      BBoxとすべてのDBoxの組み合わせにおけるjaccard係数(IoU)
      テンソルの形状は(box_aのボックス数, box_bのボックス数)
    '''
    # box_aとbox_bのすべてのボックスの組み合わせで重なり部分の面積(A∩B)を取得
    # テンソルの形状は(box_aのボックス数, box_bのボックス数)
    inter = intersect(box_a, box_b)

    # box_aのすべてのボックスの面積を求める
    # box_a(0)の面積を求めてbox_bの数だけ並べ、これをbox_a(n)まで繰り返す
    #    
    # 1.(Aの数[面積],)をunsqueeze(1)でインデックス1の次元を拡張する(Aの数[面積],1)
    # 2.expand_as(inter)でAの面積をBの数だけ並べて(Aの数,Bの数)の形にする
    #
    #  ←--同じAの面積がBの数だけ並ぶ--→
    # [ [A(0)の面積, ..., A(0)の面積]   ↑
    #   ...                           Aの数
    #   [A(n)の面積, ..., A(n)の面積] ] ↓
    #
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
    
    # box_bのすべてのボックスの面積を求める
    # box_b(0)～(n)の面積を求めて並べ、これをbox_a(n)まで繰り返す
    #
    # 1.(Bの数[面積],)をunsqueeze(0)で0の次元を拡張する(1,Bの数[面積])
    # 2.expand_as(inter)でBの面積の列をAの数だけ並べて(Aの数,Bの数)の形にする
    #
    #  ←---Bの面積がBの数だけ並ぶ----→
    # [ [B(0)の面積, ..., B(n)の面積]   ↑
    #   ...                           Aの数
    #   [B(0)の面積, ..., B(n)の面積] ] ↓
    #
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)
    
    # area_aとarea_bのすべての組み合わせについて結合部分の面積を求める
    # テンソルの形状はすべて(box_aのボックス数, box_bのボックス数)
    # area_a + area_b - inter(A ∩ B) = A∪B(結合部分の面積)
    union = area_a + area_b - inter
    
    # area_aとarea_bのすべての組み合わせについてIoU値を求める
    # A ∩ B / A ∪ B = IoU値
    #
    # テンソルの形状は(box_aのボックス数, box_bのボックス数)
    # [[<box_a(0)に対する> box_b(0)のIoU値, ...,  box_b(n)のIoU値],
    #  ...
    #  [<box_a(n)に対する> box_b(0)のIoU値, ...,  box_b(n)のIoU値]]
    return inter / union

'''
4. 教師データloc、confを作成する関数
'''
def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    '''教師データloc、confを作成する
    
    Parameters:
        threshold(float):
          jaccard係数の閾値(0.5)
        truths(Tensor):
          ミニバッチの現在の画像におけるBBoxの座標情報
          (BBoxの数, 4[xmin, ymin, xmax, ymax])
        priors(Tensor):
          DBoxの情報
          (8732, 4[cx, cy, width, height])
        variances(list):
          DBoxを変形するオフセット値を計算する際に使用する係数
          [0.1, 0.2]
        labels(list[int]):
          正解ラベルのリスト[BBox1のラベル, BBox2のラベル, …]
        loc_t(Tensor):
          各DBoxに一番近い正解のBBoxの情報を格納するための
          (バッチサイズ, 8732, 4)の形状の3階テンソル
        conf_t(tensor):
          各DBoxに一番近い正解のBBoxのラベルを格納するための
          (バッチサイズ, 8732)の形状の3階テンソル
        idx(int):
          現在のミニバッチのインデックス
          
    Return(なし):
      教師データloc_t、conf_tに現在の画像のloc、confを追加する
    '''
    # BBoxとすべてのDBoxとの組み合わせについて「A ∩ B/A ∪ B = IoU値」を求める
    # 戻り値のoverlapsの形状は(正解BBoxの数, DBoxの数])
    # [[<BBox(0)に対する> DBox(0)のIoU値, ...,  DBox(n)のIoU値],
    #  ...
    #  [<BBox(n)に対する> DBox(0)のIoU値, ...,  DBox(n)のIoU値]]
    overlaps = jaccard(
        # ミニバッチの現在の画像におけるBBoxの座標情報
        # (BBoxの数, 4[xmin, ymin, xmax, ymax])
        truths, 
        # DBoxの(8732, 4[cx, cy, width, height])を
        # (8732, 4[xmin, ymin, xmax, ymax])に変換
        point_form(priors)
    )
    
    # BBoxにマッチするDBoxを抽出
    #
    # BBox(0)～BBox(n)の各ボックスのIoU値が最高になるDBoxを取得
    #
    # overlaps(正解BBoxの数, DBoxの数)に対し、
    # overlaps.max(1,keepdim=True)で1の次元[DBox(0), ..., DBox(8732)]
    # から最高IoU値と、そのIoU値を出したDBoxのインデックスを取得
    #
    # best_prior_overlap:(BBoxの数, 1[BBoxごとの最高IoU値])
    # best_prior_idx:(BBoxの数, 1[BBoxにマッチしたDBoxのインデックス])
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    
    # DBoxにマッチするBBoxを抽出
    #
    # DBox(0)～DBox(n)の各ボックスのIoU値が最高になるBBoxを取得
    #
    # overlaps(正解BBoxの数, DBoxの数)に対し、
    # overlaps.max(0, keepdim=True)で0の次元[BBox(0), ..., BBox(n)]
    # から最高IoU値と、そのIoU値を出したBoxのインデックスを取得
    #
    # best_truth_overlap:(1[DBoxごとの最高IoU値], DBoxの数)
    # best_truth_idx:(1[DBoxにマッチしたBBoxのインデックス], DBoxの数)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    
    # (1[DBoxにマッチしたBBoxのインデックス], DBoxの数)の0の次元を削除
    # →(DBoxの数[DBoxにマッチしたBBoxのインデックス],)
    best_truth_idx.squeeze_(0)
    # (1[DBoxごとの最高IoU値], DBoxの数)の0の次元を削除
    # →(DBoxの数[DBoxごとの最高IoU値,)
    best_truth_overlap.squeeze_(0)

    # (BBoxの数, 1[BBoxにマッチしたDBoxのインデックス])の1の次元を削除
    # →(BBoxの数[BBoxにマッチしたDBoxのインデックス],)
    best_prior_idx.squeeze_(1)
    # (BBoxの数, 1[BBoxごとの最高IoU値])の1の次元を削除
    # →(Boxの数[BBoxごとの最高IoU値],)
    best_prior_overlap.squeeze_(1)
    
    # best_truth_overlap:(DBoxの数[DBoxごとの最高IoU値],)
    # BBoxに最も類似したDBox(複数あり)のIoU値をMax値の倍の「2」に置き換え
    best_truth_overlap.index_fill_(
        0, # 操作対象の次元
        best_prior_idx, # (BBoxの数[BBoxに最も類似したDBoxのインデックス],)
        2) # IoU値の上限1の2倍の数に書き換え
    
    # best_truth_idxにリストアップされたBBoxをbest_prior_idxの結果と一致させる
    # best_prior_idxの要素数(BBoxの数)だけループ
    for j in range(best_prior_idx.size(0)):
        # (DBoxの数[最もIoUが高いBBoxのインデックス],)の要素について
        # best_prior_idxでBoxごとのマッチしたDBoxを調べ、BBoxにマッチしている
        # DBoxがあればマッチング先のBBoxのインデックスに書き換える
        best_truth_idx[best_prior_idx[j]] = j
    
    # BBoxのアノテーションからDBoxがマッチするBBoxのアノテーションを収集
    # matchesの形状:(DBoxの数,4[xmin, ymin, xmax, ymax])
    matches = truths[best_truth_idx]
    # DBoxの正解ラベルを作成
    # DBoxにマッチするBBoxの正解ラベルを取得して+1する
    # (背景ラベル0を入れるため1ずらす)
    # confの形状: (DBoxの数[マッチするBBoxの正解ラベル],)
    conf = labels[best_truth_idx] + 1
    # IoU値が0.5より小さいDBoxの正解ラベルを背景(0)にする
    # confの形状: (DBoxの数[0への置き換え処理後の正解ラベル],)
    conf[best_truth_overlap < threshold] = 0
    
    # DBoxのオフセット情報(DBoxの数,4[Δcx,Δcy,Δw,Δh])を生成
    loc = encode(
        matches,   # DBoxにマッチしたBBoxのアノテーションデータ(DBoxの数, 4)
        priors,    # DBoxの情報(8732, 4[cx, cy, width, height])
        variances) # オフセット値を計算する際に使用する係数[0.1, 0.2]
    
    # 教師データの登録
    #
    # loc_t[現在のミニバッチのインデックス]に
    # DBoxのオフセット情報loc(DBoxの数,4[Δcx,Δcy,Δw,Δh])を格納
    loc_t[idx] = loc
    # conf_t[現在のミニバッチのインデックス]に
    # 正解ラベルconf(DBoxの数[0への置き換え処理後の正解ラベル],)を格納
    conf_t[idx] = conf

'''
5. DBoxのオフセット情報を作る関数
'''
def encode(matched, priors, variances):
    '''DBoxの情報[cx, cy, width, height]を
       DBoxのオフセット情報[Δcx,Δcy,Δw,Δh]に変換する
       
        matched(Tensor):
          DBoxにマッチしたBBoxのアノテーションデータ
          形状は(DBoxの数, 4)
        priors(Tensor) :
          DBoxの情報
          形状は(8732, 4[cx, cy, width, height])
        variances(list[float]):
          DBoxを変形するオフセット値を計算する際に使用する係数[0.1, 0.2]
    Return:
        DBoxのオフセット情報
        (DBoxの数, 4[Δcx,Δcy,Δw,Δh])
    '''
    # dist b/t match center and prior's center
    # DBoxのオフセット情報Δcx, Δcy, Δw, Δhを求める

    # DBoxのオフセット情報Δcx, Δcyを求める
    # g_cxcy[(cx - cx_d),
    #        (cy - cy_d)]
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # g_cxcy[ (cx - cx_d) / + 0.1 * w_d,
    #         (cy - cy_d) / + 0.1 * w_h ]
    g_cxcy /= (variances[0] * priors[:, 2:])
    
    # DBoxのオフセット情報Δw, Δhを求める
    # g_wh=[(BBoxのw/DBoxのw), (BBoxのh/DBoxのh)]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    # g_wh=[(log(BBoxのw/DBoxのw) / 0.2), (log(BBoxのh/DBoxのh) / 0.2)]
    g_wh = torch.log(g_wh) / variances[1]
    
    # Δcx,ΔcyとΔw,Δhを1の次元で連結して
    # (DBoxの数,4[Δcx,Δcy,Δw,Δh])の形状にする
    return torch.cat([g_cxcy, g_wh], 1)
