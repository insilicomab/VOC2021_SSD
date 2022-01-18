'''
1. データの拡張処理を構成するクラス
'''
import cv2
import numpy as np
from numpy import random

class Compose(object):
    def __init__(self, transforms):
        '''
        Args:
            transforms (List[Transform]): 変換処理のリスト
        Example:
            >>> augmentations.Compose([
            >>>     transforms.CenterCrop(10),
            >>>     transforms.ToTensor(),
            >>> ])

            '''
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

'''
2. ピクセルデータのint型をfloat32に変換するクラス
'''
class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


'''
3. アノテーションデータの正規化を元の状態に戻すクラス
'''
class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels

'''
4. 輝度(明るさ)をランダムに変化させるクラス
'''
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels

'''
5. コントラストをランダムに変化させるクラス
'''
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

'''
6. BGRとHSVを相互変換するクラス
'''
class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels

'''
7. 彩度をランダムに変化させるクラス
'''
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels

'''
8. ランダムに色相を変化させるクラス
'''
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

'''
9. 測光に歪みを加えるクラス
'''
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels

'''
10. 色チャネルの並び順を変えるクラス
'''
class SwapChannels(object):
    def __init__(self, swaps):
        '''
        Args:
            swaps (int triple): final order of channels
                eg: (2, 1, 0)
        '''
        self.swaps = swaps

    def __call__(self, image):
        '''
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        '''
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

'''
11. 輝度(明るさ)、彩度、色相、コントラストを変化させ、歪みを加えるクラス
'''
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            # コントラスト(BGRに適用)
            RandomContrast(),
            # カラーモデルをHSVにコンバート
            ConvertColor(transform='HSV'),
            # 彩度の変化(HSVに適用)
            RandomSaturation(),
            # 色相の変化(HSVに適用)
            RandomHue(),
            # カラーモデルをHSVからBGRにコンバート
            ConvertColor(current='HSV', transform='BGR'),
            # コントラストの変化(BGRに適用)
            RandomContrast()
        ]
        # 輝度
        self.rand_brightness = RandomBrightness()
        # 測光の歪み
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        # 明るさの変化
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        # 彩度、色相、コントラストの適用は上限と下限の間でランダムに
        # 歪みオフセットを選択することにより、確率0.5で適用
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        # 彩度、色相、コントラストの適用
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

'''
12. イメージをランダムに拡大するクラス
'''
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels

'''
13. イメージの左右をランダムに反転するクラス
'''
class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

'''
14. アノテーションデータを0～1.0の範囲に正規化するクラス
'''
class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels

'''
15. イメージのサイズをinput_sizeにリサイズするクラス
'''
class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image,
                           (self.size, self.size))
        return image, boxes, labels

'''
16. 色情報(RGB値）から平均値を引き算するクラス
'''
class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels
      
'''
17. 2セットのBBoxの重なる部分を検出する

'''
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

'''
18. 2セットのBBoxの類似度を示すジャッカード係数を計算する
'''
def jaccard_numpy(box_a, box_b):
    '''
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    '''
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

'''
19. イメージの特定の領域をランダムに切り出すクラス
'''
class RandomSampleCrop(object):
    '''
    イメージの切り出しに合わせてバウンディングボックスも変形させる
    
    Arguments:
        img (Image): トレーニング中に入力されるイメージ
        boxes (Tensor): オリジナルのバウンディングボックス
        labels (Tensor): バウンディングボックスのラベル
        mode (float tuple): 2セットのBBoxの類似度を示すジャッカード係数
    Return:
        (img, boxes, classes)
            img (Image): トリミングされたイメージ
            boxes (Tensor): 調整後のバウンディングボックス
            labels (Tensor): バウンディングボックスのラベル
    '''
    def __init__(self):
        self.sample_options = (
            # 元の入力イメージ全体を使用
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # パッチをランダムにサンプリングする
            (None, None),
        )
        # オリジナルの実装ではタプルの要素のサイズが異なるため警告が出ます
        # 警告を回避するためにobject型のndarrayに変換するようにしました
        self.sample_options = np.array(self.sample_options, dtype=object)
        
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # ランダムにモードを選択
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # トレースの最大値(50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # アスペクト比 constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # イメージからトリミングする領域を作る
                # x1,y1,x2,y2を整数(int)に変換
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                # オリジナルのBBoxとトリミング領域の
                # IoU（ジャッカードオーバーラップ）を計算
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                # 最小および最大のオーバーラップ制約が満たされていなければ再試行
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                # イメージから切り抜く
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                # gt boxとサンプリングされたパッチのセンターを合わせる
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                # 左上側にあるすべてのgtボックスをマスクする
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                # 右下側にあるすべてのgtボックスをマスクする
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                # 有効なボックスがなければ再試行
                if not mask.any():
                    continue

                # take only matching gt boxes
                # gtボックスを取得
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                # ラベルを取得
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                # ボックスの左上隅を使用する
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                # トリミングされた状態に合わせる
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                
                return current_image, current_boxes, current_labels