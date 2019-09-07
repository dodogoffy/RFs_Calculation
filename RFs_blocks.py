import math

# convnet = [ 
#             [blocks_times, [[kernel, stride, padding, dilation], [...]...],
#             [blocks_times, [[kernel, stride, padding, dilation], [...]...],
#             [blocks_times, [[kernel, stride, padding, dilation], [...]...],
#             [blocks_times, [[kernel, stride, padding, dilation], [...]...]
#           ]
# 如果是池化层，就直接将padding=0，因为pool和convolution 在计算RFs唯一的区别就是在padding
# darknet-53 结构
# block里，前面是一个1*1卷积，后面是一个3*3的卷积，所有的block结构一致
convnet = [
            [1, [[3, 1, 1, 1]]],
            [1, [[3, 2, 1, 1]]],               # stride convolution
            [1, [[1, 1, 0, 1], [3, 1, 1, 1]]],
            [1, [[3, 2, 1, 1]]],               # stride convolution
            [2, [[1, 1, 0, 1], [3, 1, 1, 1]]],
            [1, [[3, 2, 1, 1]]],               # stride convolution
            [8, [[1, 1, 0, 1], [3, 1, 1, 1]]],
            [1, [[3, 2, 1, 1]]],               # stride convolution
            [8, [[1, 1, 0, 1], [3, 1, 1, 1]]],
            [1, [[3, 2, 1, 1]]],               # stride convolution
            [4, [[1, 1, 0, 1], [3, 1, 1, 1]]]
          ]

def outFromIn(conv, layerIn):
  n_in = layerIn[0]   # Width or height
  j_in = layerIn[1] 
  r_in = layerIn[2]   # receprive field
  start_in = layerIn[3] 
 
  stage = 0  # 表示network的不同阶段
  # for blocks, layers in conv.items():
  for items in convnet:
    blocks  = items[0]  # blocks堆叠次数
    layers  = items[1]  # 每个blocks的具体内容
    stage += 1
    for _ in range(blocks):              # 表示这个blocks要循环几次
      for nlayer in layers:             # 每个blocks具体怎么样
        k = nlayer[0]  # kerner_size
        s = nlayer[1]  # stride
        p = nlayer[2]  # padding
        d = nlayer[3]  # dilation

        n_out = math.floor((n_in - d*(k-1)-1 + 2*p)/s) + 1  # 求取feature map的输出大小
        actualP = (n_out-1)*s - n_in + k                    # 实际上的padding 
        pL = math.floor(actualP/2)                          # 返回下限
      
        j_out = j_in * s                                    # stride
        r_out = r_in + d*(k - 1)*j_in                         # 计算的是当前层的感受野
        start_out = start_in + ((k-1)/2 - pL)*j_in          # 当这一层是stride convolution / pool时，第一个feature map的起点start会改变

        # 为下一轮准备
        n_in = n_out
        j_in = j_out
        r_in = r_out
        start_in = start_out

    # 在每个block结束
    print("stage{0}: \n\tfeature map:{1}x{1}, \n\tstarts:{4}, \n\tstrides:{3}, \n\tRFs:{2}"
          .format(stage, n_out, r_out, j_out, start_out))
  # return n_out, j_out, r_out, start_out

#%%
# [image_szie, init_stride, init_RFs, init_start]
# only image_size can change for your backbone.
init_layer = [416, 1, 1, 0.5]
outFromIn(conv=convnet, layerIn=init_layer)
