import sys
sys.path.append('../')
from pycore.tikzeng import *

# 修改后的to_Bottleneck函数
#def to_Bottleneck(name, s_filer, n_filer, offset="(0,0,0)", to="(0,0,0)", width=(2,2,4), height=40, depth=40, stride=1):
def to_Bottleneck(name, s_filer, n_filer, offset="(0,0,0)", to="(0,0,0)", width=(2,2,4), height=40, depth=40, downsample=False): 
    # 第二个卷积层的位置偏移可能需要根据步幅进行调整
    #offset_conv2 = "(1,0,0)" if stride == 1 else "(2,0,0)"  # 假设步幅为2时，间距更大
    s_filer_conv2 = s_filer // 2 if downsample else s_filer

    layers = [
        #to_Conv(name=f'{name}_conv1', s_filer=s_filer, n_filer=n_filer[0], offset=offset, to=to, width=width[0], height=height, depth=depth),  # 1x1卷积
        #to_Conv(name=f'{name}_conv2', s_filer=s_filer, n_filer=n_filer[1], offset=offset_conv2, to=f"({name}_conv1-east)", width=width[1], height=height, depth=depth),  # 3x3卷积
        #to_Conv(name=f'{name}_conv3', s_filer=s_filer, n_filer=n_filer[2], offset="(1,0,0)", to=f"({name}_conv2-east)", width=width[2], height=height, depth=depth),  # 1x1卷积
        #to_Skip(of=f'{name}_conv1', to=f'{name}_conv3', pos=1.5),
        to_Conv(name=f'{name}_conv1', s_filer=s_filer, n_filer=n_filer[0], offset=offset, to=to, width=width[0], height=height, depth=depth),
        to_Conv(name=f'{name}_conv2', s_filer=s_filer_conv2, n_filer=n_filer[1], offset="(1,0,0)", to=f"({name}_conv1-east)", width=width[1], height=height, depth=depth),
        to_Conv(name=f'{name}_conv3', s_filer=s_filer_conv2, n_filer=n_filer[2], offset="(1,0,0)", to=f"({name}_conv2-east)", width=width[2], height=height, depth=depth),
        to_skip(of=f'{name}_conv1', to=f'{name}_conv3', pos=1.5),
    ]

    # 添加跳过连接
    #layers.append(to_skip(of=f'{name}_conv1', to=f'{name}_conv3', pos=1.5))

    return layers

# 模拟全连接层的函数
def to_FullyConnected(name, s_filer, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=""):
    return [
        to_Conv(name=name, s_filer=s_filer, n_filer=1, offset=offset, to=to, width=width, height=height, depth=depth, caption=caption)
    ]


arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    
    to_input('img_1.png', width=8, height=8),
    to_ConvConvRelu(name='conv1', s_filer=64, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40),
    to_Pool(name='pool1', offset="(0,0,0)", to="(conv1-east)", width=1, height=35, depth=35),

    # 第一组Bottleneck块
    *to_Bottleneck("bottleneck1_1", s_filer=56, n_filer=(64, 64, 256), offset="(3,0,0)", to="(pool1-east)", width=(2,2,4), height=40, depth=40),  # 第一个残差块不进行下采样
    *to_Bottleneck("bottleneck1_2", s_filer=56, n_filer=(64, 64, 256), offset="(8,0,0)", to="(bottleneck1_1-east)", width=(2,2,4), height=40, depth=40),
    *to_Bottleneck("bottleneck1_3", s_filer=56, n_filer=(64, 64, 256), offset="(13,0,0)", to="(bottleneck1_2-east)", width=(2,2,4), height=40, depth=40),

    # 第二组Bottleneck块
    *to_Bottleneck("bottleneck2_1", s_filer=28, n_filer=(128, 128, 512), offset="(18,0,0)", to="(bottleneck1_3-east)", width=(3.5,3.5,4), height=35, depth=35, downsample=True),
    *to_Bottleneck("bottleneck2_2", s_filer=28, n_filer=(128, 128, 512), offset="(23,0,0)", to="(bottleneck2_1-east)", width=(3.5,3.5,4), height=35, depth=35),
    *to_Bottleneck("bottleneck2_3", s_filer=28, n_filer=(128, 128, 512), offset="(28,0,0)", to="(bottleneck2_2-east)", width=(3.5,3.5,4), height=35, depth=35),
    *to_Bottleneck("bottleneck2_4", s_filer=28, n_filer=(128, 128, 512), offset="(33,0,0)", to="(bottleneck2_3-east)", width=(3.5,3.5,4), height=35, depth=35),

    # 第三组Bottleneck块
    *to_Bottleneck("bottleneck3_1", s_filer=14, n_filer=(256, 256, 1024), offset="(38,0,0)", to="(bottleneck2_4-east)", width=(4,4,4), height=30, depth=30,downsample=True),
    *to_Bottleneck("bottleneck3_1", s_filer=14, n_filer=(256, 256, 1024), offset="(43,0,0)", to="(bottleneck2_4-east)", width=(4,4,4), height=30, depth=30),
    *to_Bottleneck("bottleneck3_1", s_filer=14, n_filer=(256, 256, 1024), offset="(48,0,0)", to="(bottleneck2_4-east)", width=(4,4,4), height=30, depth=30),
    *to_Bottleneck("bottleneck3_1", s_filer=14, n_filer=(256, 256, 1024), offset="(53,0,0)", to="(bottleneck2_4-east)", width=(4,4,4), height=30, depth=30),
    *to_Bottleneck("bottleneck3_1", s_filer=14, n_filer=(256, 256, 1024), offset="(58,0,0)", to="(bottleneck2_4-east)", width=(4,4,4), height=30, depth=30),
    *to_Bottleneck("bottleneck3_1", s_filer=14, n_filer=(256, 256, 1024), offset="(63,0,0)", to="(bottleneck2_4-east)", width=(4,4,4), height=30, depth=30),

    # 第四组Bottleneck块
    *to_Bottleneck("bottleneck4_1", s_filer=7, n_filer=(512, 512, 2048), offset="(68,0,0)", to="(bottleneck3_6-east)", width=(4.5,4.5,4), height=25, depth=25, downsample=True),
    *to_Bottleneck("bottleneck4_2", s_filer=7, n_filer=(512, 512, 2048), offset="(73,0,0)", to="(bottleneck4_1-east)", width=(4.5,4.5,4), height=25, depth=25),
    *to_Bottleneck("bottleneck4_3", s_filer=7, n_filer=(512, 512, 2048), offset="(78,0,0)", to="(bottleneck4_2-east)", width=(4.5,4.5,4), height=25, depth=25),

    to_Pool(name="pool2", offset="(83,0,0)", to="(bottleneck4_3-east)", width=1, height=25, depth=25, opacity=0.5),

    #to_FullyConnected(name="fc", s_filer=9, offset="(88,0,0)", to="(pool2-east)", width=1, height=1, depth=40, caption="Fully Connecte"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()


