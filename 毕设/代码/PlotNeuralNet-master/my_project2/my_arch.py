import sys
sys.path.append('../')
from pycore.tikzeng import *


# 定义Bottleneck块
#def to_Bottleneck(name, s_filer, n_filer, offset="(0,0,0)", to="(0,0,0)", width=(2,2,2), height=40, depth=40):
def to_Bottleneck(name, s_filer, n_filer, offset="(0,0,0)", to="(0,0,0)", width=(2,2,2), height=40, depth=40):
    # 对第一个块进行特殊处理，设置步幅
    return [
        # to_Conv(name=f'{name}_conv1', s_filer=s_filer, n_filer=n_filer[0], offset=offset, to=to, width=width[0], height=height, depth=depth),
        # to_Conv(name=f'{name}_conv2', s_filer=s_filer, n_filer=n_filer[1], offset="(0,0,0)", to=f"({name}_conv1-east)", width=width[1], height=height, depth=depth),
        # to_Conv(name=f'{name}_conv3', s_filer=s_filer, n_filer=n_filer[2], offset="(0,0,0)", to=f"({name}_conv2-east)", width=width[2], height=height, depth=depth),
        # to_skip(of=f'{name}_conv1', to=f'{name}_conv3', pos=1.5),

        to_Conv(name=f'{name}_conv1', s_filer=s_filer, n_filer=n_filer[0], offset=offset, to=to, width=1, height=height, depth=depth),  # 1x1卷积
        to_Conv(name=f'{name}_conv2', s_filer=s_filer, n_filer=n_filer[1], offset="(0,0,0)", to=f"({name}_conv1-east)", width=width[1], height=height, depth=depth),  # 3x3卷积
        to_Conv(name=f'{name}_conv3', s_filer=s_filer, n_filer=n_filer[2], offset="(0,0,0)", to=f"({name}_conv2-east)", width=1, height=height, depth=depth),  # 1x1卷积
        to_skip(of=f'{name}_conv1', to=f'{name}_conv3', pos=1.5),
    ]

# 定义网络结构
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    
    to_input('img_1.png', width=8, height=8),
    to_ConvConvRelu(name='conv1', s_filer=64, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40),
    to_Pool(name='pool1', offset="(0,0,0)", to="(conv1-east)", width=1, height=35, depth=35),

    # 第一组Bottleneck块
    *to_Bottleneck("bottleneck1_1", s_filer=56, n_filer=(64, 64, 256), offset="(1,0,0)", to="(pool1-east)", width=(2,2,2)),  # 第一个残差块不进行下采样
    *to_Bottleneck("bottleneck1_2", s_filer=28, n_filer=(64, 64, 256), offset="(1,0,0)", to="(bottleneck1_1-east)", width=(2,2,2)),
    *to_Bottleneck("bottleneck1_3", s_filer=28, n_filer=(64, 64, 256), offset="(1,0,0)", to="(bottleneck1_2-east)", width=(2,2,2)),


    # #2 ... 重复添加更多的Bottleneck块以表示其他层 ...
    # *to_Bottleneck("bottleneck2_1", s_filer=28, n_filer=(128, 128, 512), offset="(2,0,0)", to="(bottleneck1_3-east)", width=(2,2,2)),
    # *to_Bottleneck("bottleneck2_2", s_filer=14, n_filer=(128, 128, 512), offset="(1.5,0,0)", to="(bottleneck2_1-east)", width=(2,2,2)),
    # *to_Bottleneck("bottleneck2_3", s_filer=14, n_filer=(128, 128, 512), offset="(1.5,0,0)", to="(bottleneck2_2-east)", width=(2,2,2)),
    # *to_Bottleneck("bottleneck2_4", s_filer=14, n_filer=(128, 128, 512), offset="(1.5,0,0)", to="(bottleneck2_3-east)", width=(2,2,2)),

    # #3
    # *to_Bottleneck("bottleneck3_1", s_filer=14, n_filer=(256, 256, 1024), offset="(2.5,0,0)", to="(bottleneck2_4-east)", width=(2,2,2)),
    # *to_Bottleneck("bottleneck3_2", s_filer=7, n_filer=(256, 256, 1024), offset="(1.5,0,0)", to="(bottleneck3_1-east)", width=(2,2,2)),
    # *to_Bottleneck("bottleneck3_3", s_filer=7, n_filer=(256, 256, 1024), offset="(1.5,0,0)", to="(bottleneck3_2-east)", width=(2,2,2)),
    # *to_Bottleneck("bottleneck3_4", s_filer=7, n_filer=(256, 256, 1024), offset="(1.5,0,0)", to="(bottleneck3_3-east)", width=(2,2,2)),
    # *to_Bottleneck("bottleneck3_5", s_filer=7, n_filer=(256, 256, 1024), offset="(1.5,0,0)", to="(bottleneck3_4-east)", width=(2,2,2)),
    # *to_Bottleneck("bottleneck3_6", s_filer=7, n_filer=(256, 256, 1024), offset="(1.5,0,0)", to="(bottleneck3_5-east)", width=(2,2,2)),

    # #4
    # *to_Bottleneck("bottleneck4_1", s_filer=7, n_filer=(512, 512, 2048), offset="(3,0,0)", to="(bottleneck3_6-east)", width=(2,2,2)),
    # *to_Bottleneck("bottleneck4_2", s_filer=3, n_filer=(512, 512, 2048), offset="(1.5,0,0)", to="(bottleneck4_1-east)", width=(2,2,2)),
    # *to_Bottleneck("bottleneck4_3", s_filer=3, n_filer=(512, 512, 2048), offset="(1.5,0,0)", to="(bottleneck4_2-east)", width=(2,2,2)),
    

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
