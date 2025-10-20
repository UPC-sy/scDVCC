import argparse

def parse_args():#解析输入参数
    parser = argparse.ArgumentParser()

    # Logical
    parser.add_argument('--save_model', action='store_true', default=True)

    # Experiments
    parser.add_argument('--name', type=str, default='Zeisel', help='Worm_neuron_cells, 10X_PBMC, Zeisel, Klein, Baron, Camp, Human_kidney_cells, MCA, Mouse_ES, Shekhar')
    parser.add_argument('--n_clusters', type=int, default=9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)

    # Preprocessing
    parser.add_argument('--HVG', type=float, default=0.2)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--sf', action='store_false', default=True)
    parser.add_argument('--log', action='store_false', default=True)
    parser.add_argument('--normal', action='store_false', default=True)

    # Layers
    parser.add_argument("--layers", nargs='?', default='[256,64]', help='[256, 128, 64], [256,64]')
    
    # learning
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--fine_lr', default=1.0, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--fine_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--ns", nargs='?', default='[2048,1024]')
    parser.add_argument('--recon', type=str, default='zinb', help='mse,zinb')

    # Hyper-Parameters
    parser.add_argument('--tau', type=float, default=0.42)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument("--granularity", nargs='?', default='[1,1.5,2]')
    parser.add_argument('--lam1', default=1.0, type=float)
    parser.add_argument('--lam2', default=0.05, type=float)
    parser.add_argument('--lam3', default=0.7, type=float)

    parser.add_argument('--r', default=0.99, type=float, help='threshold to terminate pre-training stage')
    parser.add_argument('--tol', default=0.00001, type=float, help='tolerance for delta clustering labels to terminate fine-tuning stage')

    # Augmentation
    parser.add_argument("--df_1", type=float, default=0.237)
    parser.add_argument("--df_2", type=float, default=0.237)
        
    return parser.parse_known_args()

def enumerateConfig(args):#返回输入参数的所有属性名称和对应的值的
    # 创建两个空列表，用于存储属性名称和对应的值
    args_names = []
    args_vals = []

    # 遍历输入参数的所有属性
    for arg in vars(args):
        # 将属性名称添加到args_names列表中
        args_names.append(arg)
        # 将属性值添加到args_vals列表中
        args_vals.append(getattr(args, arg))
    # 返回属性名称列表和属性值列表
    return args_names, args_vals
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    return args_names, args_vals

def printConfig(args):
    # 调用enumerateConfig函数，获取输入参数的所有属性名称和对应的值
    args_names, args_vals = enumerateConfig(args)
    # 创建一个空字符串，用于存储最终的输出字符串
    st = ''
    # 遍历属性名称和对应的值
    for name, val in zip(args_names, args_vals):
        # 如果属性值为False，则跳过当前循环
        if val == False:
            continue
        # 将属性名称和对应的值格式化为字符串，并添加到输出字符串中
        st_ = "{} <- {} / ".format(name, val)
        st += st_
    # 返回输出字符串，去掉最后的一个字符（即"/"）
    return st[:-1]



