import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", default=42,
                        type=int)

    # Specify network architectures
    parser.add_argument("-net_type", default='mlp',
                        type=str, help='mlp or cnn')
    parser.add_argument("-cnn_arch", default='64-64-avg-128-128',
                        type=str, help='determines cnn architecture. string should\
                            consist of ints and `avg` separated by dashes. ints \
                            determine number of channels, `avg` denotes 2x2 avg-pooling. \
                            output of conv-layers is followed by global average pooling, \
                            one fully connected layer.')
    parser.add_argument("-n_hidden_layers", default=1,
                        type=int, help='for mlps only')
    parser.add_argument("-width", default=1000,
                        type=int, help='for mlps only')
    parser.add_argument("-bn", default=0,
                        type=int, help='batch norm, can be `0` or `1`')

    parser.add_argument("-dataset", default='fashion',
                        type=str, help='`mnist`, `fashion`, or `cifar10`')

    # Specify training details
    parser.add_argument("-n_param_updates", default=500,
                        type=int, help='how long is net trained')
    parser.add_argument("-measure_period", default=500,
                        type=int, help='how often are weights stored. \
                            determines points at which performance and \
                            permutations can be permuted post-training.')
    parser.add_argument("-opt", default='AdamW',
                        type=str, help='`SGD` or `AdamW`, determines optimizer for first net')
    parser.add_argument("-opt_2", default='same',
                        type=str, help='`same`, `SGD` or `AdamW` determines optimizer of second net')
    parser.add_argument("-batch_size", default=100,
                        type=int, help='batch size with which first net is trained')
    parser.add_argument("-batch_size_2", default=-1,
                        type=int, help='batch size with which second net is trained.\
                         -1 means using same bs as first net.')
    parser.add_argument("-lr", default=0.001,
                        type=float)
    parser.add_argument("-lr_2", default=-1.,
                        type=float)
    parser.add_argument("-sgd_mom", default=0.9,
                        type=float)
    parser.add_argument("-sgd_mom_2", default=-1.,
                        type=float)

    # Choose device
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-visible_gpu", default='0')

    return parser


def preprocess_args(args):
    attr_list = ['lr', 'batch_size', 'opt', 'sgd_mom']
    for attr in attr_list:
        y = getattr(args, attr+'_2')
        if y in [-1, 'same']:
            x = getattr(args, attr)
            setattr(args, attr+'_2', x)
    return args
    