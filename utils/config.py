import argparse


def new_parser(name=None):
    return argparse.ArgumentParser(prog=name)


def add_dataset_argument(parser):
    parser.add_argument('--dataset', type=str,
                        help='dataset name, currently support datasets are: \
                            YahooAnswers', default='YahooAnswers')


def add_model_argument(parser):
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--epoch_threshold', type=int, help='use epoch_threshold')
    parser.add_argument('--coef_', type=float, default=0.5, help='use epoch_threshold')
    parser.add_argument('--init_2', type=int, default=0, help='use epoch_threshold')
    parser.add_argument('--learn', type=int, default=0, help='the lambda hyperparameter learning in the training process or not')

    parser.add_argument('--task_num', type=int, help='number of tasks')
    parser.add_argument('--use_MOO', type=int, help='use MOO')
    parser.add_argument('--MOO_name', type=str, help='MOO Variant')

    parser.add_argument('--num_topics', type=int, default=50)
    parser.add_argument('--num_groups', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_pretrainWE', action='store_true',
                        default=False, help='Enable use_pretrainWE mode')
    parser.add_argument('--theta_temp', type=float, default=1.)
    parser.add_argument('--DT_alpha', type=float, default=3.)
    parser.add_argument('--TW_alpha', type=float, default=2.)

    parser.add_argument('--weight_ECR', type=float, default=40.)
    parser.add_argument('--weight_GR', type=float, default=1.)
    parser.add_argument('--alpha_ECR', type=float, default=20.)
    parser.add_argument('--alpha_GR', type=float, default=5.)
    parser.add_argument('--weight_CTR', type=float, default=0.)
    parser.add_argument('--weight_InfoNCE', type=float, default=50.)
    parser.add_argument('--weight_CL', type=float, default=0.0)
    parser.add_argument('--beta_temp', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0)


def add_training_argument(parser):
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the model, cuda or cpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_scheduler', type=str,
                        help='learning rate scheduler, dont use if not needed, \
                            currently support: step')
    parser.add_argument('--lr_step_size', type=int, default=125,
                        help='step size for learning rate scheduler')

    # FSAM
    parser.add_argument('--rho', type=float, default=0.05,
                         help='rho')
    parser.add_argument('--sigma', type=float, default=1,
                        help='sigma') 
    parser.add_argument('--lmbda', type=float, default=0.9,
                        help='lmbda') 
    parser.add_argument('--acc_step', type=float, default=8,
                        help='acc_step') 


def save_config(args, path):
    with open(path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_config(path):
    args = argparse.Namespace()
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if value.isdigit():
                if value.find('.') != -1:
                    value = float(value)
                else:
                    value = int(value)
            setattr(args, key, value)
    print(args)
    return args
