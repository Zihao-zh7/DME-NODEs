import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--remote', action='store_true', help='the code run on a server')
parser.add_argument('--num-gpu', type=int, default=0, help='the number of the gpu to use')
parser.add_argument('--epochs', type=int, default=200, help='train epochs')
parser.add_argument('--batch-size', type=int, default=4, help='batch size')
parser.add_argument('--filename', type=str, default='pems04')
parser.add_argument('--train-ratio', type=float, default=0.6, help='the ratio of training dataset')
parser.add_argument('--valid-ratio', type=float, default=0.2, help='the ratio of validating dataset')
parser.add_argument('--his-length', type=int, default=12, help='the length of history time series of input')
parser.add_argument('--pred-length', type=int, default=3, help='the length of target time series for prediction')
parser.add_argument('--sigma1', type=float, default=0.1, help='sigma for the semantic matrix')
parser.add_argument('--sigma2', type=float, default=10, help='sigma for the spatial matrix')
parser.add_argument('--thres1', type=float, default=0.6, help='the threshold for the semantic matrix')
parser.add_argument('--thres2', type=float, default=0.5, help='the threshold for the spatial matrix')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')
parser.add_argument('--step_size', type=int, default=10, help='StepLR scheduler step size')
parser.add_argument('--gamma', type=float, default=0.9, help='StepLR scheduler gamma')
parser.add_argument('--log', action='store_true', help='if write log to files')
parser.add_argument('--loss-alpha', type=float, default=0.7,
                    help='weight for MAE in mixed loss (alpha*MAE + (1-alpha)*SmoothL1)')
parser.add_argument('--smoothl1-beta', type=float, default=1.0,
                    help='beta parameter for SmoothL1 loss')
parser.add_argument('--solver', type=str, default='rk4', choices=['rk4', 'euler'],
                    help='ODE solver type: rk4 (Runge-Kutta) or euler (exponential Euler)')
args = parser.parse_args()
