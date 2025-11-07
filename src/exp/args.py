from argparse import ArgumentParser

METRIC_LIST = ['CCE', 'F1', 'Aff-F1', 'UAff-F1', 'AUC-ROC', 'VUS-PR']


def get_parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--d_model', type=int, default=32, help='Dimension of model embeddings')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers (0 for no LSTM)')
    parser.add_argument('--bidirectional', type=int, default=0, help='Use bidirectional LSTM if set')
    parser.add_argument('--model_name', type=str, default='STAND', help='Name of the model to use')
    parser.add_argument('--task_name', type=str, default='unsupervised', help='Task type: supervised or unsupervised', 
                        choices=['supervised', 'unsupervised', 'semisupervised', 'supervised_limit'])
    parser.add_argument('--win_size', type=int, default=3, help='Sliding window size')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for sliding window')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model_saving_path', type=str, default='/home/zzj/projects/STAND/model_saving', help='Path to save the trained model')
    # parser.add_argument('--model_saving_path', type=str, default='/share/home/202220143416/project/STAND/model_saving', help='Path to save the trained model')
    # parser.add_argument('--model_saving_path', type=str, default='/public/home/202220143416/projects/STAND/model_saving', help='Path to save the trained model')
    parser.add_argument('--dataset_path', type=str, default='/home/zzj/projects/FTSAD/datasets', help='Path to the dataset')
    # parser.add_argument('--dataset_path', type=str, default='/share/home/202220143416/project/FTSAD/datasets', help='Path to the dataset')
    # parser.add_argument('--dataset_path', type=str, default='/public/home/202220143416/projects/FTSAD/datasets', help='Path to the dataset')
    parser.add_argument('--dataset_name', type=str, default='PSM', help='Name of the dataset to use')
    parser.add_argument('--if_save', type=int, default=0, help='Whether to save the trained model (1 for yes, 0 for no)')
    parser.add_argument('--index', type=int, default=0, help='Index of the dataset to use')

    # Dataset parameter
    parser.add_argument('--train_test_split', type=float, default=0.5, help='Train-test split ratio')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1, help='Ratio of anomalies in the dataset')
    parser.add_argument('--train_split_max', type=float, default=0.3, help='Maximum ratio of training data in supervised learning')
    parser.add_argument('--metric_list', type=str, nargs='+', default=METRIC_LIST, help='List of metrics to evaluate')
    parser.add_argument('--quantile', type=float, default=0.95, help='Quantile for thresholding in anomaly detection')

    args = parser.parse_args()
    return args