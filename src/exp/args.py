from argparse import ArgumentParser

def get_parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--d_model', type=int, default=32, help='Dimension of model embeddings')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers (0 for no LSTM)')
    parser.add_argument('--bidirectional', type=int, default=0, help='Use bidirectional LSTM if set')
    parser.add_argument('--model_name', type=str, default='STAND', help='Name of the model to use')
    parser.add_argument('--task_name', type=str, default='unsupervised', help='Task type: supervised or unsupervised', 
                        choices=['supervised', 'unsupervised', 'semisupervised'])
    parser.add_argument('--win_size', type=int, default=32, help='Sliding window size')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    return args