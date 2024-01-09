import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=7, type=int, help="seed")
    parser.add_argument("--use_cuda_if_available", default=True, type=bool, help="Use GPU")
    
    parser.add_argument("--data_dir", default="../../data", type=str, help="")
    
    parser.add_argument("--output_dir", default="./outputs/", type=str, help="")
    
    parser.add_argument("--batch_size", default=16, type=int, help="")
    parser.add_argument("--emb_dim", default=100, type=int, help="")
    parser.add_argument("--hidden_dim", default=128, type=int, help="")
    parser.add_argument("--n_layers", default=2, type=int, help="")
    parser.add_argument("--n_heads", default=8, type=int, help="")

    parser.add_argument("--seq_len", default=32, type=int, help="")
    
    parser.add_argument("--n_epochs", default=20, type=int, help="")
    parser.add_argument("--lr", default=0.001, type=float, help="")
    parser.add_argument("--model_dir", default="./models/", type=str, help="")
    parser.add_argument("--model_name", default="best_model.pt", type=str, help="")

    parser.add_argument("--verbose", default=False, type=bool, help="")
    

    args = parser.parse_args()

    return args
