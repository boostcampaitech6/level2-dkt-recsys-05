import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--use_cuda_if_available", type=bool, help="Use GPU")
    parser.add_argument("--data_dir", type=str, help="")
    parser.add_argument("--output_dir", type=str, help="")
    parser.add_argument("--model_dir", type=str, help="")
    parser.add_argument("--model_name", type=str, help="")
    
    parser.add_argument("--batch_size", type=int, help="")
    parser.add_argument("--emb_size", type=int, help="")
    parser.add_argument("--hidden_size", type=int, help="")
    parser.add_argument("--n_layers", type=int, help="")
    parser.add_argument("--n_head", type=int, help="")
    parser.add_argument("--seq_len", type=int, help="")
    parser.add_argument("--n_epochs", type=int, help="")
    parser.add_argument("--lr", type=float, help="")
    parser.add_argument("--dropout", type=float, help="")

    parser.add_argument("--verbose", type=bool, help="")
    

    args = parser.parse_args()

    return args
