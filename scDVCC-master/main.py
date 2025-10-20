import torch
from src.utils import set_seed
from src.argument import parse_args
from models.model import model_Trainer
import optuna



def main():
    set_seed()
    args, _ = parse_args()
    torch.set_num_threads(3)
    embedder = model_Trainer(args)
    embedder.train()

if __name__ == "__main__":
    main()