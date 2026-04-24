import train_B
import train_A
import argparse
import os

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a model for the Kerberos challenge.")
    argparser.add_argument("--dataset",    required=True, choices=["A", "B", "C"])
    argparser.add_argument("--task",       required=True, choices=["node", "link"], help="Task type: node classification (A/B) or link prediction (C)")
    argparser.add_argument("--data_dir",   required=True, help="Absolute path to the shared datasets directory")
    argparser.add_argument("--model_dir", required=False, default=None, help="Directory to save the trained model.")
    argparser.add_argument("--kerberos",   required=True, help="Your Kerberos ID (used to name the output file)")
    args = argparser.parse_args()
    if not os.path.isabs(args.data_dir):
        argparser.error("--data_dir must be an absolute path")
    
    if args.dataset == "A":
        train_A.train(args.data_dir, args.model_dir, args.kerberos)
    elif args.dataset == "B":
        train_B.train(args.data_dir, args.model_dir, args.kerberos)
    else:
        raise NotImplementedError(f"Training not implemented for dataset {args.dataset}")