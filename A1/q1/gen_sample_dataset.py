import random
import sys
from collections import defaultdict


def generate_dataset(universal_itemset: int, num_transactions: int, output_file: str = "generated_transactions.dat"):
    random.seed(42)
    
    # To keep all items frequent up to 90%,
    target_frequency = 0.9
    # Calculate average sample size and variation
    avg_sample_size = int(universal_itemset * target_frequency)
    sample_variation = max(1, int(universal_itemset * 0.15))
    
    print(f"  Sample size: {avg_sample_size} ± {sample_variation} items per transaction")
    
    transactions = []
    
    # Generate given number of transactions
    for _ in range(num_transactions):
        # Randomly vary sample size within defined variations
        sample_size = random.randint(
            avg_sample_size - sample_variation,
            avg_sample_size + sample_variation
        )
        # Safety size check
        sample_size = max(1, min(sample_size, universal_itemset))
        
        # Random sampling from universal itemset
        transaction = random.sample(range(universal_itemset), sample_size)
        transactions.append(sorted(transaction))
    
    # Write to file
    with open(output_file, 'w') as f:
        for transaction in transactions:
            f.write(' '.join(map(str, transaction)) + '\n')

def main():
    if len(sys.argv) < 3:
        print("\nUsage: python gen_data.py <universal_itemset> <num_transactions>")
        print("\nParameters:")
        print("  <universal_itemset>  Number of distinct items (given: 25-40)")
        print("  <num_transactions>   Number of transactions (given: 15000)")
        print("\nExample usage:")
        print("  python gen_sample_dataset.py 30 15000")
        sys.exit(1)
    
    universal_itemset = int(sys.argv[1])
    num_transactions = int(sys.argv[2])
    
    if universal_itemset <= 0 or num_transactions <= 0:
        print("Arguments must be positive integers")
        sys.exit(1)
    
    generate_dataset(universal_itemset, num_transactions)

if __name__ == "__main__":
    main()