import random

def gaussian(mu, sigma):
    return random.gauss(mu, sigma)

def generate_random_centers(k, num_dimensions):
    min_center, max_center = -10, 10
    min_std, max_std = 1, 3
    return [
        [
            (random.uniform(min_center, max_center), random.uniform(min_std, max_std))
            for _ in range(num_dimensions)
        ]
        for _ in range(k)
    ]

def to_file(features):
    return ' '.join(map(str, features))

def main():
    output_dir = "KMeans_clustering/data.csv"
    num_samples = 100000
    num_dimensions = 20
    k = 5

    centers = generate_random_centers(k, num_dimensions)

    with open(output_dir, "w") as file:
        for i in range(num_samples):
            designated_center = random.randint(0, k - 1)
            center_characteristics = centers[designated_center]
            features = [gaussian(mu, sigma) for mu, sigma in center_characteristics]
            line = to_file(features)
            file.write(line + "\n")
    
if __name__ == "__main__":
    main()
