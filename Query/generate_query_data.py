import pandas as pd
import numpy as np
import random
import string


def random_string(min_length, max_length):
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def random_user_age():
    return max(0, 42 + int(np.random.normal() * 20))


def generate_user(id):
    return {'id': id, 'userName': random_string(5, 15), 'userAge': random_user_age(), 'termsAccepted': random.choice([True, False])}


def main():
    num_users = 10000000

    users = [generate_user(id) for id in range(num_users)]
    users_df = pd.DataFrame(users)

    users_df.to_csv("query/data.csv", index=False, header=True)


main()
