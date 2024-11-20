import random

def generate_instance(n: int):
    random.seed(0)
    # U
    U:list = [random.randint(0,100)for _ in range(n)]
    # C
    C:list = [[random.randint(0,1)for _ in range(n)]for _ in range(n)]
    
    with open(f"pco_{n}.txt", "+w") as f:
        f.write("N\n")
        f.write(f"{n}\n")
        f.write("U\n")
        for i in range(n):
            f.write(f"{i};{U[i]}\n")
        
        f.write("C\n")
        for i in range(n):
            for j in range(n):
                f.write(f"{i},{j};{C[i][j]}\n")
    return 


if __name__ == '__main__':
    generate_instance(3)