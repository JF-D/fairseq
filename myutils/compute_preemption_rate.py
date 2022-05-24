import random
import math

# This function implies: my initial setting of preemption is over approximation!!!
# > prob: 0.05, preem_iters: 5
#   hourly preemption rate 0.111
#
# > prob: 0.05, preem_iters: 10
#   hourly preemption rate 0.201
#
# > prob: 0.1, preem_iters: 5
#   hourly preemption rate 0.200
#
# > prob: 0.1, preem_iters: 10
#   hourly preemption rate 0.334
def compute_an_hour(nnodes, iter_time, prob=0.05, preem_iters=5):
    niters = math.floor(3600 / iter_time)

    cnt = 0
    for n in range(nnodes):
        start = 0
        while start < niters:
            start += 1
            if random.random() < prob:
                p_iters = random.randint(0, preem_iters)
                start += p_iters
                cnt += p_iters
    print(f'> prob: {prob}, preem_iters: {preem_iters}')
    print(f'  hourly preemption rate {cnt / (niters * nnodes):.3f}')
    print()

if __name__ == '__main__':
    compute_an_hour(32, 0.2, prob=0.05, preem_iters=5)
    compute_an_hour(32, 0.2, prob=0.05, preem_iters=10)
    compute_an_hour(32, 0.2, prob=0.1, preem_iters=5)
    compute_an_hour(32, 0.2, prob=0.1, preem_iters=10)
