from ik import get_ik
import numpy as np
from numpy import equal, all

def main():
    ee_poses = [np.array([274, 0, 204, 0]),
               np.array([16, 4, 336, 15]),
                np.array([0, -270, 106, 0])]
    
    solutions =  [[0, 0, 0, 0],
                   [15, -45, -60, 90],
                    [-90, 15, 30, -45]]
    
    for e, s in zip(ee_poses, solutions):
        func_out = get_ik(e)
        print("\nexpected output:", s)
        print(f"get_ik output: {func_out}")
        #assert(all(equal(solution, func_out )))

if __name__ == '__main__':
    main()