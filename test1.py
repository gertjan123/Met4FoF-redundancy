import itertools
import numpy as np
#from math import comb
import redundancy1

from scipy.special import comb
#print(comb(10,3, exact=True))

redundancy1.test_calc_best_estimate()
redundancy1.test_calc_lcs()

if False:
  c = np.array(['a', 'b', 'c'])
  Vy = np.array([[1,2,3,4],[5,6,7,8], [9,10,11,12],[13,14,15,16]])
  print(c)
  print(Vy)
  subsets = itertools.combinations(c, 2)
  for s in subsets:
    #  s2 = list(s)
    #  z = c[s2]
    #  v = Vy[np.ix_(s2, s2)]
      print(s)
    #  print(z)
    #  print(v)



