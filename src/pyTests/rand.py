


# np.random.seed(1337)
# def build_TestNumpyInterface(seed, num, ext):
#     rsg = np.random.RandomState(seed)
#     mnkls = rsg.randint(1, 20, (num,4))

#     def make_test_to_ndarray(mn, kl):
#         def test(self):
#             ls = rsg.randn(ext, *mn)
#             rs = rsg.randn(ext, *kl)
#             for l,r in zip(ls,rs):
#                 lr = np.kron(l,r)
#                 L,R = project(lr, mn, kl)
#                 diff = norm(lr - np.kron(L,R))
#                 self.assertLessEqual(diff, 1e-12)
#         return test

#     def make_test_from_ndarray(mn, kl):
#         def test(self):
#             ls = rsg.randn(ext, *mn)
#             rs = rsg.randn(ext, *kl)
#             for l,r in zip(ls,rs):
#                 lr = np.kron(l,r)
#                 L,R = project(lr, mn, kl)
#                 diff = norm(lr - np.kron(L,R))
#                 self.assertLessEqual(diff, 1e-12)
#         return test

#     odir = dict()
#     for mnkl in mnkls:
#         name = "test_random_{}x{}_{}x{}".format(*mnkl)
#         test = make_test(mnkl[:2], mnkl[2:])
#         odir[name] = test

#     return type("TestRandomKroneckerSplitting", (unittest.TestCase,), odir)
# TestRandomKroneckerSplitting = build_TestNumpyInterface(0, 100, 4)
