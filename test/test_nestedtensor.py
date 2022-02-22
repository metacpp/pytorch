# Owner(s): ["module: nestedtensor"]

import torch
from torch.testing._internal.common_utils import TestCase, load_tests
from torch import nested_tensor

# Tests are ported from pytorch/nestedtensor.
# This makes porting as_nested_tensor easier in the future.
def _iter_constructors():
    # yield as_nested_tensor
    yield nested_tensor

class TestNestedTensor(TestCase):

    def test_unbind(self):

        def _test_fn(unbind_fn):
            def _test(a, b, c, d, e):
                nt = nested_tensor([a, b])
                a1, b1 = nt.unbind()
                self.assertTrue(a is not a1)
                self.assertTrue(b is not b1)

                nt = nested_tensor([a, b])
                a1, b1 = unbind_fn(nt, 0)
                self.assertEqual(a, a1)
                self.assertEqual(b, b1)

                a = utils.gen_float_tensor(1, (2, 3)).add_(1)
                nt = nested_tensor([a])
                self.assertEqual(a, unbind_fn(nt, 0)[0])

            _test(torch.tensor([1, 2]),
                  torch.tensor([7, 8]),
                  torch.tensor([3, 4]),
                  torch.tensor([5, 6]),
                  torch.tensor([6, 7]))
            _test(torch.tensor([1]),
                  torch.tensor([7]),
                  torch.tensor([3]),
                  torch.tensor([5]),
                  torch.tensor([6]))
            _test(torch.tensor(1),
                  torch.tensor(7),
                  torch.tensor(3),
                  torch.tensor(5),
                  torch.tensor(6))
            _test(torch.tensor([]),
                  torch.tensor([]),
                  torch.tensor([]),
                  torch.tensor([]),
                  torch.tensor([]))

        # Both of these tests are necessary, because we're using
        # torch_function.
        _test_fn(lambda x, dim: x.unbind(dim))
        _test_fn(lambda x, dim: torch.unbind(x, dim))

    def test_unbind_dim(self):
        # Unbinding across tensors dimensions
        # is akin splitting up the tree across a level.

        def _test_fn(unbind_fn):

            a = torch.rand(3, 2)
            nt = nested_tensor([a])
            result = (
                nested_tensor([unbind_fn(a, 0)[0]]),
                nested_tensor([unbind_fn(a, 0)[1]]),
                nested_tensor([unbind_fn(a, 0)[2]]))
            for x, y in zip(unbind_fn(nt, 1), result):
                self.assertEqual(x, y, ignore_contiguity=True)
            result = (
                nested_tensor([unbind_fn(a, 1)[0]]),
                nested_tensor([unbind_fn(a, 1)[1]]))
            for x, y in zip(unbind_fn(nt, 2), result):
                self.assertEqual(x, y, ignore_contiguity=True)

            b = torch.rand(2, 3)
            nt = nested_tensor([a, b])
            self.assertEqual(unbind_fn(nt, 0), (a, b))
            result = (
                nested_tensor(
                    [unbind_fn(a, 0)[0], unbind_fn(b, 0)[0]]),
                nested_tensor(
                    [unbind_fn(a, 0)[1], unbind_fn(b, 0)[1]]),
                nested_tensor([unbind_fn(a, 0)[2]]))
            for x, y in zip(unbind_fn(nt, 1), result):
                self.assertEqual(x, y, ignore_contiguity=True)

        # Both of these tests are necessary, because we're using
        # torch_function.
        _test_fn(lambda x, dim: x.unbind(dim))
        _test_fn(lambda x, dim: torch.unbind(x, dim))

    def test_nested_tensor(self):
        self.assertRaises(
            RuntimeError, lambda: nested_tensor([3.0]))
        self.assertRaises(
            TypeError, lambda: nested_tensor(torch.tensor([3.0])))
        for nested_tensor2 in _iter_nested_tensors():
            nested_tensor(
                nested_tensor2([torch.tensor([3.0])]))
        for nested_tensor2 in _iter_nested_tensors():
            self.assertRaises(RuntimeError, lambda: nested_tensor(
                [torch.tensor([2.0]), nested_tensor2([torch.tensor([3.0])])]))
        self.assertRaises(TypeError, lambda: nested_tensor(4.0))

    def test_default_nested_tensor(self):
        self.assertRaises(TypeError, lambda: nested_tensor())
        default_nested_tensor = nested_tensor([])
        default_tensor = torch.tensor([])
        self.assertEqual(default_nested_tensor.nested_dim(), 1)
        self.assertEqual(default_nested_tensor.nested_size().unbind(), [])
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        self.assertEqual(default_nested_tensor.layout,
                         default_tensor.layout)
        self.assertEqual(default_nested_tensor.device,
                         default_tensor.device)
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        self.assertEqual(default_nested_tensor.requires_grad,
                         default_tensor.requires_grad)
        self.assertIsNone(default_tensor.grad)
        self.assertEqual(default_nested_tensor.is_pinned(),
                         default_tensor.is_pinned())

    def test_element_size(self):
        for constructor in _iter_constructors():
            nt1 = constructor([])
            self.assertEqual(nt1.element_size(), torch.randn(1).element_size())
            a = torch.randn(4).int()
            nt2 = constructor([a])
            self.assertEqual(a.element_size(), nt2.element_size())

    def test_nested_size(self):
        for constructor in _iter_constructors():
            a = constructor([])
            self.assertEqual(len(a.nested_size()), 0)
            self.assertRaises(RuntimeError, lambda: a.nested_size()[0])

            a = constructor([torch.tensor(1)])
            self.assertEqual(len(a.nested_size()), 1)
            self.assertEqual(a.nested_size()[0], torch.Size([]))
            self.assertEqual(a.nested_size(0), 1)
            self.assertRaises(IndexError, lambda: a.nested_size(1))

            a = constructor([torch.randn(1)])
            self.assertEqual(a.nested_size()[0], torch.Size([1]))
            self.assertEqual(a.nested_size()[0][0], 1)
            self.assertEqual(a.nested_size(0), 1)
            self.assertEqual(a.nested_size(1), (1,))
            self.assertRaises(IndexError, lambda: a.nested_size(2))

            a = constructor([torch.randn(1, 2)])
            self.assertEqual(a.nested_size()[0], torch.Size([1, 2]))
            self.assertEqual(a.nested_size(0), 1)
            self.assertEqual(a.nested_size(1), (1,))
            self.assertEqual(a.nested_size(2), (2,))
            self.assertRaises(IndexError, lambda: a.nested_size(3))

            # Make sure object is not bound to life-time of NestedTensor instance
            b = a.nested_size()
            del a
            self.assertEqual(len(b), 1)
            self.assertEqual(b[0], torch.Size([1, 2]))
            self.assertEqual(b[0][0], 1)
            self.assertEqual(b[0][1], 2)

    @torch.inference_mode()
    def test_nested_stride(self):
        for constructor in _iter_constructors():
            tensors = [torch.rand(1, 2, 4)[:, :, 0], torch.rand(
                2, 3, 4)[:, 1, :], torch.rand(3, 4, 5)[1, :, :]]
            a = constructor(tensors)
            na = list(list(t.contiguous().stride()) for t in tensors)
            ans = a.nested_stride()
            result = tuple(ans[i] for i in range(len(ans)))
            for r, s in zip(result, na):
                self.assertEqual(r, s)

    def test_len(self):
        for constructor in _iter_constructors():
            a = constructor([torch.tensor([1, 2]),
                             torch.tensor([3, 4]),
                             torch.tensor([5, 6]),
                             torch.tensor([7, 8])])
            self.assertEqual(len(a), 4)
            a = constructor([torch.tensor([1, 2]),
                             torch.tensor([7, 8])])
            self.assertEqual(len(a), 2)
            a = constructor([torch.tensor([1, 2])])
            self.assertEqual(len(a), 1)

    def test_equal(self):
        for constructor in _iter_constructors():
            a1 = constructor([torch.tensor([1, 2]),
                              torch.tensor([7, 8])])
            a2 = constructor([torch.tensor([1, 2]),
                              torch.tensor([7, 8])])
            a3 = constructor([torch.tensor([3, 4]),
                              torch.tensor([5, 6])])
            self.assertTrue((a1 == a2).all())
            self.assertTrue((a1 != a3).all())
            self.assertTrue(not (a1 != a2).any())
            self.assertTrue(not (a1 == a3).any())

            a1 = constructor([torch.tensor([1, 2]),
                              torch.tensor([2, 8])])
            a2 = constructor([torch.tensor([0, 1]),
                              torch.tensor([1, 0])], dtype=torch.bool)
            a3 = constructor([torch.tensor([1, 0]),
                              torch.tensor([0, 1])], dtype=torch.bool)
            self.assertEqual((a1 == 2), a2)
            self.assertEqual((a1 != 2), a3)
            self.assertEqual((a1 == 2.0), a2)
            self.assertEqual((a1 != 2.0), a3)

    def test_dim(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor(3.)])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor([1, 2, 3, 4])])
            self.assertEqual(a1.dim(), 2)

#    def test_repr_string(self):
#        a = constructor(
#            [
#            ])
#        expected = "nested_tensor(["\
#                   "\n\n])"
#        self.assertEqual(str(a), expected)
#        self.assertEqual(repr(a), expected)
#
#        a = constructor(
#            [
#                torch.tensor(1),
#            ])
#        expected = "nested_tensor(["\
#                   "\n\ttensor(1)"\
#                   "\n])"
#        self.assertEqual(str(a), expected)
#        self.assertEqual(repr(a), expected)
#        # str(a)
#        # repr(a)
#
#        a = constructor(
#            [
#                torch.tensor([[1, 2]]),
#                torch.tensor([[4, 5]]),
#            ])
#        expected = "nested_tensor(["\
#                   "\n\ttensor([[1, 2]])"\
#                   ","\
#                   "\n\ttensor([[4, 5]])"\
#                   "\n])"
#        self.assertEqual(str(a), expected)
#        self.assertEqual(repr(a), expected)
#        # str(a)
#        # repr(a)
