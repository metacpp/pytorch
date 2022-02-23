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
                print("a1: ", a1)
                print("a: ", a)
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
        return

        # Both of these tests are necessary, because we're using
        # torch_function.
        _test_fn(lambda x, dim: x.unbind(dim))
        _test_fn(lambda x, dim: torch.unbind(x, dim))

    def test_unbind_dim(self):

        def _test_fn(unbind_fn):
            a = torch.rand(3, 2)
            b = torch.rand(2, 3)
            nt = nested_tensor([a, b])
            self.assertRaises(RuntimeError,
                    lambda: unbind_fn(nt, 1))

        # Both of these tests are necessary, because we're using
        # torch_function.
        _test_fn(lambda x, dim: x.unbind(dim))
        _test_fn(lambda x, dim: torch.unbind(x, dim))

    def test_nested_tensor(self):
        self.assertRaises(
            TypeError, lambda: nested_tensor([3.0]))
        self.assertRaises(
            TypeError, lambda: nested_tensor(torch.tensor([3.0])))
        self.assertRaises(TypeError, lambda: nested_tensor(4.0))

    def test_default_nested_tensor(self):
        self.assertRaises(TypeError, lambda: nested_tensor())
        default_nested_tensor = nested_tensor([])
        default_tensor = torch.tensor([])
        self.assertEqual(default_nested_tensor.nested_dim(), 1)
        # self.assertEqual(default_nested_tensor.nested_size(), ())
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

    def test_dim(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor(3.)])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor([1, 2, 3, 4])])
            self.assertEqual(a1.dim(), 2)

    def test_numel(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(RuntimeError, "numel is disabled", lambda: a1.numel(),)

    def test_size(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(RuntimeError, "NestedTensorImpl doesn't support sizes", lambda: a1.size())

    def test_stride(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(RuntimeError, "NestedTensorImpl doesn't support strides", lambda: a1.stride())

    def test_is_contiguous(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(RuntimeError, "is_contiguous is disabled", lambda: a1.is_contiguous())

    def test_repr_string(self):
        a = nested_tensor(
            [
            ])
        expected = "nested_tensor(["\
                   "\n\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = nested_tensor(
            [
                torch.tensor(1),
            ])
        expected = "nested_tensor(["\
                   "\n  tensor(1.)"\
                   "\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = nested_tensor(
            [
                torch.tensor([[1, 2]]),
                torch.tensor([[4, 5]]),
            ])
        expected = "nested_tensor(["\
                   "\n  tensor([[1., 2.]])"\
                   ","\
                   "\n  tensor([[4., 5.]])"\
                   "\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)
