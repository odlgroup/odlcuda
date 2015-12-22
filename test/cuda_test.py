""" Simple tests for odlpp.cuda

This is mostly compilation tests, main suite is in odl
"""

import odlpp.odlpp_cuda as cuda
import pytest

vec_ids = [str(el) for el in dir(cuda) if el.startswith('CudaVector')]
vec_params = [getattr(cuda, vec_id) for vec_id in vec_ids]
vector_type_fixture = pytest.fixture(scope="module", ids=vec_ids,
                                     params=vec_params)


@vector_type_fixture
def vector_type(request):
    return request.param


def test_has_vectors():
    # Assert at least 1 vector exists
    assert len(vec_ids) > 0


def test_vector_init(vector_type):
    vec = vector_type(1)
    assert vec is not None


def test_get_set(vector_type):
    vec = vector_type(1)
    vec[0] = 3
    assert vec[0] == 3


def test_slice(vector_type):
    vec = vector_type(3)
    vec[0] = 1
    vec[1] = 2
    vec[2] = 3
    result = vec.getslice(slice(None, None, None))
    assert [result[0], result[1], result[2]] == [1, 2, 3]

    reverse = vec.getslice(slice(None, None, -1))
    print(reverse)
    assert [reverse[0], reverse[1], reverse[2]] == [3, 2, 1]


def test_slice_host(vector_type):
    vec = vector_type(3)
    vec[0] = 1
    vec[1] = 2
    vec[2] = 3
    result = vec.get_to_host(slice(None, None, None))
    assert [result[0], result[1], result[2]] == [1, 2, 3]

    reverse = vec.get_to_host(slice(None, None, -1))
    print(reverse)
    assert [reverse[0], reverse[1], reverse[2]] == [3, 2, 1]


def test_sum(vector_type):
    vec = cuda.CudaVectorFloat32(3)
    vec[0] = 1
    vec[1] = 2
    vec[2] = 3
    assert vec.sum() == 6


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
