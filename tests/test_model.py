import torch
import pytest
from src.clearsky_sentinel import ClearSkySentinel, load_clearsky_weights


@pytest.fixture
def small_model():
    return ClearSkySentinel(features=16, blocks=2)

@pytest.fixture
def dummy_input():
    return torch.randn(2, 15, 32, 32)

def test_forward_output_shape(small_model, dummy_input):
    out = small_model(dummy_input)
    assert out.shape == (2, 13, 32, 32)   #13 band output
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "bad_shape",
    [
        pytest.param((2, 13, 32, 32), id='wrong_channel_count'),
        pytest.param((15, 32, 32), id='3d_instead_of_4d'),
        pytest.param((2, 15, 32, 32, 1), id="5d_instead_of_4d"),
        pytest.param((2, 0, 32, 32), id="zero_channels"),

    ]
)
def test_invalid_raises(small_model, bad_shape):
    bad_x = torch.randn(bad_shape)
    with pytest.raises(ValueError):
        small_model(bad_x)


def test_gradient_flow(small_model, dummy_input):
    loss = small_model(dummy_input).mean()
    loss.backward()
    for name, param in small_model.named_parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cpu_cuda_consistency(small_model, dummy_input):
    small_model.eval()

    small_model.to('cpu')
    cpu_input = dummy_input.to('cpu')
    with torch.no_grad():
        cpu_out = small_model(cpu_input)

    small_model.to('cuda')
    cuda_input = dummy_input.to('cuda')
    with torch.no_grad():
        cuda_out = small_model(cuda_input)

    torch.testing.assert_close(cuda_out.cpu(), cpu_out, rtol=1e-3, atol=1e-3)


def test_load_clearsky_weights(small_model, dummy_input, tmp_path):
    ckpt_path = tmp_path / "ckpt.pt"
    ddp_state_dict = {
        "state_dict": {f"module.{k}": v for k, v in small_model.state_dict().items()}
    }
    torch.save(ddp_state_dict, ckpt_path)

    loaded_model = ClearSkySentinel(features=16, blocks=2)
    load_clearsky_weights(loaded_model, ckpt_path)

    small_model.eval()
    loaded_model.eval()

    with torch.no_grad():
        expected_output = small_model(dummy_input)
        actual_output = loaded_model(dummy_input)

    torch.testing.assert_close(actual_output, expected_output)

    lfs_path = tmp_path / 'lfs.pt'
    lfs_path.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")

    fresh_model = ClearSkySentinel(features=16, blocks=2)
    with pytest.raises(ValueError):
        load_clearsky_weights(fresh_model, lfs_path)