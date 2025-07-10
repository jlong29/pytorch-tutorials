import torch

def test_cuda_tensor_ops(verbose=True):
    """
    Tests basic tensor operations on all available CUDA devices.

    Returns:
        results (dict): Mapping from device index to success/failure and message.
    """
    if not torch.cuda.is_available():
        return {"cuda": "CUDA not available on this system."}

    results = {}

    num_devices = torch.cuda.device_count()

    for idx in range(num_devices):
        device = torch.device(f"cuda:{idx}")
        torch.cuda.set_device(device)
        try:
            # Create tensors in different ways
            x = torch.tensor([1., 2.], device=device)
            y = torch.tensor([3., 4.]).to(device)
            z = torch.tensor([5., 6.]).cuda(idx)

            # Perform operations
            a = x + y
            b = y + z
            c = x + z

            # Allocate within context (redundant but shown)
            with torch.cuda.device(idx):
                d = torch.randn(2, device=device)
                e = torch.randn(2).to(device)
                f = torch.randn(2).cuda(idx)

                _ = d + e + f

            results[f"cuda:{idx}"] = "✅ Success"
            if verbose:
                print(f"[cuda:{idx}] Passed tensor creation and addition test.")

        except Exception as e:
            results[f"cuda:{idx}"] = f"❌ Failed: {str(e)}"
            if verbose:
                print(f"[cuda:{idx}] Failed: {str(e)}")

    return results
