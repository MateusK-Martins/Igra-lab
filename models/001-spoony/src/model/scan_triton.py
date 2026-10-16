import torch
import triton
import triton.language as tl


@triton.jit
def combine(a_left, b_left, a_right, b_right):
    a_combined = a_left * a_right
    b_combined = b_right + a_right * b_left

    return a_combined, b_combined


@triton.jit
def affine_scan_kernel(
    update_ptr,
    decay_ptr,
    output_ptr,
    initial_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
):
    batch = tl.program_id(0)
    feature_block = tl.program_id(1)

    times = tl.arange(0, BLOCK_T)
    features = feature_block * BLOCK_D + tl.arange(0, BLOCK_D)

    offsets = batch * T * D + times[:, None] * D + features[None, :]

    valid = (times[:, None] < T) & (features[None, :] < D)

    update = tl.load(update_ptr + offsets, mask=valid, other=0.0)
    decay = tl.load(decay_ptr + offsets, mask=valid, other=1.0)

    update = update.to(tl.float32)
    decay = decay.to(tl.float32)

    retention, states = tl.associative_scan(
        (decay, update),
        axis=0,
        combine_fn=combine,
    )

    if HAS_INITIAL:
        initial = tl.load(
            initial_ptr + batch * D + features,
            mask=features < D,
            other=0.0,
        ).to(tl.float32)

        states = states + retention * initial[None, :]

    tl.store(output_ptr + offsets, states, mask=valid)


@triton.jit
def affine_scan_backward_kernel(
    grad_output_ptr,
    decay_ptr,
    states_ptr,
    initial_ptr,
    grad_update_ptr,
    grad_decay_ptr,
    grad_initial_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
):
    batch = tl.program_id(0)
    feature_block = tl.program_id(1)

    times = tl.arange(0, BLOCK_T)
    features = feature_block * BLOCK_D + tl.arange(0, BLOCK_D)

    offsets = batch * T * D + times[:, None] * D + features[None, :]
    valid = (times[:, None] < T) & (features[None, :] < D)

    grad_output = tl.load(
        grad_output_ptr + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    next_decay = tl.load(
        decay_ptr + offsets + D,
        mask=(times[:, None] + 1 < T) & (features[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    _, grad_state = tl.associative_scan(
        (next_decay, grad_output),
        axis=0,
        combine_fn=combine,
        reverse=True,
    )

    tl.store(grad_update_ptr + offsets, grad_state, mask=valid)

    previous_state = tl.load(
        states_ptr + offsets - D,
        mask=valid & (times[:, None] > 0),
        other=0.0,
    ).to(tl.float32)

    if HAS_INITIAL:
        initial = tl.load(
            initial_ptr + batch * D + features,
            mask=features < D,
            other=0.0,
        ).to(tl.float32)

        previous_state = tl.where(
            times[:, None] == 0,
            initial[None, :],
            previous_state,
        )

    grad_decay = grad_state * previous_state
    tl.store(grad_decay_ptr + offsets, grad_decay, mask=valid)

    if HAS_INITIAL:
        first_decay = tl.load(
            decay_ptr + batch * T * D + features,
            mask=features < D,
            other=0.0,
        ).to(tl.float32)

        first_grad_state = tl.sum(
            tl.where(times[:, None] == 0, grad_state, 0.0),
            axis=0,
        )

        tl.store(
            grad_initial_ptr + batch * D + features,
            first_decay * first_grad_state,
            mask=features < D,
        )


def _affine_prefix_sum_forward(
    update: torch.Tensor,
    decay: torch.Tensor,
    initial: torch.Tensor | None = None,
) -> torch.Tensor:
    if update.ndim != 3 or update.shape != decay.shape:
        raise ValueError("update and decay must have matching [B, T, D] shapes")
    if not update.is_cuda or update.device != decay.device:
        raise ValueError("update and decay must be on the same CUDA device")
    if not update.is_floating_point() or not decay.is_floating_point():
        raise ValueError("update and decay must be floating-point tensors")
    if not update.is_contiguous() or not decay.is_contiguous():
        raise ValueError("update and decay must be contiguous")

    batch_size, sequence_length, features = update.shape

    if initial is not None:
        if initial.shape != (batch_size, 1, features):
            raise ValueError("initial must have shape [B, 1, D]")
        if initial.device != update.device:
            raise ValueError("initial must be on the same device as update")
        if not initial.is_floating_point() or not initial.is_contiguous():
            raise ValueError("initial must be floating-point and contiguous")

    output = torch.empty(
        update.shape,
        device=update.device,
        dtype=torch.float32,
    )

    if output.numel() == 0:
        return output

    block_features = 16
    grid = (batch_size, triton.cdiv(features, block_features))

    with torch.cuda.device(update.device):
        affine_scan_kernel[grid](
            update,
            decay,
            output,
            initial if initial is not None else update,
            T=sequence_length,
            D=features,
            BLOCK_T=triton.next_power_of_2(sequence_length),
            BLOCK_D=block_features,
            HAS_INITIAL=initial is not None,
        )

    return output


class AffineScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, update, decay, initial):
        output = _affine_prefix_sum_forward(update, decay, initial)

        ctx.has_initial = initial is not None
        ctx.update_dtype = update.dtype
        ctx.decay_dtype = decay.dtype

        if initial is None:
            ctx.save_for_backward(decay, output)
        else:
            ctx.save_for_backward(decay, output, initial)

        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        decay, states = saved[:2]
        initial = saved[2] if ctx.has_initial else None

        grad_output = grad_output.contiguous()
        grad_update = torch.empty_like(states)
        grad_decay = torch.empty_like(states)
        grad_initial = torch.zeros_like(initial) if initial is not None else None

        batch_size, sequence_length, features = states.shape

        if states.numel() > 0:
            block_features = 16
            grid = (
                batch_size,
                triton.cdiv(features, block_features),
            )

            with torch.cuda.device(states.device):
                affine_scan_backward_kernel[grid](
                    grad_output,
                    decay,
                    states,
                    initial if initial is not None else states,
                    grad_update,
                    grad_decay,
                    grad_initial if initial is not None else states,
                    T=sequence_length,
                    D=features,
                    BLOCK_T=triton.next_power_of_2(sequence_length),
                    BLOCK_D=block_features,
                    HAS_INITIAL=ctx.has_initial,
                )

        return (
            grad_update.to(ctx.update_dtype),
            grad_decay.to(ctx.decay_dtype),
            grad_initial,
        )


def affine_prefix_sum_triton(
    update: torch.Tensor,
    decay: torch.Tensor,
    initial: torch.Tensor | None = None,
) -> torch.Tensor:
    return AffineScanFunction.apply(update, decay, initial)
