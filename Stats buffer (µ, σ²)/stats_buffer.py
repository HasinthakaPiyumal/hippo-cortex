import torch


class StatsBuffer:
    """
    Stores per-task running statistics (μ, σ²) of hidden states.

    Uses Welford's online algorithm so memory is CONSTANT — we never
    store raw samples, only (count, mean, M2) per task.

    This is the biological "hippocampal compression": the hippocampus
    does not replay every experience verbatim; it compresses episodes
    into statistical summaries that the neocortex can consolidate.

    Usage
    -----
        buf = StatsBuffer(num_tasks=20, hidden_dim=512)

        # During / after each task's forward pass:
        buf.update(task_id=0, H=hidden_states)   # H: (N, D) tensor

        # Later — retrieve statistics:
        mu, var = buf.get_stats(task_id=0)        # both shape (D,)
    """

    def __init__(self, num_tasks: int, hidden_dim: int, device: str = "cpu"):
        """
        Args:
            num_tasks  : total number of continual-learning tasks (e.g. 20)
            hidden_dim : dimensionality of the hidden state vector  (e.g. 512)
            device     : 'cpu' or 'cuda'
        """
        self.num_tasks  = num_tasks
        self.hidden_dim = hidden_dim
        self.device     = device

        # Welford accumulators — shape (T, D), constant memory
        self.register_buffers()

    # ------------------------------------------------------------------
    def register_buffers(self):
        """Allocate zero-filled accumulators for all tasks."""
        T, D = self.num_tasks, self.hidden_dim

        # Number of samples seen so far per task  (T,)
        self.count = torch.zeros(T, dtype=torch.long,  device=self.device)

        # Running mean  (T, D)
        self.mean  = torch.zeros(T, D, dtype=torch.float32, device=self.device)

        # Running sum of squared deviations from mean  (T, D)
        self.M2    = torch.zeros(T, D, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def update(self, task_id: int, H: torch.Tensor):
        """
        Incrementally update statistics for *task_id* with a new batch
        of hidden states.

        Args:
            task_id : int in [0, num_tasks)
            H       : FloatTensor of shape (N, D)  — a batch of hidden states
                      N = number of samples, D = hidden_dim
        """
        if not (0 <= task_id < self.num_tasks):
            raise ValueError(
                f"task_id must be in [0, {self.num_tasks}), got {task_id}"
            )

        H = H.to(self.device).float()

        if H.dim() == 1:
            H = H.unsqueeze(0)          # treat a single vector as batch of 1

        if H.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got H.shape={H.shape}"
            )

        # ── Welford's online algorithm (batch version) ─────────────────
        # Process each sample in the batch one at a time.
        # (For large batches a parallel Welford is faster, but this is
        #  numerically stable and easy to follow.)
        for x in H:                              # x: (D,)
            self.count[task_id] += 1
            n   = self.count[task_id].item()
            delta  = x - self.mean[task_id]
            self.mean[task_id] += delta / n
            delta2 = x - self.mean[task_id]
            self.M2[task_id]   += delta * delta2

    # ------------------------------------------------------------------
    def get_stats(self, task_id: int):
        """
        Return (μ, σ²) for the given task.

        Returns
        -------
            mu  : FloatTensor (D,)  — per-feature mean
            var : FloatTensor (D,)  — per-feature variance
                  (returns zeros if no samples have been seen yet)
        """
        n = self.count[task_id].item()

        if n == 0:
            zeros = torch.zeros(self.hidden_dim, device=self.device)
            return zeros, zeros

        mu  = self.mean[task_id].clone()
        var = self.M2[task_id] / n          # population variance
        return mu, var

    # ------------------------------------------------------------------
    def get_all_stats(self):
        """
        Return statistics for every task as stacked tensors.

        Returns
        -------
            mu  : FloatTensor (T, D)
            var : FloatTensor (T, D)
        """
        mu_list, var_list = [], []
        for t in range(self.num_tasks):
            mu, var = self.get_stats(t)
            mu_list.append(mu)
            var_list.append(var)
        return torch.stack(mu_list), torch.stack(var_list)

    # ------------------------------------------------------------------
    def memory_usage_bytes(self) -> int:
        """How many bytes are used — always constant regardless of data seen."""
        return (
            self.mean.nelement() * 4 +   # float32 = 4 bytes
            self.M2.nelement()   * 4 +
            self.count.nelement() * 8    # int64   = 8 bytes
        )

    # ------------------------------------------------------------------
    def __repr__(self):
        mb = self.memory_usage_bytes() / 1024 / 1024
        return (
            f"StatsBuffer(num_tasks={self.num_tasks}, "
            f"hidden_dim={self.hidden_dim}, "
            f"memory={mb:.3f} MB)"
        )


# ── Quick self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing StatsBuffer ...\n")

    NUM_TASKS  = 20
    HIDDEN_DIM = 512
    BATCH_SIZE = 32

    buf = StatsBuffer(num_tasks=NUM_TASKS, hidden_dim=HIDDEN_DIM)
    print(buf)
    print()

    # Simulate feeding hidden states for each task
    for task_id in range(NUM_TASKS):
        for _ in range(5):                          # 5 batches per task
            H = torch.randn(BATCH_SIZE, HIDDEN_DIM) # fake hidden states
            buf.update(task_id, H)

        mu, var = buf.get_stats(task_id)
        print(
            f"Task {task_id + 1:02d} | "
            f"samples seen: {buf.count[task_id].item():4d} | "
            f"mu[:3]  = {mu[:3].tolist()} | "
            f"var[:3] = {[round(v, 4) for v in var[:3].tolist()]}"
        )

    print(f"\nTotal constant memory used: "
          f"{buf.memory_usage_bytes() / 1024:.1f} KB")
    print("\nAll 20 tasks stats computed successfully!")
