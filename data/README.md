# `data/`

Raw datasets live here. **Contents are gitignored** — only `.gitkeep` and this README are tracked.

## Expected layout

```
data/
├── cifar100/              # torchvision download target for Split-CIFAR100
├── imagenet-r/            # ImageNet-R (download separately — not on torchvision)
└── robot/                 # Stage-2 recordings from the Go2 Edu
    ├── terrain/
    ├── objects/
    ├── proximity/
    └── mapping/
```

## Notes

- Don't commit datasets. If storage becomes a problem, switch to an external drive and symlink subfolders in here — update this README when you do.
- Dataset download scripts live in `../scripts/`.
- Dataset loader code lives in `../hippocortex/data/`.
