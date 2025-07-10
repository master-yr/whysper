default:
    @just --list

# Run pre-commit hooks on all files, including autoformatting
pre-commit-all:
    pre-commit run --all-files

# Run 'cargo run' on the project
run *ARGS:
    # WGPU_BACKEND=gl cargo run {{ARGS}}
    # WGPU_BACKEND=vulkan VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json cargo run {{ARGS}}
    # RUST_LOG=wgpu_hal=debug WGPU_BACKEND=vulkan VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json cargo run {{ARGS}}
    WGPU_BACKEND=gl cargo run {{ARGS}}
    # cargo run {{ARGS}}

train:
    WGPU_BACKEND=gl RUST_LOG=debug cargo run --release -- train ./dataset

train-cuda:
    WGPU_BACKEND=gl RUST_LOG=debug cargo run --release --features cuda -- train ./dataset

voice:
    RUST_LOG=debug cargo run --release -- recognize

# Run 'bacon' to run the project (auto-recompiles)
watch *ARGS:
	bacon --job run -- -- {{ ARGS }}
