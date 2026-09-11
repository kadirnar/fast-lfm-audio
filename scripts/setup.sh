#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p vendor

checkout_dependency() {
    local name="$1" url="$2" revision="$3"
    if [[ ! -d "vendor/$name/.git" ]]; then
        git init "vendor/$name"
        git -C "vendor/$name" remote add origin "$url"
    fi
    if [[ "$(git -C "vendor/$name" rev-parse HEAD 2>/dev/null || true)" != "$revision" ]]; then
        if [[ -n "$(git -C "vendor/$name" status --porcelain)" ]]; then
            printf 'Dependency has local changes: %s\n' "$name" >&2
            exit 1
        fi
        git -C "vendor/$name" fetch --depth 1 origin "$revision"
        git -C "vendor/$name" checkout --detach FETCH_HEAD
    fi
}

# The exact head of huggingface/transformers PR #48249 (add-lfm2-audio).
checkout_dependency transformers https://github.com/kadirnar/transformers.git 843101f38c800b98d49b704215f0b76b92e48e64
checkout_dependency fast-mimi https://github.com/kadirnar/fast-mimi.git f6825daa223d2851d6918561f5fde02cf28d6d9e
checkout_dependency liquid-audio https://github.com/Liquid4All/liquid-audio.git 19e65845923a7f136442c95137884ec61eb386aa

if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.13 .venv
fi
uv pip install --python .venv/bin/python \
    -r requirements-lock.txt \
    -e vendor/transformers -e 'vendor/fast-mimi[fp16]' -e vendor/liquid-audio \
    -e . pytest ruff
uv pip check --python .venv/bin/python
