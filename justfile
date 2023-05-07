VERSION := "2.4.0-alpha.1"

fmt:
  poetry run isort .
  poetry run black .

fmt-docs:
  prettier --write '**/*.md'

check:
  poetry run pyright tests \
    paddle-stubs/_typing/* \
    paddle-stubs/hapi/hub.pyi \
    paddle-stubs/hapi/model.pyi \
    paddle-stubs/hapi/model_summary.pyi \
    paddle-stubs/optimizer/* \
    paddle-stubs/nn/layer/* \
    paddle-stubs/vision/models/* \
    paddle-stubs/vision/transforms/transforms.pyi \
    paddle-stubs/hub.pyi \
    paddle-stubs/linalg.pyi \
    paddle-stubs/regularizer.pyi \
    paddle-stubs/signal.pyi \
    paddle-stubs/sysconfig.pyi

lint:
  poetry run flake8 \
    paddle-stubs/_typing/* \
    paddle-stubs/hapi/hub.pyi \
    paddle-stubs/hapi/model.pyi \
    paddle-stubs/hapi/model_summary.pyi \
    paddle-stubs/optimizer/* \
    paddle-stubs/nn/layer/* \
    paddle-stubs/vision/models/* \
    paddle-stubs/vision/transforms/transforms.pyi \
    paddle-stubs/hub.pyi \
    paddle-stubs/linalg.pyi \
    paddle-stubs/regularizer.pyi \
    paddle-stubs/signal.pyi \
    paddle-stubs/sysconfig.pyi
  poetry run ruff --fix \
    paddle-stubs/_typing/* \
    paddle-stubs/hapi/hub.pyi \
    paddle-stubs/hapi/model.pyi \
    paddle-stubs/hapi/model_summary.pyi \
    paddle-stubs/optimizer/* \
    paddle-stubs/nn/layer/* \
    paddle-stubs/vision/models/* \
    paddle-stubs/vision/transforms/transforms.pyi \
    paddle-stubs/hub.pyi \
    paddle-stubs/linalg.pyi \
    paddle-stubs/regularizer.pyi \
    paddle-stubs/signal.pyi \
    paddle-stubs/sysconfig.pyi

build:
  poetry build

publish:
  poetry publish --build
  git tag "v{{VERSION}}"
  git push --tags
  just clean-builds

clean:
  find . -name "*.pyc" -print0 | xargs -0 rm -f
  rm -rf .pytest_cache/
  rm -rf .mypy_cache/
  find . -maxdepth 3 -type d -empty -print0 | xargs -0 -r rm -r

clean-builds:
  rm -rf build/
  rm -rf dist/
  rm -rf *.egg-info/

ci-install:
  poetry install --no-interaction --no-root

ci-fmt-check:
  poetry run isort --check-only .
  poetry run black --check --diff .
  prettier --check '**/*.md'

ci-lint:
  just lint
