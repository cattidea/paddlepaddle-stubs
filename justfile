VERSION := "2.3.1-dev.0"

fmt:
  poetry run isort .
  poetry run black .

fmt-docs:
  prettier --write '**/*.md'

lint:
  poetry run pyright paddle-stubs tests

build:
  poetry build

publish:
  touch paddle-stubs/py.typed
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
