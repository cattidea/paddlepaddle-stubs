# PaddlePaddle Stubs <sup>WIP</sup>

A stubs package as described in [PEP 561](https://peps.python.org/pep-0561/) for [PaddlePaddle](https://github.com/PaddlePaddle/Paddle).

<p align="center">
   <a href="https://python.org/" target="_blank"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/paddlepaddle-stubs?logo=python&style=flat-square"></a>
   <a href="https://pypi.org/project/paddlepaddle-stubs/" target="_blank"><img src="https://img.shields.io/pypi/v/paddlepaddle-stubs?style=flat-square" alt="pypi"></a>
   <a href="https://pypi.org/project/paddlepaddle-stubs/" target="_blank"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/paddlepaddle-stubs?style=flat-square"></a>
   <a href="LICENSE"><img alt="LICENSE" src="https://img.shields.io/github/license/cattidea/paddlepaddle-stubs?style=flat-square"></a>
   <a href="https://github.com/astral-sh/ruff"><img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square"></a>
   <a href="https://gitmoji.dev"><img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67?style=flat-square" alt="Gitmoji"></a>
</p>

目前仅针对 [Pyright](https://github.com/microsoft/pyright) 进行了编写和测试，尚未支持 [Mypy](https://github.com/python/mypy)（Mypy 相对于 Pyright 太难用了，缺失功能太多），因此目前可能仅仅与 VS Code 的 Pylance 扩展一起工作良好～

> [!NOTE]
>
> 本 repo 非运行时库，因此本 repo 中所有示例和单测可能在运行时并不能正常运行，仅仅是为了更加全面和方便对类型进行检查而已。

## Usage

```bash
pip install paddlepaddle-stubs --pre
```

此时再打开编辑器，查看编辑器的类型提示～～～

VS Code 推荐配置：

```jsonc
{
   "python.languageServer": "Pylance",
   // 现在有一些类型在 strict mode 工作的并不是很好，推荐先使用 basic mode
   "python.analysis.typeCheckingMode": "basic",
   "python.analysis.inlayHints.functionReturnTypes": true,
   "python.analysis.inlayHints.variableTypes": true,
}
```

## Status

本项目将会作为 [【Hackathon 6th】Fundable Project 任务一 —— 为 Paddle 框架 API 添加类型提示（Type Hints）](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91FundableProject%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#%E4%B8%80%E4%B8%BA-paddle-%E6%A1%86%E6%9E%B6-api-%E6%B7%BB%E5%8A%A0%E7%B1%BB%E5%9E%8B%E6%8F%90%E7%A4%BAtype-hints)的参考项目，如成功集成至 Paddle，本项目不再单独维护。
