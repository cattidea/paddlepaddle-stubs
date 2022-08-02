# PaddlePaddle Stubs <sup>WIP</sup>

A stubs package as described in [PEP 561](https://peps.python.org/pep-0561/) for [PaddlePaddle](https://github.com/PaddlePaddle/Paddle).

<p align="center">
   <a href="https://python.org/" target="_blank"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/paddlepaddle-stubs?logo=python&style=flat-square"></a>
   <a href="https://pypi.org/project/paddlepaddle-stubs/" target="_blank"><img src="https://img.shields.io/pypi/v/paddlepaddle-stubs?style=flat-square" alt="pypi"></a>
   <a href="https://pypi.org/project/paddlepaddle-stubs/" target="_blank"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/paddlepaddle-stubs?style=flat-square"></a>
   <a href="LICENSE"><img alt="LICENSE" src="https://img.shields.io/github/license/ShigureLab/paddlepaddle-stubs?style=flat-square"></a>
   <a href="https://github.com/psf/black"><img alt="black" src="https://img.shields.io/badge/code%20style-black-000000?style=flat-square"></a>
   <a href="https://gitmoji.dev"><img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67?style=flat-square" alt="Gitmoji"></a>
</p>

目前仅针对 [Pyright](https://github.com/microsoft/pyright) 进行了编写和测试，并没有针对 Mypy 进行编写和测试，因此目前可能仅仅与 VS Code 的 Pylance 扩展一起工作良好～

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
   "python.analysis.inlayHints.variableTypes": true
}
```

## Status

起步阶段，最低支持 Python3.7，目前基于 PaddlePaddle 2.3.1 开发，第一个可用版本应当是 2.3.1 版本或者更晚的 2.4 版本～

## Roadmap

See [paddlepaddle-stubs 2.3.1 Roadmap](https://github.com/orgs/ShigureLab/projects/1)

-  [ ] 2.3.1 alpha（public，并发布到 PyPI）：完善全部 P1 级别的类型信息（strict 下工作良好）
-  [ ] 2.3.1 beta：完善全部 P2 级别的类型信息（strict 下工作良好）
-  [ ] 2.3.1 rc：全部类型信息 basic 下工作良好
-  [ ] ... 之后应该是一边升级到新版本（2.4.0），一边继续完善 P3、P4、P5 级别的类型信息（strict 下工作良好）

要添加的还有很多，各个 API 类型信息需要仔细填写（修改自动生成的 `Any`，添加返回值类型），有兴趣的小伙伴可以一起来参与呀～
