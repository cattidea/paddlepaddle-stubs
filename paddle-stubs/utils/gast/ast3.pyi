from __future__ import annotations

from typing import Any

from . import gast as gast
from .astn import AstToGAst as AstToGAst
from .astn import GAstToAst as GAstToAst

class Ast3ToGAst(AstToGAst):
    def visit_ExtSlice(self, node: Any): ...
    def visit_Index(self, node: Any): ...
    def visit_Module(self, node: Any): ...
    def visit_Num(self, node: Any): ...
    def visit_Ellipsis(self, node: Any): ...
    def visit_Str(self, node: Any): ...
    def visit_Bytes(self, node: Any): ...
    def visit_FunctionDef(self, node: Any): ...
    def visit_AsyncFunctionDef(self, node: Any): ...
    def visit_For(self, node: Any): ...
    def visit_AsyncFor(self, node: Any): ...
    def visit_With(self, node: Any): ...
    def visit_AsyncWith(self, node: Any): ...
    def visit_Call(self, node: Any): ...
    def visit_NameConstant(self, node: Any): ...
    def visit_arguments(self, node: Any): ...
    def visit_Name(self, node: Any): ...
    def visit_arg(self, node: Any): ...
    def visit_ExceptHandler(self, node: Any): ...
    def visit_comprehension(self, node: Any): ...

class GAstToAst3(GAstToAst):
    def visit_Subscript(self, node: Any): ...
    def visit_Module(self, node: Any): ...
    def visit_Constant(self, node: Any): ...
    def visit_Name(self, node: Any): ...
    def visit_ExceptHandler(self, node: Any): ...
    def visit_Call(self, node: Any): ...
    def visit_ClassDef(self, node: Any): ...
    def visit_FunctionDef(self, node: Any): ...
    def visit_AsyncFunctionDef(self, node: Any): ...
    def visit_For(self, node: Any): ...
    def visit_AsyncFor(self, node: Any): ...
    def visit_With(self, node: Any): ...
    def visit_AsyncWith(self, node: Any): ...
    def visit_arguments(self, node: Any): ...

def ast_to_gast(node: Any): ...
def gast_to_ast(node: Any): ...
