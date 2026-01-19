#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
URDF -> XRDF 转换工具（基于 cuRobo）
===========================================================

功能:
1. 调用 cuRobo 的 XRDF 生成器，将任意 URDF 转为 XRDF
2. 支持指定 base_link / ee_link
3. 作为可达性分析的前置步骤
"""

from __future__ import annotations

import inspect
import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class XRDFConversionResult:
    urdf_path: str
    xrdf_path: str
    base_link: Optional[str] = None
    ee_link: Optional[str] = None


class XRDFConversionError(RuntimeError):
    pass


def _call_with_supported_args(callable_obj: Callable, **kwargs):
    """根据签名过滤参数后调用。"""
    signature = inspect.signature(callable_obj)
    supported = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters and value is not None
    }
    return callable_obj(**supported)


def _build_generator(module):
    """从 cuRobo 模块中解析 XRDF 生成器。"""
    candidate_funcs = [
        "generate_xrdf_from_urdf",
        "urdf_to_xrdf",
        "create_xrdf",
    ]
    for func_name in candidate_funcs:
        if hasattr(module, func_name):
            return getattr(module, func_name)

    candidate_classes = [
        "XRDFGenerator",
        "XrdfGenerator",
        "UrdfToXrdf",
    ]
    for cls_name in candidate_classes:
        if hasattr(module, cls_name):
            cls = getattr(module, cls_name)
            return cls

    return None


def _load_xrdf_module() -> object:
    if importlib.util.find_spec("curobo") is None:
        raise XRDFConversionError("未找到 cuRobo，请先安装 curobo")

    candidate_modules = [
        "curobo.util.xrdf_generator",
        "curobo.util.xrdf_utils",
    ]

    for module_name in candidate_modules:
        if importlib.util.find_spec(module_name) is not None:
            return importlib.import_module(module_name)

    import curobo

    xrdf_modules = [
        name
        for name in (n.name for n in pkgutil.walk_packages(curobo.__path__, prefix="curobo."))
        if "xrdf" in name.lower()
    ]
    for module_name in xrdf_modules:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue

    raise XRDFConversionError("已安装 cuRobo，但未找到 XRDF 生成器模块")


def convert_urdf_to_xrdf(
    urdf_path: str,
    xrdf_path: str,
    base_link: Optional[str] = None,
    ee_link: Optional[str] = None,
) -> XRDFConversionResult:
    """将 URDF 转换为 XRDF。"""
    urdf_path = str(Path(urdf_path).expanduser().resolve())
    xrdf_path = str(Path(xrdf_path).expanduser().resolve())

    module = _load_xrdf_module()

    generator = _build_generator(module)
    if generator is None:
        raise XRDFConversionError("无法在 cuRobo 中找到 XRDF 生成器")

    if inspect.isclass(generator):
        instance = _call_with_supported_args(
            generator,
            urdf_path=urdf_path,
            base_link=base_link,
            ee_link=ee_link,
        )
        if hasattr(instance, "save"):
            _call_with_supported_args(instance.save, xrdf_path=xrdf_path, output_path=xrdf_path)
        elif hasattr(instance, "write"):
            _call_with_supported_args(instance.write, xrdf_path=xrdf_path, output_path=xrdf_path)
        else:
            raise XRDFConversionError("XRDF 生成器没有 save/write 方法")
    else:
        _call_with_supported_args(
            generator,
            urdf_path=urdf_path,
            xrdf_path=xrdf_path,
            output_path=xrdf_path,
            base_link=base_link,
            ee_link=ee_link,
        )

    if not Path(xrdf_path).exists():
        raise XRDFConversionError("XRDF 文件未生成，请检查 cuRobo 输出")

    return XRDFConversionResult(
        urdf_path=urdf_path,
        xrdf_path=xrdf_path,
        base_link=base_link,
        ee_link=ee_link,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="URDF -> XRDF 转换（基于 cuRobo）",
    )
    parser.add_argument("urdf", help="URDF 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出 XRDF 路径")
    parser.add_argument("--base-link", default=None, help="基座链接名")
    parser.add_argument("--ee-link", default=None, help="末端执行器链接名")

    args = parser.parse_args()

    result = convert_urdf_to_xrdf(
        urdf_path=args.urdf,
        xrdf_path=args.output,
        base_link=args.base_link,
        ee_link=args.ee_link,
    )

    print("[OK] XRDF 生成成功:")
    print(f"  URDF: {result.urdf_path}")
    print(f"  XRDF: {result.xrdf_path}")


if __name__ == "__main__":
    main()
