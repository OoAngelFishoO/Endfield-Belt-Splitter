# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import re

GUI_APP_PATH = Path("gui/gui_app.py")
GUI_APP_TEXT = GUI_APP_PATH.read_text(encoding="utf-8")


def _read_app_constant(name: str) -> str:
    pattern = rf'^{name}\s*=\s*"([^"]+)"'
    match = re.search(pattern, GUI_APP_TEXT, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing {name} in {GUI_APP_PATH}")
    return match.group(1)


APP_NAME = _read_app_constant("APP_NAME")
APP_VERSION = _read_app_constant("APP_VERSION")
PACKAGE_NAME = f"{APP_NAME}-v{APP_VERSION}"

block_cipher = None

a = Analysis(
    ['gui/gui_app.py'],
    pathex=[
        'tree_generation',
        'image_generation',
    ],
    binaries=[],
    datas=[
        ('icons', 'icons'),
    ],
    hiddenimports=[
        'NSGA2',
        'GA',
        'conveyer_tree',
        'layout_preview',
        'layout_generator',
        'PIL._tkinter_finder',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=PACKAGE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
