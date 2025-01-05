# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

datas = [('Model/kazakh_signs_resnet_final.resNet.keras', 'Model'), ('Model/labels.txt', 'Model'), ('font.ttf', '.')]
hiddenimports = ['tensorflow', 'cv2', 'cvzone', 'mediapipe', 'PIL', 'numpy']
datas += collect_data_files('mediapipe')
datas += collect_data_files('tensorflow')
datas += collect_data_files('cv2')
hiddenimports += collect_submodules('tensorflow')
hiddenimports += collect_submodules('cv2')
hiddenimports += collect_submodules('mediapipe')


a = Analysis(
    ['realtime_recognition.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide6', 'PySide2'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='KazakhSignLanguageRecognition',
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
