@echo off
rem Load the VS2022 developer environment (MSVC + SDK paths)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 goto fail
echo INCLUDE=%INCLUDE%
echo LIB=%LIB%
cd /d craft
C:\Users\nisha\miniconda3\python.exe -c "import _bootstrap; from transformer_vm.evaluator import _load_hull; ext = _load_hull(); print('hull_ext:', ext)"
goto done
:fail
echo vcvarsall failed
:done
