# Запуск скрипта venv MediaPipe из процесса вне пакета Claude (WMI): python.exe по настоящему пути, site-packages venv – в sys.path.
# Причина: Claude установлен пакетом MSIX, AppData\Roaming виртуализирован, в pyvenv.cfg – виртуальный путь (CLAUDE.md, «Уроки 21.10»).
# Запуск: wmipy.cmd <скрипт> <аргументы> (найдено Петей 30.10.2026).
import runpy, sys
sys.path.insert(0, r"D:\data\models\mediapipe\venv\Lib\site-packages")
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
