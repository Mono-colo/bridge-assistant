# BridgeFix 231 自动过关器

这是一个 Windows 本地自动器。启动后，它会寻找正在运行的 `BridgeFix 231`，识别当前题面，与游戏安装目录中的内置题库匹配，输入答案并进入下一题。默认模式不需要手动选择窗口、采集、识别或点击“求解”。

## 支持范围

- 全部 10 个难度：`VERY_EASY`、`EASY`、`MEDIUM`、`BIG`、`HUGE` 及对应的 5 个 `*_TRICKY` 难度。
- 标准数桥规则：岛屿度数、最多双桥、禁止交叉和全图连通。
- `MagicLink`：两条指定连接桥数相等；游戏会同步另一条连接，自动器只操作一个代表边。
- `BorderHint`：指定整行或整列的桥数总和满足 `=`、`>=` 或 `<=`。
- 完成当前难度后，根据游戏的 `solved.dat` 进度切换到下一未完成难度。

题库通过 `UnityPy` 在运行时读取游戏自身的 `resources.assets`。项目不会复制或修改游戏题库。

## 准备

1. 安装并启动 Steam 版 `BridgeFix 231`。
2. 保持游戏窗口可见，并确保它与自动器使用相同的 Windows 权限级别。
3. 安装 Python 3.10+ 和项目依赖：

```powershell
python -m pip install -r requirements.txt
```

## 运行

已构建的单文件版本不需要另装 Python，启动游戏后双击：

```text
dist\bridge_assistant.exe
```

也可以在 PowerShell 中给 EXE 传入下文相同的 `--once`、`--levels`、`--difficulty`、`--resources-path` 或 `--manual` 参数。

默认持续自动过关：

```powershell
python -m bridge_assistant
```

也可以双击 `run_bridge.bat`，或使用安装后的 `bridge-assistant` 命令。

验收和定向运行参数：

```powershell
# 解一题，确认胜利并进入下一题后退出
python -m bridge_assistant --once

# 连续解 3 题
python -m bridge_assistant --levels 3

# 切到指定难度并连续解 3 题
python -m bridge_assistant --levels 3 --difficulty VERY_EASY_TRICKY

# 显式指定游戏目录、可执行文件或 resources.assets
python -m bridge_assistant --once --resources-path "C:\Program Files (x86)\Steam\steamapps\common\BridgeFix 231"

# 打开旧的手动识别 GUI
python -m bridge_assistant --manual
```

运行中按 `Esc` 可停止。`pyautogui` 的左上角故障保护也保持启用。

## 自动流程

每题按以下闭环执行：

1. 同时按精确标题和进程名绑定游戏窗口并激活前台。
2. 截取客户区，定位岛屿，并以布局、数字和候选连接唯一匹配内置题目。
3. 校验官方答案及标准规则、MagicLink 和 BorderHint。
4. 对所有候选走廊右键清零；MagicLink 联通组只清理一个代表边。
5. 对答案中的每条代表边左键精确点击 1 或 2 次，并逐次检查桥走廊变化。
6. 任一输入未确认时停止本轮，整题清零后重试；每题最多尝试 3 轮，不会追加左键盲重试。
7. 通过 `solved.dat` 新增题号确认胜利；已有胜利画面也能在重启后恢复。
8. 胜利后只点击一次 Next，至少等待 1.5 秒，并确认题号或布局已经变化。

窗口尺寸或 DPI 变化后，每道新题都会重新采集和计算坐标。自动器依赖屏幕上的实际像素和系统鼠标，因此置顶遮挡、独占全屏、UAC 安全桌面或权限不一致会导致安全停止。

## 测试

```powershell
python -m pip install pytest
python -m pytest -q
python scripts/bridge_assistant_launcher.py --self-test
dist\bridge_assistant.exe --self-test
```

测试覆盖视觉定位、题库解析和布局匹配、标准及 Tricky 约束求解、MagicLink 操作去重、有限整题重试、胜利恢复、单次 Next 和窗口坐标映射。

## 构建

发布脚本会构建 wheel 和单文件 EXE，并让 EXE 从已安装的游戏中实际解析 5000 道题完成自检：

```powershell
python -m pip install build pyinstaller
powershell -File scripts\build_release.ps1
```

旧的通用桌面鼠标验收仍可运行：

```powershell
python scripts/run_desktop_e2e.py
```
