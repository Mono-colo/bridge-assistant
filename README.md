# BridgeFix 231 自动过关器

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

这是一个 Windows 本地自动器。启动后，它会寻找正在运行的 `BridgeFix 231`，识别当前题面，与游戏安装目录中的内置题库匹配，输入答案并进入下一题。默认模式不需要手动选择窗口、采集、识别或点击“求解”。

## 声明与免责

- 本项目开源，面向个人学习、研究与技术交流。**代码部分的使用条款以 [MIT License](LICENSE) 为准。**
- 所有题库数据均在运行时从**使用者本地已安装的游戏文件**（`resources.assets`）中读取。**本仓库不包含、不复制、不分发任何游戏资源、题库、美术素材或游戏本体。**
- 本项目与游戏开发商、发行商及 Steam 不存在任何关联，未获得其授权、认可或赞助。游戏中出现的所有名称、商标与内容，版权均归其各自权利人所有。
- 自动化操作可能违反游戏的用户协议（EULA）与服务条款，并可能导致账号受限或封禁。请在使用前自行确认相关条款并评估风险，一切后果由使用者自行承担。
- 软件按“原样”提供，不附带任何形式的明示或默示担保。

## 支持范围

- 全部 10 个难度：`VERY_EASY`、`EASY`、`MEDIUM`、`BIG`、`HUGE` 及对应的 5 个 `*_TRICKY` 难度。
- 标准数桥规则：岛屿度数、最多双桥、禁止交叉和全图连通。
- `MagicLink`：两条指定连接桥数相等；游戏会同步另一条连接，自动器只操作一个代表边。
- `BorderHint`：指定整行或整列的桥数总和满足 `=`、`>=` 或 `<=`。
- 完成当前难度后，仅在难度菜单画面明确显示 `100.0%` 时切换到下一未完成难度。

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

也可以在 PowerShell 中给 EXE 传入下文相同的 `--once`、`--levels`、`--difficulty`、`--resources-path`、`--log-dir` 或 `--manual` 参数。

默认持续自动过关：

```powershell
python -m bridge_assistant
```

也可以双击 `run_bridge.bat`，或使用安装后的 `bridge-assistant` 命令。

验收和定向运行参数：

```powershell
# 解一题，确认胜利并进入下一题后退出
python -m bridge_assistant --once

# 新完成 3 题；启动时恢复的已有胜利页不计入数量
python -m bridge_assistant --levels 3

# 切到指定难度并连续解 3 题
python -m bridge_assistant --levels 3 --difficulty VERY_EASY_TRICKY

# 显式指定游戏目录、可执行文件或 resources.assets
python -m bridge_assistant --once --resources-path "C:\Program Files (x86)\Steam\steamapps\common\BridgeFix 231"

# 覆盖持久日志目录
python -m bridge_assistant --levels 3 --log-dir "D:\BridgeAssistantLogs"

# 打开旧的手动识别 GUI
python -m bridge_assistant --manual
```

显式指定 `--log-dir` 时，运行日志和 `last-run.json` 都保存在该目录中。

运行中按 `Esc` 可停止。`pyautogui` 的左上角故障保护也保持启用。同一用户只能运行一个自动过关实例，重复启动会立即退出，避免多个进程争抢鼠标。默认日志写入 `%LOCALAPPDATA%\BridgeAssistant\runs`，保留最近 10 次运行；`%LOCALAPPDATA%\BridgeAssistant\last-run.json` 始终记录最近一次运行的状态、题号、计数和错误。等待期间每 5 秒记录一次心跳，因此双击 EXE 后即使窗口关闭也能查到退出原因。

## 自动流程

每题按以下闭环执行：

1. 同时按精确标题和进程名绑定游戏窗口并激活前台。
2. 截取客户区，定位岛屿，并以布局、数字和候选连接唯一匹配内置题目。
3. 校验官方答案及标准规则、MagicLink 和 BorderHint。
4. 对所有候选走廊右键清零；MagicLink 联通组只清理一个代表边。
5. 对答案中的每条代表边左键精确点击 1 或 2 次，并逐次检查桥走廊变化。
6. 任一输入未确认时停止本轮，整题清零后重试；每题最多尝试 3 轮，不会追加左键盲重试。
7. 仅通过游戏自己的 `solved.dat` 新增当前题号确认新胜利；普通按钮亮度不会被当作胜利，恢复的已有胜利页也不会增加完成计数。
8. 胜利后点击 Next，至少等待 1.5 秒并确认题号或布局变化；只有旧题身份不变且非 Next 动画区域连续稳定至少 0.75 秒才允许重试，最多点击 3 次。
9. 每题胜利后打开难度菜单读取画面百分比；只有当前难度明确显示 `100.0%` 才切到菜单中下一项未满难度，否则返回关卡并点击原生 Next。

窗口尺寸或 DPI 变化后，每道新题都会重新采集和计算坐标。自动器依赖屏幕上的实际像素和系统鼠标，因此置顶遮挡、独占全屏、UAC 安全桌面或权限不一致会导致安全停止。

## 测试

```powershell
python -m pip install pytest
python -m pytest -q
python scripts/bridge_assistant_launcher.py --self-test
dist\bridge_assistant.exe --self-test
```

测试覆盖视觉定位、题库解析和布局匹配、标准及 Tricky 约束求解、MagicLink 操作去重、有限整题重试、精确进度差集、胜利恢复、Next 有界重试、持久日志和窗口坐标映射。

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

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
