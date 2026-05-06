# Endfield Conveyer Belt

明日方舟：终末地 可视化传送带分流计算器

基于：github.com/madSUNitist/Endfield-Conveyer-Belt 的算法

提供：
- 图形界面搜索拓扑方案
- 展示候选结果的误差、成本和预览图

## 运行方式

安装依赖后，直接运行：

```bash
python gui/gui_app.py
```

## 目录说明

- `gui/`：图形界面
- `tree_generation/`：拓扑生成与遗传算法
- `image_generation/`：结构预览图生成
- `icons/`：界面图标资源
- `dist/`：打包后的可执行文件

## 打包

```bash
pyinstaller EndFieldConveyerBelt.spec
```
