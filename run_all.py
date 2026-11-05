"""
一键运行并验证全部 12 个 LangGraph 学习章节脚本
运行方式：
    uv run python run_all.py
"""

import subprocess
import sys
from pathlib import Path

CHAPTER_FILES = sorted(Path(__file__).parent.glob("[0-1][0-9]_*.py"))


def main():
    print(f"Python 版本: {sys.version}")
    print(f"共发现 {len(CHAPTER_FILES)} 个学习章节文件，开始逐一验证...\n")

    for script in CHAPTER_FILES:
        print(f">>> 正在执行: {script.name}")
        result = subprocess.run([sys.executable, str(script)], check=False)
        if result.returncode != 0:
            print(f"❌ {script.name} 执行失败 (退出码: {result.returncode})")
            sys.exit(result.returncode)
        print()

    print("🎉 全部 12 个 LangGraph 学习章节均已独立执行并验证通过！")


if __name__ == "__main__":
    main()
