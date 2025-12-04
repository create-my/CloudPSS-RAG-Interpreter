"""
测试客户端
用于批量测试任务执行
"""

import requests
import time
import sys
import csv
import argparse
from typing import List, Tuple, Optional


class TaskClient:
    """任务执行客户端"""

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url

    def execute_task(self, task: str) -> Tuple[Optional[str], int]:
        """
        执行单个任务

        Args:
            task: 任务描述

        Returns:
            (未完成的任务描述或None, 对话轮数)
        """
        url = f"{self.server_url}/chat"

        try:
            response = requests.get(url, params={'message': task}, timeout=300)
            response.raise_for_status()
            json_data = response.json()

            if not json_data:
                print("未返回任何内容")
                return "未返回任何内容", 0

            # 检查是否有代码生成
            code_results = []
            for message in json_data:
                if message.get('role') == 'assistant' and message.get('type') == 'code':
                    code_results.append(message)
                elif message.get('role') == 'assistant' and message.get('type') == 'message':
                    content = message.get('content', '')
                    if ("已完成" in content or "已经完成" in content) and code_results:
                        print(f"任务完成: {task[:50]}...")
                        return None, len(json_data)

            if not code_results:
                return "没有生成代码", 0

            print(f"任务未完成: {task[:50]}...")
            return task, len(json_data)

        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return "请求失败", 0

    def run_tasks_from_csv(self, csv_path: str, encoding: str = 'utf-8') -> dict:
        """
        从CSV文件批量执行任务

        Args:
            csv_path: CSV文件路径
            encoding: 文件编码

        Returns:
            执行结果统计
        """
        tasks = []
        with open(csv_path, mode='r', newline='', encoding=encoding) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    tasks.append((row[0], row[1]))

        print(f"加载了 {len(tasks)} 个任务")

        results = {
            "total": len(tasks),
            "completed": 0,
            "failed": 0,
            "failed_tasks": [],
            "conversation_counts": []
        }

        for name, task in tasks:
            unfinished, conv_count = self.execute_task(task)
            results["conversation_counts"].append(conv_count)

            # 重试逻辑
            retry_count = 0
            while unfinished == "未返回任何内容" and retry_count < 3:
                print("等待10秒后重试...")
                time.sleep(10)
                unfinished, conv_count = self.execute_task(task)
                retry_count += 1

            while unfinished == "没有生成代码" and retry_count < 3:
                print("重新请求，强调生成代码...")
                task_with_emphasis = task + "，请生成代码！"
                unfinished, conv_count = self.execute_task(task_with_emphasis)
                retry_count += 1

            if unfinished:
                results["failed"] += 1
                results["failed_tasks"].append({"name": name, "task": task, "reason": unfinished})
            else:
                results["completed"] += 1

        return results


def main():
    parser = argparse.ArgumentParser(description="CloudPSS任务测试客户端")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="服务器地址")
    parser.add_argument("--csv", type=str, help="任务CSV文件路径")
    parser.add_argument("--task", type=str, help="单个任务")
    parser.add_argument("--encoding", type=str, default="utf-8", help="CSV文件编码")
    args = parser.parse_args()

    client = TaskClient(args.server)

    if args.csv:
        results = client.run_tasks_from_csv(args.csv, args.encoding)

        print("\n" + "=" * 50)
        print("执行结果统计:")
        print(f"  总任务数: {results['total']}")
        print(f"  完成数: {results['completed']}")
        print(f"  失败数: {results['failed']}")

        if results['conversation_counts']:
            avg_conv = sum(results['conversation_counts']) / len(results['conversation_counts'])
            print(f"  平均对话轮数: {avg_conv:.2f}")

        completion_rate = results['completed'] / results['total'] * 100 if results['total'] > 0 else 0
        print(f"  完成率: {completion_rate:.1f}%")

        if results['failed_tasks']:
            print("\n失败任务列表:")
            for item in results['failed_tasks'][:10]:
                print(f"  - {item['name']}: {item['reason']}")

    elif args.task:
        result, conv_count = client.execute_task(args.task)
        if result:
            print(f"任务未完成: {result}")
        else:
            print("任务完成!")
        print(f"对话轮数: {conv_count}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
