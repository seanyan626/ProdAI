#!/usr/bin/env python3
"""
测试配置加载
"""

from configs.config import config

print("=== DashScope 配置测试 ===")
print(f"DashScope API Key: {'已配置' if config.dashscope.api_key else '未配置'}")
print(f"DashScope API URL: {config.dashscope.api_url}")
print(f"DashScope LLM Model: {config.dashscope.llm_model_name}")
print(f"DashScope Embedding Model: {config.dashscope.embedding_model_name}")
print(f"DashScope OCR Model: {config.dashscope.ocr_model_name}")

print("\n=== OpenAI 配置 ===")
print(f"OpenAI API Key: {'已配置' if config.openai.api_key else '未配置'}")
print(f"OpenAI Embedding API Key: {'已配置' if config.openai.embedding_api_key else '未配置'}")
print(f"OpenAI Embedding Name: {config.openai.embedding_name}")
print(f"OpenAI Embedding API URL: {config.openai.embedding_api_url}")

print("\n=== OCR 配置 ===")
print(f"默认OCR模型: {config.ocr.default_model}")
print(f"OCR超时时间: {config.ocr.timeout}秒")
print(f"OCR最大重试次数: {config.ocr.max_retries}")
print(f"阿里云OCR API密钥: {'已配置' if config.ocr.aliyun_api_key else '未配置'}")
print(f"百度OCR API密钥: {'已配置' if config.ocr.baidu_api_key else '未配置'}")