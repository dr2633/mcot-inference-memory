from setuptools import setup, find_packages

setup(
    name="mcot_grpo_ipo",
    version="0.2.0",
    author="Derek Rosenzweig",
    author_email="derek.rosenzweig1@gmail.com",
    description="Inference Preference Optimization: Conditioning CoT and GRPO with Memory",
    long_description=open("documentation/3-2/3-2-README-roadmap.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dr2633/mCoT-GRPO-IPO",
    packages=find_packages(),  # Automatically find packages in the repository
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.22.0",
        "datasets>=2.14.0",
        "tiktoken>=0.4.0",
        "einops>=0.6.0",
        "transformers_stream_generator>=0.1.0",
        "faiss-cpu>=1.7.4",
        "trl>=0.7.9",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "jsonlines>=3.1.0",
        "matplotlib>=3.7.0",
        "sentence-transformers>=2.2.0"
    ],
    entry_points={
        "console_scripts": [
            # Baseline CoT and mCoT evaluation
            "run_baseline_cot=cot.run_baseline_cot:main",
            "run_memory_cot=cot.run_memory_cot:main",

            # Training and Evaluation
            "train_rl_memory=rl.train_rl_memory:main",
            "evaluate_cot_vs_mcot=scripts.evaluate_cot_vs_mcot:main",
            "evaluate_rl=scripts.evaluate_rl:main",
            "user_preference=scripts.user_preference:main",

            # FAISS Memory Management Utilities
            "build_faiss_index=memory.retrieval_faiss:build_index",
            "update_faiss_index=memory.retrieval_faiss:update_index",
            "retrieve_memory=memory.retrieval_faiss:retrieve_memory",
            "manage_memory=memory.manage_memory:main",  # Summarization + memory consolidation

            # Debugging and testing utilities
            "test_cot=tests.test_cot:main",
            "test_memory=tests.test_memory:main",
            "test_rl=tests.test_rl:main",
            "test_integration=tests.test_integration:main",
        ]
    },
)
