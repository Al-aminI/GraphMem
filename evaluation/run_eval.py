#!/usr/bin/env python3
"""
GraphMem vs Naive RAG Evaluation
================================

Choose your backend:
- TURSO: Lightweight, no external servers (run_eval_turso.py)
- NEO4J: Enterprise with Neo4j + Redis (run_eval_neo4j.py)

Uses MultiHopRAG dataset from HuggingFace:
- 2556 multi-hop QA samples
- 609 news article corpus
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="GraphMem Evaluation - Choose Your Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Quick Turso evaluation (default, no servers needed)
  python run_eval.py --backend turso --api-key YOUR_KEY
  
  # Full Turso evaluation  
  python run_eval.py --backend turso --api-key YOUR_KEY --full
  
  # Neo4j + Redis evaluation
  python run_eval.py --backend neo4j --api-key YOUR_KEY \\
    --neo4j-uri neo4j+ssc://xxx.databases.neo4j.io \\
    --neo4j-password YOUR_PASSWORD \\
    --redis-url redis://default:pass@host:port
  
  # Or run directly:
  python run_eval_turso.py --api-key YOUR_KEY --full
  python run_eval_neo4j.py --api-key YOUR_KEY --neo4j-uri ...
        """
    )
    
    parser.add_argument(
        "--backend", 
        choices=["turso", "neo4j"],
        default="turso",
        help="Storage backend to use (default: turso)"
    )
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--corpus-docs", type=int, default=100)
    parser.add_argument("--qa-samples", type=int, default=50)
    parser.add_argument("--full", action="store_true", help="Run full evaluation")
    
    # Neo4j options
    parser.add_argument("--neo4j-uri", help="Neo4j URI")
    parser.add_argument("--neo4j-password", help="Neo4j password")
    parser.add_argument("--redis-url", help="Redis URL")
    
    args = parser.parse_args()
    
    # Build command for subprocess
    if args.backend == "turso":
        script = os.path.join(os.path.dirname(__file__), "run_eval_turso.py")
        cmd = [sys.executable, script]
        
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        if args.full:
            cmd.append("--full")
        else:
            cmd.extend(["--corpus-docs", str(args.corpus_docs)])
            cmd.extend(["--qa-samples", str(args.qa_samples)])
            
    elif args.backend == "neo4j":
        script = os.path.join(os.path.dirname(__file__), "run_eval_neo4j.py")
        cmd = [sys.executable, script]
        
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        if args.neo4j_uri:
            cmd.extend(["--neo4j-uri", args.neo4j_uri])
        if args.neo4j_password:
            cmd.extend(["--neo4j-password", args.neo4j_password])
        if args.redis_url:
            cmd.extend(["--redis-url", args.redis_url])
        if args.full:
            cmd.append("--full")
        else:
            cmd.extend(["--corpus-docs", str(args.corpus_docs)])
            cmd.extend(["--qa-samples", str(args.qa_samples)])
    
    # Run the appropriate script
    print(f"\n🚀 Starting {args.backend.upper()} evaluation...")
    print(f"   Command: {' '.join(cmd)}\n")
    
    import subprocess
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
