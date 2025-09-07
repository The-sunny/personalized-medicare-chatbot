import argparse
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

def get_pc():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment/.env")
    return Pinecone(api_key=api_key)

def create_index(name: str, dim: int, cloud: str, region: str):
    pc = get_pc()
    spec = ServerlessSpec(cloud=cloud, region=region)
    existing = [i.name for i in pc.list_indexes()]
    if name in existing:
        print(f"Index '{name}' already exists.")
        return
    pc.create_index(name=name, dimension=dim, metric="cosine", spec=spec)
    print(f"Created index '{name}' (dim={dim}, metric=cosine, cloud={cloud}, region={region})")

def delete_index(name: str):
    pc = get_pc()
    pc.delete_index(name)
    print(f"Deleted index '{name}'")

def get_index(name: str):
    pc = get_pc()
    return pc.Index(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--index", required=True)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--cloud", default=os.getenv("PINECONE_CLOUD","aws"))
    parser.add_argument("--region", default=os.getenv("PINECONE_REGION","us-east-1"))
    args = parser.parse_args()
    if args.create:
        create_index(args.index, args.dim, args.cloud, args.region)
    elif args.delete:
        delete_index(args.index)
    else:
        print("Specify --create or --delete")
