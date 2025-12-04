"""
Milvus数据库初始化脚本
用于创建技能集合并导入数据
"""

import csv
import os
import uuid
from pymilvus import (
    connections,
    utility,
    Collection,
    DataType,
    FieldSchema,
    CollectionSchema
)
from pymilvus import model

# 配置参数
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "state_skill_collection"
DIMENSION = 768  # 嵌入维度


def connect_milvus():
    """连接到Milvus服务"""
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    print(f"已连接到Milvus服务 {MILVUS_HOST}:{MILVUS_PORT}")


def create_collection():
    """创建集合"""
    # 如果集合已存在则删除
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"已删除现有集合: {COLLECTION_NAME}")

    # 创建字段schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="state", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="skill", dtype=DataType.VARCHAR, max_length=10100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20000)
    ]

    schema = CollectionSchema(fields, description="State skill collection")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"已创建集合: {COLLECTION_NAME}")

    return collection


def import_data(collection, csv_path: str):
    """从CSV文件导入数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件未找到: {csv_path}")

    embedding_fn = model.DefaultEmbeddingFunction()
    batch_data = {"ids": [], "vectors": [], "states": [], "skills": [], "texts": []}

    with open(csv_path, mode='r', newline='', encoding='gbk') as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                state = row.get('state', '')
                skill = row.get('skill', '')
                unique_id = str(uuid.uuid4())
                document_text = skill

                vector = embedding_fn.encode_documents([document_text])[0]

                batch_data["ids"].append(unique_id)
                batch_data["vectors"].append(vector.tolist())
                batch_data["states"].append(state)
                batch_data["skills"].append(skill)
                batch_data["texts"].append(document_text)

                # 批量插入
                if len(batch_data["ids"]) >= 100:
                    data = [
                        batch_data["ids"],
                        batch_data["vectors"],
                        batch_data["states"],
                        batch_data["skills"],
                        batch_data["texts"]
                    ]
                    collection.insert(data)
                    print(f"已插入 {len(batch_data['ids'])} 条记录")
                    batch_data = {"ids": [], "vectors": [], "states": [], "skills": [], "texts": []}

            except Exception as e:
                print(f"处理行时出错: {row}, 错误: {str(e)}")
                continue

        # 插入剩余数据
        if batch_data["ids"]:
            data = [
                batch_data["ids"],
                batch_data["vectors"],
                batch_data["states"],
                batch_data["skills"],
                batch_data["texts"]
            ]
            collection.insert(data)
            print(f"已插入最后 {len(batch_data['ids'])} 条记录")

    print("数据导入完成!")


def create_index(collection):
    """创建索引"""
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": 'COSINE',
        "params": {"nlist": 128}
    }
    collection.create_index("vector", index_params, index_name="vector_index")
    print("已创建向量索引")

    collection.load()
    print("集合已加载到内存")


def search_skills(query_text: str, limit: int = 5):
    """搜索相似技能"""
    collection = Collection(COLLECTION_NAME)
    embedding_fn = model.DefaultEmbeddingFunction()

    query_vector = embedding_fn.encode_queries([query_text])[0]

    search_params = {
        "metric_type": 'COSINE',
        "params": {"nprobe": 10}
    }

    results = collection.search(
        [query_vector.tolist()],
        "vector",
        search_params,
        limit=limit,
        output_fields=["state", "skill", "text"]
    )

    formatted_results = []
    for hits in results:
        for hit in hits:
            formatted_results.append({
                "id": hit.id,
                "distance": hit.distance,
                "state": hit.entity.get("state"),
                "skill": hit.entity.get("skill"),
                "text": hit.entity.get("text")
            })

    return formatted_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milvus数据库初始化")
    parser.add_argument("--csv", type=str, help="CSV数据文件路径")
    parser.add_argument("--search", type=str, help="搜索测试查询")
    args = parser.parse_args()

    connect_milvus()

    if args.csv:
        collection = create_collection()
        import_data(collection, args.csv)
        create_index(collection)
    elif args.search:
        results = search_skills(args.search)
        print("\n搜索结果:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 相似度: {result['distance']:.4f}")
            print(f"   技能: {result['skill'][:100]}...")
    else:
        print("使用方法:")
        print("  初始化数据库: python milvus_init.py --csv data/skills.csv")
        print("  搜索测试: python milvus_init.py --search '获取模型元件'")
