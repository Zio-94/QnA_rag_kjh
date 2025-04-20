# src/chatbot_api/modules/vector_store.py
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

# pip install chromadb
import chromadb
from chromadb import Collection, GetResult, QueryResult
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError, IDAlreadyExistsError

logger = logging.getLogger(__name__)

class VectorStore:
    """
    ChromaDB 클라이언트를 감싸는 비동기 래퍼 클래스.
    동기 ChromaDB 호출을 asyncio.to_thread를 사용해 비동기로 처리합니다.
    """
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).resolve()) # 절대 경로 사용 권장
        logger.info(f"Initializing ChromaDB client with path: {self.db_path}")
        try:
            
            chroma_settings = ChromaSettings(
                persist_directory=self.db_path,
                anonymized_telemetry=False # 개인 정보 수집 비활성화 권장
            )
            self.client = chromadb.PersistentClient(path=self.db_path, settings=chroma_settings)
            logger.info("ChromaDB client initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB client: {e}")
            raise # 초기화 실패 시 앱 실행 불가

    async def collection_exists(self, name: str) -> bool:
        """컬렉션 존재 여부를 비동기로 확인합니다."""
        def _sync_list_collections():
            return self.client.list_collections()
        try:
            collections = await asyncio.to_thread(_sync_list_collections)
            return name in [c.name for c in collections]
        except Exception as e:
            logger.error(f"Error checking if collection '{name}' exists: {e}")
            return False # 오류 발생 시 없다고 가정

    async def get_or_create_collection(self, name: str) -> Collection:
        """컬렉션을 가져오거나 없으면 생성합니다 (비동기)."""
        def _sync_get_or_create():
            # get_or_create_collection은 메타데이터, 임베딩 함수 등 추가 인자 가능
            return self.client.get_or_create_collection(name=name)
        try:
            logger.info(f"Getting or creating collection: {name}")
            collection = await asyncio.to_thread(_sync_get_or_create)
            logger.info(f"Collection '{name}' ready.")
            return collection
        except Exception as e:
            logger.error(f"Error getting or creating collection '{name}': {e}")
            raise

    async def delete_collection(self, name: str):
        """컬렉션을 삭제합니다 (비동기)."""
        def _sync_delete():
            logger.warning(f"Attempting to delete collection: {name}")
            self.client.delete_collection(name=name)
            logger.info(f"Collection '{name}' deleted.")
        try:
            await asyncio.to_thread(_sync_delete)
        except ValueError: # 존재하지 않는 컬렉션 삭제 시
             logger.warning(f"Collection '{name}' not found for deletion.")
        except Exception as e:
            logger.error(f"Error deleting collection '{name}': {e}")
            raise

    async def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        """문서, 임베딩, 메타데이터를 컬렉션에 추가합니다 (비동기)."""
        if not ids:
            logger.warning("No documents to add.")
            return
        try:
            collection = await self.get_or_create_collection(collection_name)
            def _sync_add():
                try:
                    collection.add(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                    logger.info(f"Added {len(ids)} documents to collection '{collection_name}'.")
                except IDAlreadyExistsError:
                     logger.warning(f"Some IDs already exist in collection '{collection_name}'. Skipping addition for duplicates.")
                     # 필요시 upsert 사용: collection.upsert(...)
                except Exception as e:
                     logger.error(f"Error during sync add to collection '{collection_name}': {e}")
                     raise # 에러를 다시 발생시켜 상위에서 처리하도록 함

            await asyncio.to_thread(_sync_add)
        except Exception as e:
            # get_or_create_collection 또는 _sync_add 에서 발생한 에러 처리
            logger.error(f"Error adding documents to collection '{collection_name}': {e}")
            # 필요시 예외 재발생 또는 특정 값 반환

    async def search_with_scores(
        self,
        collection_name: str,
        query_embedding: List[float],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        쿼리 임베딩과 유사한 문서를 검색하고 유사도 점수(거리)를 포함하여 반환합니다 (비동기).
        ChromaDB는 거리를 반환하므로, 필요시 유사도로 변환해야 할 수 있습니다.
        :param filter_metadata: 검색 시 적용할 메타데이터 필터 (예: {"source_faq_id": "faq_123"})
        :return: [{'id': ..., 'document': ..., 'metadata': ..., 'distance': ...}, ...]
        """
        try:
            collection = await self.get_or_create_collection(collection_name)
            def _sync_query():
                # include 옵션: 반환할 필드 지정 ('distances', 'metadatas', 'documents')
                # where 옵션: 메타데이터 필터링
                results: QueryResult = collection.query(
                    query_embeddings=[query_embedding], # 리스트로 감싸야 함
                    n_results=k,
                    include=["metadatas", "documents", "distances"], # 거리(distance) 포함
                    where=filter_metadata # 메타데이터 필터 적용
                )
                return results

            query_results = await asyncio.to_thread(_sync_query)

            # 결과 파싱 및 재구성
            parsed_results = []
            if query_results and query_results.get('ids') and query_results['ids'][0]:
                ids = query_results['ids'][0]
                documents = query_results['documents'][0] if query_results.get('documents') else [None] * len(ids)
                metadatas = query_results['metadatas'][0] if query_results.get('metadatas') else [None] * len(ids)
                distances = query_results['distances'][0] if query_results.get('distances') else [None] * len(ids)

                for i, doc_id in enumerate(ids):
                    parsed_results.append({
                        "id": doc_id,
                        "document": documents[i] if documents else None,
                        "metadata": metadatas[i] if metadatas else None,
                        "distance": distances[i] if distances else None # ChromaDB는 거리 반환
                        # 필요시 score (유사도) 로 변환: e.g., score = 1 / (1 + distance) or 1 - distance
                    })
            logger.info(f"Search returned {len(parsed_results)} results for collection '{collection_name}'.")
            return parsed_results

        except NotFoundError:
             logger.warning(f"Collection '{collection_name}' not found for search.")
             return []
        except Exception as e:
            logger.error(f"Error searching collection '{collection_name}': {e}")
            return [] # 오류 시 빈 리스트 반환

    async def get_documents_by_ids(
        self, collection_name: str, ids: List[str]
    ) -> List[Dict[str, Any]]:
        """ID 목록으로 문서를 가져옵니다 (비동기)."""
        if not ids: return []
        try:
            collection = await self.get_or_create_collection(collection_name)
            def _sync_get():
                results: GetResult = collection.get(
                    ids=ids,
                    include=["metadatas", "documents"]
                )
                return results

            get_results = await asyncio.to_thread(_sync_get)

            parsed_results = []
            if get_results and get_results.get('ids'):
                 result_map = {id: i for i, id in enumerate(get_results['ids'])}
                 for id_to_find in ids:
                      if id_to_find in result_map:
                           idx = result_map[id_to_find]
                           parsed_results.append({
                                "id": id_to_find,
                                "document": get_results['documents'][idx] if get_results.get('documents') else None,
                                "metadata": get_results['metadatas'][idx] if get_results.get('metadatas') else None,
                           })
                      else:
                           logger.warning(f"Document with ID '{id_to_find}' not found in get results.")

            logger.info(f"Retrieved {len(parsed_results)} documents by IDs from collection '{collection_name}'.")
            return parsed_results
        except NotFoundError:
             logger.warning(f"Collection '{collection_name}' not found for get_documents_by_ids.")
             return []
        except Exception as e:
            logger.error(f"Error getting documents by IDs from collection '{collection_name}': {e}")
            return []