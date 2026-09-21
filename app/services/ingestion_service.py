import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models.postgres_models import (
    ProductCategoryModel,
    StandardModel,
    AmendmentModel,
    CrossReferenceModel,
    CertificationRuleModel
)
from app.schemas.ingestion import BatchIngestRequest, IngestionResponse, IngestionStatsResponse
from app.services.normalizer import DataNormalizer
from app.services.embedding_service import embedding_service
from app.db.neo4j_client import get_neo4j_client
from app.db.vector_store import vector_store_client
from app.db.bm25_index import bm25_index_client

logger = logging.getLogger(__name__)

class IngestionService:
    @staticmethod
    def ingest_batch(db: Session, batch: BatchIngestRequest) -> IngestionResponse:
        """
        Executes the Fan-Out Ingestion Pipeline:
        1. PostgreSQL Master Write (Source of Truth)
        2. Neo4j Graph Write (Derived projection)
        3. ChromaDB Vector Write (Derived projection)
        4. BM25 Keyword Index Write (Derived projection)
        """
        stats_pg = {"categories": 0, "standards": 0, "amendments": 0, "cross_references": 0, "certification_rules": 0}

        # -------------------------------------------------------------
        # STEP 1: PostgreSQL Master Write (Single Source of Truth)
        # -------------------------------------------------------------
        # 1.1 Ingest Product Categories
        for cat in batch.categories:
            db_cat = db.query(ProductCategoryModel).filter(ProductCategoryModel.category_id == cat.category_id).first()
            if not db_cat:
                db_cat = ProductCategoryModel(
                    category_id=cat.category_id,
                    name=cat.name,
                    description=DataNormalizer.clean_text(cat.description or ""),
                    parent_category_id=cat.parent_category_id
                )
                db.add(db_cat)
            else:
                db_cat.name = cat.name
                db_cat.description = DataNormalizer.clean_text(cat.description or "")
                db_cat.parent_category_id = cat.parent_category_id
            stats_pg["categories"] += 1
            # Neo4j Projection
            get_neo4j_client().upsert_category(cat.category_id, cat.name)

        db.flush()

        # 1.2 Ingest Standards & Amendments & Cross-References
        vector_ids = []
        vector_texts = []
        vector_metadatas = []
        bm25_doc_records = []

        for std in batch.standards:
            norm_number = DataNormalizer.normalize_is_number(std.standard_number)
            db_std = db.query(StandardModel).filter(StandardModel.standard_number == norm_number).first()
            
            clean_title = DataNormalizer.clean_text(std.title)
            clean_scope = DataNormalizer.clean_text(std.scope_text)
            clean_abstract = DataNormalizer.clean_text(std.abstract or "")

            if not db_std:
                db_std = StandardModel(
                    standard_number=norm_number,
                    title=clean_title,
                    scope_text=clean_scope,
                    abstract=clean_abstract,
                    year_published=std.year_published,
                    current_version=std.current_version,
                    status=std.status,
                    category_id=std.category_id,
                    sector=std.sector
                )
                db.add(db_std)
            else:
                db_std.title = clean_title
                db_std.scope_text = clean_scope
                db_std.abstract = clean_abstract
                db_std.year_published = std.year_published
                db_std.current_version = std.current_version
                db_std.status = std.status
                db_std.category_id = std.category_id
                db_std.sector = std.sector

            stats_pg["standards"] += 1

            # Neo4j Standard Node Projection
            get_neo4j_client().upsert_standard(
                standard_number=norm_number,
                title=clean_title,
                status=std.status,
                category_id=std.category_id
            )

            # Clear existing amendments for clean overwrite
            db.query(AmendmentModel).filter(AmendmentModel.standard_number == norm_number).delete()
            for amd in std.amendments:
                db_amd = AmendmentModel(
                    standard_number=norm_number,
                    amendment_number=amd.amendment_number,
                    date_issued=amd.date_issued,
                    change_summary=DataNormalizer.clean_text(amd.change_summary)
                )
                db.add(db_amd)
                stats_pg["amendments"] += 1

            # Cross references write
            for xref in std.cross_references:
                target_norm = DataNormalizer.normalize_is_number(xref.target_standard_number)
                existing_ref = db.query(CrossReferenceModel).filter(
                    CrossReferenceModel.source_standard_number == norm_number,
                    CrossReferenceModel.target_standard_number == target_norm,
                    CrossReferenceModel.relationship_type == xref.relationship_type
                ).first()
                if not existing_ref:
                    db_xref = CrossReferenceModel(
                        source_standard_number=norm_number,
                        target_standard_number=target_norm,
                        relationship_type=xref.relationship_type,
                        description=DataNormalizer.clean_text(xref.description or "")
                    )
                    db.add(db_xref)
                    stats_pg["cross_references"] += 1

                # Neo4j Edge Projection
                get_neo4j_client().upsert_relationship(
                    source_number=norm_number,
                    target_number=target_norm,
                    rel_type=xref.relationship_type,
                    description=xref.description
                )

            # Assemble content for Vector & BM25 indexing
            full_text_to_embed = f"Standard Number: {norm_number}\nTitle: {clean_title}\nScope: {clean_scope}\nAbstract: {clean_abstract}"
            vector_ids.append(norm_number)
            vector_texts.append(full_text_to_embed)
            vector_metadatas.append({
                "standard_number": norm_number,
                "title": clean_title,
                "category_id": std.category_id or "",
                "sector": std.sector or "",
                "status": std.status,
                "year_published": std.year_published or 0
            })

            bm25_doc_records.append({
                "standard_number": norm_number,
                "title": clean_title,
                "scope_text": clean_scope,
                "abstract": clean_abstract
            })

        # 1.3 Ingest Certification Rules
        for rule in batch.certification_rules:
            norm_std = DataNormalizer.normalize_is_number(rule.standard_number) if rule.standard_number else None
            db_rule = db.query(CertificationRuleModel).filter(CertificationRuleModel.rule_id == rule.rule_id).first()
            if not db_rule:
                db_rule = CertificationRuleModel(
                    rule_id=rule.rule_id,
                    category_id=rule.category_id,
                    standard_number=norm_std,
                    scheme_type=rule.scheme_type,
                    mandatory_flag=rule.mandatory_flag,
                    description=DataNormalizer.clean_text(rule.description or "")
                )
                db.add(db_rule)
            else:
                db_rule.category_id = rule.category_id
                db_rule.standard_number = norm_std
                db_rule.scheme_type = rule.scheme_type
                db_rule.mandatory_flag = rule.mandatory_flag
                db_rule.description = DataNormalizer.clean_text(rule.description or "")
            stats_pg["certification_rules"] += 1

            # Neo4j Projection
            get_neo4j_client().upsert_certification(
                category_id=rule.category_id,
                standard_number=norm_std,
                scheme_type=rule.scheme_type
            )

        # Commit PostgreSQL transaction
        db.commit()

        # -------------------------------------------------------------
        # STEP 2: Derived Store Writes (Vector DB & BM25 Index)
        # -------------------------------------------------------------
        vector_count = 0
        if vector_ids:
            embeddings = embedding_service.embed_texts(vector_texts)
            vector_store_client.upsert_embeddings(
                ids=vector_ids,
                embeddings=embeddings,
                documents=vector_texts,
                metadatas=vector_metadatas
            )
            vector_count = len(vector_ids)

        bm25_count = 0
        if bm25_doc_records:
            bm25_index_client.build_index(bm25_doc_records)
            bm25_count = len(bm25_doc_records)

        return IngestionResponse(
            status="success",
            message=f"Successfully ingested batch with {stats_pg['standards']} standards into Postgres, Neo4j, Vector DB, and BM25.",
            postgres_inserted=stats_pg,
            neo4j_synced=get_neo4j_client().connected,
            vector_db_indexed_count=vector_count,
            bm25_indexed_count=bm25_count
        )

    @staticmethod
    def rebuild_derived_stores(db: Session) -> Dict[str, Any]:
        """
        Rebuild Routine: Regenerates Neo4j, Vector DB, and BM25 Index directly
        from PostgreSQL (the Single Source of Truth).
        """
        logger.info("Starting rebuild of all derived stores from PostgreSQL...")

        # Purge derived stores
        get_neo4j_client().clear_all()
        vector_store_client.clear()
        bm25_index_client.clear()

        # Fetch all records from Postgres
        categories = db.query(ProductCategoryModel).all()
        standards = db.query(StandardModel).all()
        cross_refs = db.query(CrossReferenceModel).all()
        cert_rules = db.query(CertificationRuleModel).all()

        # 1. Repopulate Neo4j
        for cat in categories:
            get_neo4j_client().upsert_category(cat.category_id, cat.name)

        for std in standards:
            get_neo4j_client().upsert_standard(
                standard_number=std.standard_number,
                title=std.title,
                status=std.status,
                category_id=std.category_id
            )

        for xref in cross_refs:
            get_neo4j_client().upsert_relationship(
                source_number=xref.source_standard_number,
                target_number=xref.target_standard_number,
                rel_type=xref.relationship_type,
                description=xref.description
            )

        for rule in cert_rules:
            get_neo4j_client().upsert_certification(
                category_id=rule.category_id,
                standard_number=rule.standard_number,
                scheme_type=rule.scheme_type
            )

        # 2. Repopulate Vector DB & BM25
        vector_ids = []
        vector_texts = []
        vector_metadatas = []
        bm25_records = []

        for std in standards:
            full_text = f"Standard Number: {std.standard_number}\nTitle: {std.title}\nScope: {std.scope_text}\nAbstract: {std.abstract or ''}"
            vector_ids.append(std.standard_number)
            vector_texts.append(full_text)
            vector_metadatas.append({
                "standard_number": std.standard_number,
                "title": std.title,
                "category_id": std.category_id or "",
                "sector": std.sector or "",
                "status": std.status,
                "year_published": std.year_published or 0
            })
            bm25_records.append({
                "standard_number": std.standard_number,
                "title": std.title,
                "scope_text": std.scope_text,
                "abstract": std.abstract or ""
            })

        if vector_ids:
            embeddings = embedding_service.embed_texts(vector_texts)
            vector_store_client.upsert_embeddings(
                ids=vector_ids,
                embeddings=embeddings,
                documents=vector_texts,
                metadatas=vector_metadatas
            )

        if bm25_records:
            bm25_index_client.build_index(bm25_records)

        return {
            "status": "rebuilt_successfully",
            "postgres_standards_count": len(standards),
            "vector_db_count": vector_store_client.count(),
            "bm25_count": bm25_index_client.count(),
            "neo4j_synced": get_neo4j_client().connected
        }

    @staticmethod
    def get_ingestion_stats(db: Session) -> IngestionStatsResponse:
        """Returns entity count & sync status across all 4 storage layers."""
        pg_standards = db.query(StandardModel).count()
        pg_categories = db.query(ProductCategoryModel).count()
        pg_rules = db.query(CertificationRuleModel).count()
        pg_xrefs = db.query(CrossReferenceModel).count()

        return IngestionStatsResponse(
            postgres={
                "standards_count": pg_standards,
                "categories_count": pg_categories,
                "certification_rules_count": pg_rules,
                "cross_references_count": pg_xrefs
            },
            neo4j=get_neo4j_client().get_stats(),
            vector_db={
                "count": vector_store_client.count(),
                "collection_name": vector_store_client.collection.name if vector_store_client.collection else ""
            },
            bm25={
                "count": bm25_index_client.count()
            }
        )
