import logging
from typing import Dict, List, Any
from lightrag.lightrag import LightRAG

from core.memory.config import memory_config

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler('cross_namespace_linker.log'))

class CrossNamespaceLinker:
    """
    Handles post-processing to link user entities to global entities.
    Removes GLOBAL_REF stubs and replaces them with direct cross-label edges.
    """

    async def link_and_cleanup(self, user_rag: LightRAG, global_rag: LightRAG, user_id: str) -> Dict[str, int]:
        """
        Executes the cross-namespace linking logic.
        1. Finds all GLOBAL_REF stubs in the user's workspace.
        2. Links connected user entities to the actual global entities.
        3. Deletes the stubs.
        """
        logger.info(f"Cross-namespace linking for user: {user_id}")
        user_workspace = f"user_{user_id}"
        global_workspace = "global"
        
        # Use simple string checks as fallback if direct attribute access fails or is different in version
        driver = user_rag.chunk_entity_relation_graph._driver
        
        links_created = 0
        stubs_removed = 0
        
        async with driver.session(database=memory_config.neo4j_database) as session:
            # 1. Find stubs in the user space
            query_find_stubs = f"""
            MATCH (n:`{user_workspace}`)
            WHERE n.entity_type = 'globalref'
            RETURN n.entity_id as stub_name
            """
            result = await session.run(query_find_stubs)
            stub_names = [record["stub_name"] for record in await result.data()]
            logger.info(f"Found {len(stub_names)} stubs in user workspace.")
            logger.info(f"Stub names: {stub_names}")

            for name in stub_names:
                logger.info(f"Processing global reference: {name}")

                # 2. Check if the entity exists in the global space
                query_check_global = f"""
                MATCH (g:`{global_workspace}`)
                WHERE g.entity_id = $name
                RETURN g
                """
                global_exists = await session.run(query_check_global, name=name)
                if not await global_exists.single():
                    logger.warning(f"Global reference '{name}' not found in global namespace. Skipping link.")
                    continue

                logger.info(f"Global reference '{name}' found in global namespace.")
                
                # 3. Transfer relationships from stub to global entity
                query_link = f"""
                MATCH (u:`{user_workspace}`)-[r]->(stub:`{user_workspace}`)
                WHERE stub.entity_id = $name AND stub.entity_type = 'globalref'
                MATCH (global:`{global_workspace}`)
                WHERE global.entity_id = $name
                CREATE (u)-[nr:CROSS_REF]->(global)
                SET nr = properties(r), 
                    nr.original_type = type(r),
                    nr.user_id = $user_id,
                    nr.workspace = $user_workspace
                RETURN count(nr) as created
                """
                
                link_res = await session.run(query_link, name=name, user_id=user_id, user_workspace=user_workspace)
                rec = await link_res.single()
                created_count = rec["created"] if rec else 0
                links_created += created_count
                
                logger.info(f"Created {created_count} links for global reference '{name}'.")
                
                # 4. Delete the stub
                query_delete_stub = f"""
                MATCH (stub:`{user_workspace}`)
                WHERE stub.entity_id = $name AND stub.entity_type = 'globalref'
                DETACH DELETE stub
                """
                await session.run(query_delete_stub, name=name)
                stubs_removed += 1
                
                logger.info(f"Deleted stub for global reference '{name}'.")
                
        logger.info(f"Cross-namespace linking for {user_id}: {links_created} links created, {stubs_removed} stubs removed.")
        return {"links_created": links_created, "stubs_removed": stubs_removed}

    async def resolve_cross_refs(self, user_id: str, rag: LightRAG) -> List[Dict[str, Any]]:
        """
        Retrieves all cross-namespace references for a user.
        Used during query merging to provide global context for user entities.
        """
        user_workspace = f"user_{user_id}"
        driver = rag.chunk_entity_relation_graph._driver
        
        async with driver.session(database=memory_config.neo4j_database) as session:
            # Note: global entities might be in 'global' workspace or just be generalized nodes
            # Assuming they are in 'global' workspace label as per design
            query = f"""
            MATCH (u:`{user_workspace}`)-[r:CROSS_REF]->(g)
            RETURN u.entity_id as source_entity, 
                   g.entity_id as global_entity, 
                   g.entity_type as global_type, 
                   g.description as global_description,
                   r.original_type as relationship_type
            """
            result = await session.run(query)
            return await result.data()
