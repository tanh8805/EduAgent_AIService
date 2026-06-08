import psycopg2
from pgvector.psycopg2 import register_vector

def store_database(postgres_uri: str, document_id: int, chunks: list, embeddings: list) -> bool:
    if len(chunks) != len(embeddings):
        print("Dữ liệu chunks và embeddings không hợp lệ hoặc không khớp độ dài.")
        return False
    try:
        conn = psycopg2.connect(postgres_uri)
        db = conn.cursor()
        register_vector(conn)

        query = """
            INSERT INTO lessons (document_id, chunk_index, content, embedding) 
            VALUES (%s, %s, %s, %s);
        """

        data = []
        for index, (chunk, embed) in enumerate(zip(chunks, embeddings)):
            data.append((document_id, index, chunk, embed))

        db.executemany(query, data)
        conn.commit()
        db.close()
        conn.close()

        print(f"Store {len(chunks)} chunks documents into database.")
        return True
    except Exception as e:
        print(f"Error Store Document into database: {e}")
        return False

def create_document(postgres_uri: str, session_id: int, user_id: int, filename: str):
    try:
        conn = psycopg2.connect(postgres_uri)
        db = conn.cursor()

        query = """
            INSERT INTO documents (session_id, user_id, title)
            VALUES (%s, %s, %s)
            RETURNING id;
        """

        data = [session_id, user_id, filename]
        db.execute(query, data)
        result = db.fetchone()
        doc_id = result[0]
        conn.commit()
        db.close()
        conn.close()

        print(f"Store documents into database.")
        return doc_id
    except Exception as e:
        print(f"Error Store Document into database: {e}")
        return None

def hybrid_search(postgres_uri: str, session_id: int, query_text: str, query_vector: list, top_k: int = 3) -> list:
    if not query_vector or not query_text:
        print("Tham số query_vector hoặc query_text không được để trống.")
        return []

    try:
        conn = psycopg2.connect(postgres_uri)
        db = conn.cursor()
        register_vector(conn)

        query = """
            WITH semantic_branch AS (
                SELECT l.id, ROW_NUMBER() OVER (ORDER BY l.embedding <=> %s ASC) AS rank
                FROM lessons l
                JOIN documents d ON l.document_id = d.id
                WHERE d.session_id = %s
                ORDER BY l.embedding <=> %s ASC
                LIMIT 20
            ),
            keyword_branch AS (
                SELECT l.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', l.content), plainto_tsquery('english', %s)) DESC) AS rank
                FROM lessons l
                JOIN documents d ON l.document_id = d.id
                WHERE d.session_id = %s AND to_tsvector('english', l.content) @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank_cd(to_tsvector('english', l.content), plainto_tsquery('english', %s)) DESC
                LIMIT 20
            )
            SELECT 
                l.chunk_index, 
                l.content,
                COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score
            FROM lessons l
            LEFT JOIN semantic_branch s ON l.id = s.id
            LEFT JOIN keyword_branch k ON l.id = k.id
            WHERE s.id IS NOT NULL OR k.id IS NOT NULL
            ORDER BY rrf_score DESC
            LIMIT %s;
        """

        params = (
            query_vector, session_id, query_vector,
            query_text, session_id, query_text, query_text,
            top_k
        )

        db.execute(query, params)
        results = db.fetchall()

        db.close()
        conn.close()

        output = []
        for row in results:
            output.append({
                "chunk_index": row[0],
                "content": row[1],
                "rrf_score": round(row[2], 5)
            })

        return output

    except Exception as e:
        print(f"Error executing Hybrid Search: {e}")
        return []